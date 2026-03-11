"""
Loop principal de inferência em tempo real com calibração per-subject.

V2: ONNXFaceMeshBackend, best_model.onnx (19 features), inference_config.json.
V3: FIX-RT-3 — HeadPoseNeutralizer (C33): zera 4 features de head pose
    para tornar o modelo agnóstico a pose em produção. Toggleável via
    --no-neutralize-pose para testes A/B com dados de lab.

Pipeline: picamera2/cv2 -> ONNXFaceMesh -> features -> scale_features (JSON)
          -> HeadPoseNeutralizer -> MLP V3 -> Safe/Danger.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

try:
    from picamera2 import Picamera2

    HAS_PICAMERA2 = True
except ImportError:
    Picamera2 = None
    HAS_PICAMERA2 = False

from .feature_extractor_rt import (
    DummyBackend,
    ONNXFaceMeshBackend,
    RealTimeFeatureExtractor,
)
from .model_loader import (
    load_best_model,
    predict_fatigue,
    scale_features,
)
from .subject_calibrator_rt import (
    CalibrationConfig,
    CalibratedFrame,
    RTSubjectCalibrator,
)
from .window_factory_rt import OnlineWindowFactory, RTWindowConfig
from . import guardrails
from .guardrails import (
    AlertLevel,
    BehaviorGuardRails,
    validate_calibration,
)
from .reflection import (
    AutoRecalibrationManager,
    DriftReflector,
    DriftStatus,
    PredictionReflector,
)


# ── FIX-RT-3: HeadPoseNeutralizer (C33) ─────────────────────────────────────


class HeadPoseNeutralizer:
    """
    FIX-RT-3: Neutraliza features de head pose para deploy em produção.

    Zera as 4 features de pose no vetor escalonado (z-score=0 = centro da
    distribuição de treino), tornando a classificação 100% dependente das
    15 features oculares/blink.

    ORDEM CRÍTICA: aplicar DEPOIS do SelectiveScaler para que o valor
    final seja exatamente 0.0.  Se aplicado ANTES, o scaler computa
    (0 - mean) / std ≠ 0, quebrando a neutralização.

    Toggleável via ``enabled`` para testes A/B (--no-neutralize-pose).

    Features neutralizadas (índices 4-7 em FEATURE_NAMES_21):
        pitch_mean, pitch_std, yaw_std, roll_std

    Ref: SALTE_COMPLETE_HISTORY §Bug #2, FeatureFactoryV4.md §4.2.
    Constraint: C33 — Head Pose Neutralization.
    """

    POSE_FEATURE_NAMES = {"pitch_mean", "pitch_std", "yaw_std", "roll_std"}

    def __init__(self, feature_names: list[str], enabled: bool = True) -> None:
        self.enabled = enabled
        self._pose_indices = [
            i for i, name in enumerate(feature_names)
            if name in self.POSE_FEATURE_NAMES
        ]
        if enabled and self._pose_indices:
            print(
                f"[neutralizer] Zeroing {len(self._pose_indices)} pose "
                f"features at indices {self._pose_indices}"
            )
        elif enabled and not self._pose_indices:
            print(
                "[neutralizer] WARNING: nenhuma feature de pose encontrada "
                "em feature_names — neutralizer é no-op"
            )

    def neutralize(self, vec: np.ndarray) -> np.ndarray:
        """Zera features de pose in-place (cópia). Retorna vetor modificado."""
        if not self.enabled:
            return vec
        v = vec.copy()
        for idx in self._pose_indices:
            v[idx] = 0.0
        return v


# ── PiCamera2 capture wrapper ───────────────────────────────────────────────


class PiCamera2Capture:
    """Wrapper picamera2 com interface compatível com cv2.VideoCapture."""

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
    ) -> None:
        if not HAS_PICAMERA2:
            raise RuntimeError(
                "picamera2 não está instalado. "
                "Instale com `sudo apt install python3-picamera2` "
                "ou use --camera-index para webcam USB."
            )
        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": resolution, "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.cam.configure(config)
        self.cam.start()
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray]:
        if not self._opened:
            return False, np.empty(0)
        frame_rgb = self.cam.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return True, frame_bgr

    def release(self) -> None:
        if self._opened:
            self.cam.stop()
            self._opened = False


# ── Main realtime loop ───────────────────────────────────────────────────────


def run_realtime(
    checkpoint_path: Union[Path, str],
    config_path: Union[Path, str, None],
    *,
    detector_path: Union[Path, str, None] = None,
    mesh_path: Union[Path, str, None] = None,
    camera_index: int = 0,
    use_picamera: bool = False,
    threshold_override: float | None = None,
    warmup_sec: float = 120.0,
    min_warmup_sec: float = 30.0,
    fps: int = 30,
    headless: bool = False,
    debug: bool = False,
    neutralize_pose: bool = True,
) -> None:
    """
    Loop realtime com calibração per-subject.

    Fases:
    1. WARM-UP: coleta frames por warmup_sec segundos
    2. CALIBRATED: inferência com Safe/Danger overlay

    Args:
        neutralize_pose: Se True (default), zera as 4 features de head pose
            após o scaler, tornando a predição agnóstica a pose (C33).
            Desativar com --no-neutralize-pose para testes com dados de lab.
    """
    model_dir = Path(checkpoint_path).parent
    if config_path is None:
        config_path = model_dir / "inference_config.json"
    if detector_path is None:
        detector_path = model_dir / "blazeface_detector.onnx"
    if mesh_path is None:
        mesh_path = model_dir / "face_mesh_landmark.onnx"

    model, config = load_best_model(checkpoint_path, config_path=config_path)
    threshold = threshold_override if threshold_override is not None else config.threshold
    feature_names = config.feature_names

    # ── Guardrails + Reflection: carregar training_stats e inicializar ────
    with open(str(config_path), encoding="utf-8") as _f:
        _raw_config = json.load(_f)
    training_stats = _raw_config.get("training_stats", {})

    behavior_guard = BehaviorGuardRails()
    drift_reflector = DriftReflector(training_stats)
    pred_reflector = PredictionReflector()
    recal_manager = AutoRecalibrationManager()
    print("[init] Guardrails + Reflection: initialized")
    # ─────────────────────────────────────────────────────────────────────

    # ── FIX-RT-3: inicializar neutralizer ────────────────────────────────
    neutralizer = HeadPoseNeutralizer(
        feature_names=feature_names,
        enabled=neutralize_pose,
    )
    if not neutralize_pose:
        print("[init] HeadPoseNeutralizer: DISABLED (--no-neutralize-pose)")
    # ─────────────────────────────────────────────────────────────────────

    print(f"[init] Scaler: JSON (inference_config)")
    print(f"[init] Threshold: {threshold} ({len(feature_names)} features)")

    try:
        backend = ONNXFaceMeshBackend(
            str(detector_path),
            str(mesh_path),
            min_face_score=0.5,
        )
        print("[init] Backend: ONNXFaceMeshBackend (BlazeFace + FaceMesh ONNX)")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[init] Backend fallback: DummyBackend ({e})")
        backend = DummyBackend()

    extractor = RealTimeFeatureExtractor(backend)
    calibrator = RTSubjectCalibrator(
        CalibrationConfig(fps=fps, search_sec=warmup_sec)
    )
    window_factory = OnlineWindowFactory(RTWindowConfig(fps=fps))

    if use_picamera:
        cap = PiCamera2Capture(fps=fps)
        print("[init] Camera: picamera2 (AI Camera)")
    else:
        cap = cv2.VideoCapture(camera_index)
        print(f"[init] Camera: cv2.VideoCapture({camera_index})")

    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera")

    print(f"[init] Headless: {headless}")
    print(
        f"[init] Warm-up: {warmup_sec}s "
        f"(press 'c' to calibrate early, 'q' to quit)"
    )

    frame_interval = 1.0 / max(fps, 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            feats = extractor.process_frame(frame)

            raw_text = (
                f"EAR:{feats.ear_avg:.3f} "
                f"MAR:{feats.mar:.3f} "
                f"Face:{int(feats.face_detected)}"
            )

            if not headless:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 0
                time.sleep(frame_interval)

            if not calibrator.is_calibrated:
                calibrated = calibrator.push(feats)
                progress = calibrator.warmup_progress

                if key == ord("c"):
                    elapsed = len(calibrator._warmup_buffer) / fps
                    if elapsed >= min_warmup_sec:
                        calibrator.force_calibrate()
                        print(f"[calibration] Forced at {elapsed:.0f}s")
                    else:
                        print(
                            f"[calibration] Need at least {min_warmup_sec}s "
                            f"(current: {elapsed:.0f}s)"
                        )

                if calibrator.is_calibrated:
                    b = calibrator.baseline

                    # ── G4: Validar qualidade da calibração ───────────
                    verdict = validate_calibration(b)

                    if verdict.recommendation == "retry":
                        print(
                            f"[guardrail] Calibração REJEITADA: "
                            f"{verdict.issues}"
                        )
                        print("[guardrail] Reiniciando warm-up...")
                        calibrator = RTSubjectCalibrator(
                            CalibrationConfig(
                                fps=fps, search_sec=warmup_sec
                            )
                        )
                        continue

                    if verdict.recommendation == "use_with_caution":
                        print(
                            "[guardrail] Calibração ACEITA com ressalvas:"
                        )
                        for issue in verdict.issues:
                            print(f"  [guardrail] {issue}")
                    # ──────────────────────────────────────────────────

                    print("[calibration] Baseline computed!")
                    print(
                        f"[calibration]   EAR:   mean={b.ear_mean:.4f}, "
                        f"std={b.ear_std:.4f}"
                    )
                    print(
                        f"[calibration]   EAR P90 (debug): {b.ear_p90_raw:.4f}"
                    )
                    pf = window_factory.cfg.perclos_factor
                    print(
                        f"[calibration]   PERCLOS baseline (=ear_mean): "
                        f"{b.ear_mean:.4f}"
                    )
                    print(
                        f"[calibration]   PERCLOS factor: {pf} "
                        f"(offline=0.80, RT=0.65)"
                    )
                    print(
                        f"[calibration]   PERCLOS threshold "
                        f"(mean*{pf}): {b.ear_mean * pf:.4f}"
                    )
                    print(
                        f"[calibration]   MAR:   mean={b.mar_mean:.4f}, "
                        f"std={b.mar_std:.4f}"
                    )
                    print(
                        f"[calibration]   Pitch: mean={b.pitch_mean:.2f}, "
                        f"std={b.pitch_std:.2f}"
                    )
                    print(
                        f"[calibration]   Yaw:   mean={b.yaw_mean:.2f}, "
                        f"std={b.yaw_std:.2f}"
                    )
                    print(
                        f"[calibration]   Roll:  mean={b.roll_mean:.2f}, "
                        f"std={b.roll_std:.2f}"
                    )
                    print(f"[calibration]   Valid: {b.is_valid}")
                    print(
                        f"[calibration]   Segment: frames "
                        f"{b.segment_start}-{b.segment_end}"
                    )

                    behavior_guard.on_calibration_complete()
                    window_factory.set_perclos_baseline(b.ear_mean)
                else:
                    if not headless:
                        _draw_warmup_overlay(
                            frame, progress, warmup_sec,
                            len(calibrator._warmup_buffer) / fps,
                            raw_text,
                        )
                        cv2.imshow("SALTE Realtime Demo", frame)

                    if key == ord("q"):
                        break
                    continue

                if calibrated is None:
                    calibrated = calibrator.calibrate(feats)

            else:
                calibrated = calibrator.calibrate(feats)

            window_feats = window_factory.push(calibrated)

            status_text = "Calibrated - waiting for window..."
            color = (255, 255, 255)
            overlay2 = ""

            if window_feats is not None:
                vec_raw = np.array(
                    [window_feats[name] for name in feature_names],
                    dtype=np.float32,
                )

                if debug:
                    print("\n[debug] Raw feature vector:")
                    for i, name in enumerate(feature_names):
                        print(f"  {name:30s} = {vec_raw[i]:12.6f}")

                # ── 1. Scaler PRIMEIRO ───────────────────────────────
                vec = scale_features(vec_raw, config)

                if debug:
                    print("[debug] After scale_features:")
                    for i, name in enumerate(feature_names):
                        print(f"  {name:30s} = {vec[i]:12.6f}")

                # ── 2. FIX-RT-3: Neutralizar DEPOIS do scaler (C33) ─
                vec = neutralizer.neutralize(vec)

                if debug and neutralizer.enabled:
                    print("[debug] After HeadPoseNeutralizer:")
                    for i, name in enumerate(feature_names):
                        print(f"  {name:30s} = {vec[i]:12.6f}")
                # ─────────────────────────────────────────────────────

                # ── 3. Predição ──────────────────────────────────────
                prob_danger, label = predict_fatigue(
                    vec, model, config, threshold_override=threshold
                )

                # ── 4. Guardrails: validar e empacotar saída ─────────
                output = guardrails.validate_and_wrap(
                    prob_danger=prob_danger,
                    label=label,
                    window_feats=window_feats,
                    feature_names=feature_names,
                    config=config,
                    timestamp_ms=calibrated.timestamp_ms,
                    threshold=threshold,
                )
                output = behavior_guard.process(output)

                # ── 5. Reflection: drift + predição ──────────────────
                drift_report = drift_reflector.push_window(window_feats)
                reflection = pred_reflector.push(prob_danger, label)

                if reflection.pattern != "stable":
                    print(
                        f"[reflection] Padrão: {reflection.pattern} | "
                        f"{reflection.suggestion}"
                    )

                if reflection.confidence_modifier < 1.0:
                    output.confidence = "low"

                if drift_report is not None:
                    if drift_report.status != DriftStatus.STABLE:
                        print(
                            f"[reflection] Drift {drift_report.status.value}: "
                            f"{drift_report.drifted_features}"
                        )
                        print(
                            f"[reflection] Z-scores: "
                            f"{drift_report.details}"
                        )
                        print(
                            f"[reflection] Recomendação: "
                            f"{drift_report.recommendation}"
                        )

                    # ── 6. Auto-recalibração ─────────────────────────
                    recal_decision = recal_manager.evaluate(
                        drift_report=drift_report,
                        pred_reflection=reflection,
                    )

                    if recal_decision.should_recalibrate:
                        print(
                            f"[reflection] AUTO-RECALIBRAÇÃO: "
                            f"{recal_decision.reason} "
                            f"(urgência: {recal_decision.urgency})"
                        )

                        if recal_decision.urgency == "immediate":
                            calibrator = RTSubjectCalibrator(
                                CalibrationConfig(
                                    fps=fps, search_sec=60.0
                                )
                            )
                            window_factory = OnlineWindowFactory(
                                RTWindowConfig(fps=fps)
                            )
                            drift_reflector = DriftReflector(
                                training_stats
                            )
                            pred_reflector = PredictionReflector()
                            behavior_guard.on_calibration_complete()
                # ─────────────────────────────────────────────────────

                # Sound alert with rate limiting
                if behavior_guard.should_sound_alert(output):
                    # Placeholder: acionar buzzer / som (GPIO ou beep)
                    print("[alert] SOUND ALERT triggered")

                status_text = (
                    f"{output.label} ({output.prob_danger:.2f}) "
                    f"[{output.confidence}]"
                )
                color = (0, 0, 255) if output.label == "Danger" else (0, 255, 0)
                if output.alert_level == AlertLevel.CRITICAL:
                    color = (0, 0, 200)  # dark red for critical
                elif output.alert_level == AlertLevel.WATCH:
                    color = (0, 165, 255)  # orange for watch

                overlay2 = (
                    f"PERCLOS:{output.perclos:.2f} "
                    f"BlinkCount:{output.blink_count:.1f} "
                    f"Microsleeps:{output.microsleep_count:.1f} "
                    f"Alert:{output.alert_level.name}"
                )

                print(
                    f"[window] label={output.label} "
                    f"prob={output.prob_danger:.3f} "
                    f"alert={output.alert_level.name} "
                    f"ear_mean_z={window_feats.get('ear_mean', 0.0):.3f} "
                    f"perclos={output.perclos:.3f} "
                    f"blinks={output.blink_count:.0f} "
                    f"microsleeps={output.microsleep_count:.0f} "
                    f"confidence={output.confidence}"
                )

            if not headless:
                _draw_inference_overlay(
                    frame, status_text, color, raw_text,
                    overlay2, calibrator, calibrated,
                )
                cv2.imshow("SALTE Realtime Demo", frame)

            if key == ord("q"):
                break

    finally:
        cap.release()
        if not headless:
            cv2.destroyAllWindows()


# ── Overlay drawing helpers ──────────────────────────────────────────────────


def _draw_warmup_overlay(
    frame: np.ndarray,
    progress: float,
    warmup_sec: float,
    elapsed_sec: float,
    raw_text: str,
) -> None:
    bar_w = int(frame.shape[1] * 0.6)
    bar_h = 30
    bar_x = (frame.shape[1] - bar_w) // 2
    bar_y = frame.shape[0] // 2

    cv2.rectangle(
        frame, (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1,
    )
    fill_w = int(bar_w * progress)
    cv2.rectangle(
        frame, (bar_x, bar_y),
        (bar_x + fill_w, bar_y + bar_h), (0, 200, 255), -1,
    )
    cv2.putText(
        frame,
        f"CALIBRATING... {elapsed_sec:.0f}/{warmup_sec:.0f}s  "
        f"(press 'c' to skip)",
        (bar_x, bar_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, raw_text, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _draw_inference_overlay(
    frame: np.ndarray,
    status_text: str,
    color: tuple[int, int, int],
    raw_text: str,
    overlay2: str,
    calibrator: RTSubjectCalibrator,
    calibrated: Optional[CalibratedFrame],
) -> None:
    cv2.putText(
        frame, status_text, (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, raw_text, (16, 64),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )

    if calibrator.is_calibrated and calibrated is not None:
        znorm_text = (
            f"EAR_z:{calibrated.ear_avg_znorm:.2f} "
            f"MAR_z:{calibrated.mar_znorm:.2f} "
            f"Pitch_z:{calibrated.head_pitch_znorm:.2f}"
        )
        cv2.putText(
            frame, znorm_text, (16, 96),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 255), 1, cv2.LINE_AA,
        )

    if overlay2:
        cv2.putText(
            frame, overlay2, (16, 128),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA,
        )


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SALTE Realtime Fatigue Detection (V3 — HeadPoseNeutralizer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir", default="MODELS",
        help="Directory containing best_model.onnx, inference_config.json, "
             "blazeface_detector.onnx, face_mesh_landmark.onnx",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to best_model.onnx (default: {model-dir}/best_model.onnx)",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to inference_config.json (default: {model-dir}/inference_config.json)",
    )
    parser.add_argument(
        "--detector-model", default=None,
        help="Path to blazeface_detector.onnx",
    )
    parser.add_argument(
        "--mesh-model", default=None,
        help="Path to face_mesh_landmark.onnx",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override decision threshold (default: from inference_config, 0.41)",
    )
    parser.add_argument(
        "--picamera", action="store_true",
        help="Use picamera2 (AI Camera IMX500 via CSI)",
    )
    parser.add_argument(
        "--camera-index", type=int, default=0,
        help="USB webcam index (ignored with --picamera)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No display output (for SSH / no X11)",
    )
    parser.add_argument(
        "--warmup", type=float, default=120.0,
        help="Warm-up duration in seconds (C6-V2)",
    )
    parser.add_argument(
        "--min-warmup", type=float, default=30.0,
        help="Minimum seconds before allowing forced calibration",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Target FPS",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print feature vector for each window (raw, scaled, neutralized)",
    )
    # ── FIX-RT-3: CLI toggle (C33) ──────────────────────────────────────
    parser.add_argument(
        "--no-neutralize-pose", action="store_true",
        help="Desativar neutralização de head pose (para testes com dados de lab). "
             "Default: neutralização ATIVADA (4 features de pose zeradas).",
    )
    # ────────────────────────────────────────────────────────────────────
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint = Path(args.model) if args.model else model_dir / "best_model.onnx"
    config_path = Path(args.config) if args.config else model_dir / "inference_config.json"
    detector_path = Path(args.detector_model) if args.detector_model else model_dir / "blazeface_detector.onnx"
    mesh_path = Path(args.mesh_model) if args.mesh_model else model_dir / "face_mesh_landmark.onnx"

    run_realtime(
        checkpoint_path=checkpoint,
        config_path=config_path,
        detector_path=detector_path,
        mesh_path=mesh_path,
        camera_index=args.camera_index,
        use_picamera=args.picamera,
        threshold_override=args.threshold,
        warmup_sec=args.warmup,
        min_warmup_sec=args.min_warmup,
        fps=args.fps,
        headless=args.headless,
        debug=args.debug,
        neutralize_pose=not args.no_neutralize_pose,
    )


if __name__ == "__main__":
    main()