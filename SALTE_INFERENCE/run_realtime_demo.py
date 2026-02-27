"""
Loop principal de inferência em tempo real com calibração per-subject.

V2: ONNXFaceMeshBackend, best_model.onnx (19 features), inference_config.json.
Pipeline: picamera2/cv2 -> ONNXFaceMesh -> features -> scale_features (JSON) -> MLP -> Safe/Danger.
"""

from __future__ import annotations

import argparse
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
) -> None:
    """
    Loop realtime com calibração per-subject.

    Fases:
    1. WARM-UP: coleta frames por warmup_sec segundos
    2. CALIBRATED: inferência com Safe/Danger overlay
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

    print(f"[init] Scaler: JSON (inference_config)")
    print(f"[init] Threshold: {threshold} (19 features)")

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
                    print("\n[debug] Raw 19-feature vector:")
                    for i, name in enumerate(feature_names):
                        print(f"  {name:30s} = {vec_raw[i]:12.6f}")

                vec = scale_features(vec_raw, config)

                if debug:
                    print("[debug] After scale_features:")
                    for i, name in enumerate(feature_names):
                        print(f"  {name:30s} = {vec[i]:12.6f}")

                prob_danger, label = predict_fatigue(
                    vec, model, config, threshold_override=threshold
                )
                status_text = f"{label} ({prob_danger:.2f})"
                color = (0, 0, 255) if label == "Danger" else (0, 255, 0)

                perclos = window_feats.get("perclos_p80_mean", 0.0)
                blink_count = window_feats.get("blink_count", 0.0)
                micros = window_feats.get("microsleep_count", 0.0)
                overlay2 = (
                    f"PERCLOS:{perclos:.2f} "
                    f"BlinkCount:{blink_count:.1f} "
                    f"Microsleeps:{micros:.1f}"
                )

                print(
                    f"[window] label={label} prob={prob_danger:.3f} "
                    f"ear_mean_z={window_feats.get('ear_mean', 0.0):.3f} "
                    f"perclos={perclos:.3f} "
                    f"blinks={blink_count:.0f} "
                    f"microsleeps={micros:.0f}"
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
        description="SALTE Realtime Fatigue Detection (V2)",
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
        help="Print 19-feature vector for each window",
    )
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
    )


if __name__ == "__main__":
    main()
