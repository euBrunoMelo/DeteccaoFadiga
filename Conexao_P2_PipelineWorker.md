# Conexão P2: Integração do PipelineWorker no Loop Principal

## Ponto de Atenção — Estado Atual

O `FrameGrabber` (P1) e o `PerformanceMonitor` (P3) estão ativos quando `--parallel` é passado. Porém, o **miolo pesado do loop** — feature extraction, calibração, window aggregation, MLP inference, guardrails, reflection, agents — ainda roda **sincronamente na thread principal**.

Evidência concreta no código (linha ~674):

```python
perf_monitor.maybe_report(
    grabber.stats,
    {"avg_process_ms": 0, "frames_processed": 0},  # ← dummy! Worker não está plugado
)
```

O `PipelineWorker` já existe em `parallel.py` e está testado (4 testes passam), mas nunca foi instanciado no loop. Este documento detalha como plugar o P2, encapsulando as ~320 linhas do loop numa `process_fn` que o Worker executa em thread separada.

---

## Diagnóstico: O Que Precisa Mudar

### O que o loop faz hoje (linhas 340-676, ~340 linhas)

```
MAIN THREAD (serial):
  1. frame = grab                          ← já pode ser paralelo (P1 ok)
  2. feats = extractor.process_frame()     ← PESADO (~15-30ms ONNX FaceMesh)
  3. calibrator.push/calibrate()           ← leve (~1ms)
  4. window_factory.push()                 ← leve (~2ms)
  5. scale + neutralize + predict()        ← PESADO (~5-10ms ONNX MLP)
  6. guardrails.validate_and_wrap()        ← leve (<1ms)
  7. behavior_guard.process()              ← leve (<1ms)
  8. drift_reflector + pred_reflector      ← leve (<1ms)
  9. recal_manager.evaluate()              ← leve (<1ms)
  10. supervisor.decide()                  ← leve (<1ms)
  11. session.on_window + feat_logger      ← leve (<1ms)
  12. overlay text preparation             ← leve (<1ms)
  13. cv2.imshow()                         ← DEVE ficar na main thread (OpenCV constraint)
  14. cv2.waitKey()                        ← DEVE ficar na main thread
  15. keyboard handling ('c', 'q')         ← DEVE ficar na main thread
```

Os passos 2-12 somam ~25-45ms no RPi5. Movê-los para o Worker libera a main thread para fazer apenas 1 (grab), 13-15 (display/input), rodando a ~60fps de display mesmo que o processamento esteja a ~25fps.

### Complicações a resolver

| Problema | Onde ocorre | Solução |
|----------|-------------|---------|
| `cv2.imshow` / `cv2.waitKey` devem ficar na main thread | Passos 13-14 | Worker retorna `PipelineResult` com dados pré-computados; main thread só desenha |
| Tecla 'c' (force calibrate) precisa chegar ao worker | Passo 15 | `threading.Event` compartilhado: main seta, worker consome |
| Tecla 'q' (quit) precisa parar tudo | Passo 15 | `threading.Event` compartilhado: `stop_event` |
| Warmup overlay precisa de `progress` e `frame` | Passo 3 | Worker retorna `PipelineResult(is_warmup=True, warmup_progress=...)` |
| Auto-recalibração reseta `calibrator`, `window_factory` etc | Passo 9 | Esses objetos vivem DENTRO do Worker (closure) — reset é interno |
| `session`, `feat_logger`, `op_store` são side-effects | Passo 11 | Mantidos dentro do Worker — acesso serial garantido (1 thread acessa) |
| PerformanceMonitor precisa de worker stats reais | Linha 674 | `perf_monitor.maybe_report(grabber.stats, worker.stats)` |

---

## Plano de Execução

### Passo 1 — Extrair `_build_process_fn()` (closure factory)

Criar uma função que captura todas as dependências do pipeline em uma closure e retorna um `Callable[[np.ndarray], PipelineResult]`.

**Local:** Nova função em `run_realtime_demo.py`, antes de `run_realtime()`.

```python
def _build_process_fn(
    *,
    extractor: RealTimeFeatureExtractor,
    calibrator: RTSubjectCalibrator,
    window_factory: OnlineWindowFactory,
    model,
    config,
    neutralizer,
    threshold: float,
    feature_names: list,
    behavior_guard: BehaviorGuardRails,
    drift_reflector: DriftReflector,
    pred_reflector: PredictionReflector,
    recal_manager: AutoRecalibrationManager,
    supervisor,  # Optional[SupervisorAgent]
    session: SessionMemory,
    feat_logger: FeatureLogger,
    op_store: OperatorStore,
    operator_id: str,
    training_stats: dict,
    fps: int,
    warmup_sec: float,
    min_warmup_sec: float,
    debug: bool,
    force_calibrate_event: threading.Event,
) -> Callable[[np.ndarray], PipelineResult]:
    """
    Constrói a process_fn para o PipelineWorker via closure.
    
    Todos os objetos stateful (extractor, calibrator, window_factory,
    behavior_guard, drift_reflector, etc.) ficam capturados na closure.
    Como o PipelineWorker chama process_fn de uma ÚNICA thread worker,
    o acesso serial a esses objetos é garantido — sem race conditions.
    
    A main thread NÃO acessa nenhum desses objetos diretamente.
    A comunicação é exclusivamente via PipelineResult (output queue).
    """
    
    def process_fn(frame: np.ndarray) -> PipelineResult:
        # Referencia nonlocal para objetos que podem ser reassigned
        # durante auto-recalibração
        nonlocal calibrator, window_factory, drift_reflector, pred_reflector
        
        session.total_frames += 1
        feats = extractor.process_frame(frame)
        
        raw_text = (
            f"EAR:{feats.ear_avg:.3f} "
            f"MAR:{feats.mar:.3f} "
            f"Face:{int(feats.face_detected)}"
        )
        
        # ── Checar se main thread pediu force calibrate ──
        if force_calibrate_event.is_set():
            force_calibrate_event.clear()
            if not calibrator.is_calibrated:
                elapsed = len(calibrator._warmup_buffer) / fps
                if elapsed >= min_warmup_sec:
                    calibrator.force_calibrate()
                    print(f"[calibration] Forced at {elapsed:.0f}s")
        
        # ── Warm-up phase ──
        if not calibrator.is_calibrated:
            calibrated = calibrator.push(feats)
            progress = calibrator.warmup_progress
            
            if calibrator.is_calibrated:
                b = calibrator.baseline
                verdict = validate_calibration(b)
                
                if verdict.recommendation == "retry":
                    print(f"[guardrail] Calibração REJEITADA: {verdict.issues}")
                    print("[guardrail] Reiniciando warm-up...")
                    calibrator = RTSubjectCalibrator(
                        CalibrationConfig(fps=fps, search_sec=warmup_sec)
                    )
                    return PipelineResult(
                        frame=frame, raw_text=raw_text,
                        status_text="Recalibrating...",
                        color=(0, 200, 255), overlay2="",
                        is_warmup=True, warmup_progress=0.0,
                        warmup_elapsed=0.0,
                    )
                
                if verdict.recommendation == "use_with_caution":
                    print("[guardrail] Calibração ACEITA com ressalvas:")
                    for issue in verdict.issues:
                        print(f"  [guardrail] {issue}")
                
                # Log calibration (mesmo bloco de prints que existe hoje)
                print("[calibration] Baseline computed!")
                print(f"[calibration]   EAR: mean={b.ear_mean:.4f}, std={b.ear_std:.4f}")
                
                behavior_guard.on_calibration_complete()
                session.on_calibration(b.ear_mean, verdict.recommendation)
                op_store.upsert_profile_from_calibration(operator_id, b)
                window_factory.set_perclos_baseline(b.ear_mean)
                
                if calibrated is None:
                    calibrated = calibrator.calibrate(feats)
            else:
                # Ainda em warmup — retornar resultado de warmup
                elapsed = len(calibrator._warmup_buffer) / fps
                return PipelineResult(
                    frame=frame, raw_text=raw_text,
                    status_text="Calibrating...",
                    color=(0, 200, 255), overlay2="",
                    is_warmup=True,
                    warmup_progress=progress,
                    warmup_elapsed=elapsed,
                )
        else:
            calibrated = calibrator.calibrate(feats)
        
        # ── Window aggregation ──
        window_feats = window_factory.push(calibrated)
        
        status_text = "Calibrated - waiting for window..."
        color = (255, 255, 255)
        overlay2 = ""
        output = None
        
        if window_feats is not None:
            # Scale → Neutralize → Predict (passos 5-6 atuais)
            vec_raw = np.array(
                [window_feats[name] for name in feature_names],
                dtype=np.float32,
            )
            vec = scale_features(vec_raw, config)
            vec = neutralizer.neutralize(vec)
            
            prob_danger, label = predict_fatigue(
                vec, model, config, threshold_override=threshold
            )
            
            # Guardrails (G1-G3)
            output = guardrails.validate_and_wrap(
                prob_danger=prob_danger, label=label,
                window_feats=window_feats,
                feature_names=feature_names,
                config=config,
                timestamp_ms=calibrated.timestamp_ms,
                threshold=threshold,
            )
            output = behavior_guard.process(output)
            
            # Reflection (R1-R3)
            drift_report = drift_reflector.push_window(window_feats)
            reflection = pred_reflector.push(prob_danger, label)
            
            if reflection.pattern != "stable":
                print(f"[reflection] Padrão: {reflection.pattern} | "
                      f"{reflection.suggestion}")
            if reflection.confidence_modifier < 1.0:
                output.confidence = "low"
            
            if drift_report is not None:
                if drift_report.status != DriftStatus.STABLE:
                    print(f"[reflection] Drift {drift_report.status.value}: "
                          f"{drift_report.drifted_features}")
                
                recal_decision = recal_manager.evaluate(
                    drift_report=drift_report,
                    pred_reflection=reflection,
                )
                if recal_decision.should_recalibrate:
                    print(f"[reflection] AUTO-RECALIBRAÇÃO: "
                          f"{recal_decision.reason}")
                    session.on_auto_recalibration()
                    if recal_decision.urgency == "immediate":
                        calibrator = RTSubjectCalibrator(
                            CalibrationConfig(fps=fps, search_sec=60.0)
                        )
                        window_factory = OnlineWindowFactory(
                            RTWindowConfig(fps=fps)
                        )
                        drift_reflector = DriftReflector(training_stats)
                        pred_reflector = PredictionReflector()
                        behavior_guard.on_calibration_complete()
            
            # Alerts
            alert_triggered = behavior_guard.should_sound_alert(output)
            if alert_triggered:
                print("[alert] SOUND ALERT triggered")
            session.on_alert(triggered=alert_triggered)
            
            # Memory
            session.on_window(
                label=output.label,
                prob_danger=output.prob_danger,
                alert_level_name=output.alert_level.name,
                perclos=output.perclos,
                ear_mean=window_feats.get("ear_mean", 0.0),
                microsleep_count=output.microsleep_count,
                microsleep_total_ms=window_feats.get("microsleep_total_ms", 0.0),
            )
            feat_logger.log(
                timestamp_ms=output.timestamp_ms,
                label=output.label,
                prob_danger=output.prob_danger,
                alert_level=output.alert_level.name,
                confidence=output.confidence,
                window_feats=window_feats,
            )
            
            # Multi-Agent
            supervisor_decision = None
            if supervisor is not None:
                supervisor_decision = supervisor.decide(window_feats)
                if debug:
                    print(f"[agents] MLP: {output.label} ({output.prob_danger:.3f})")
                    print(f"[agents] Supervisor: {supervisor_decision.label}")
            
            # Build overlay strings
            status_text = (
                f"{output.label} ({output.prob_danger:.2f}) "
                f"[{output.confidence}]"
            )
            color = (0, 0, 255) if output.label == "Danger" else (0, 255, 0)
            if output.alert_level == AlertLevel.CRITICAL:
                color = (0, 0, 200)
            elif output.alert_level == AlertLevel.WATCH:
                color = (0, 165, 255)
            
            overlay2 = (
                f"PERCLOS:{output.perclos:.2f} "
                f"BlinkCount:{output.blink_count:.1f} "
                f"Microsleeps:{output.microsleep_count:.1f} "
                f"Alert:{output.alert_level.name}"
            )
            if (supervisor_decision is not None
                    and supervisor_decision.fatigue_type != "none"):
                overlay2 += f" Type:{supervisor_decision.fatigue_type}"
            
            print(f"[window] label={output.label} "
                  f"prob={output.prob_danger:.3f} "
                  f"alert={output.alert_level.name}")
        
        return PipelineResult(
            frame=frame,
            raw_text=raw_text,
            status_text=status_text,
            color=color,
            overlay2=overlay2,
            output=output,
            window_feats=window_feats,
        )
    
    return process_fn
```

### Passo 2 — Refatorar o main loop para modo Worker

O `run_realtime()` ganha dois caminhos: `--parallel` com Worker completo, e sem `--parallel` com o loop serial existente (backward-compatible).

```python
    # ── Sinais inter-thread ──────────────────────────────────────────
    force_calibrate_event = threading.Event()
    stop_event = threading.Event()

    if parallel:
        # Construir process_fn via closure
        process_fn = _build_process_fn(
            extractor=extractor,
            calibrator=calibrator,
            window_factory=window_factory,
            model=model,
            config=config,
            neutralizer=neutralizer,
            threshold=threshold,
            feature_names=feature_names,
            behavior_guard=behavior_guard,
            drift_reflector=drift_reflector,
            pred_reflector=pred_reflector,
            recal_manager=recal_manager,
            supervisor=supervisor,
            session=session,
            feat_logger=feat_logger,
            op_store=op_store,
            operator_id=_op_id,
            training_stats=training_stats,
            fps=fps,
            warmup_sec=warmup_sec,
            min_warmup_sec=min_warmup_sec,
            debug=debug,
            force_calibrate_event=force_calibrate_event,
        )
        
        worker = PipelineWorker(process_fn=process_fn, queue_size=2)
        worker.start()
        print("[init] Parallel: PipelineWorker started")
        
        # ── MAIN LOOP: PARALLEL MODE ────────────────────────────────
        try:
            last_result: Optional[PipelineResult] = None
            
            while not stop_event.is_set():
                # 1. Grab frame (from FrameGrabber thread)
                frame = grabber.get_frame(timeout=0.1)
                if frame is not None:
                    perf_monitor.on_capture()
                    worker.put_frame(frame)
                
                # 2. Get processed result (non-blocking)
                result = worker.get_result(timeout=0.005)
                if result is not None:
                    perf_monitor.on_process_complete(
                        worker.avg_process_time_ms
                    )
                    last_result = result
                
                # 3. Display (always show latest result)
                if not headless and last_result is not None:
                    if last_result.is_warmup:
                        _draw_warmup_overlay(
                            last_result.frame,
                            last_result.warmup_progress,
                            warmup_sec,
                            last_result.warmup_elapsed,
                            last_result.raw_text,
                        )
                    else:
                        _draw_inference_overlay(
                            last_result.frame,
                            last_result.status_text,
                            last_result.color,
                            last_result.raw_text,
                            last_result.overlay2,
                            calibrator,
                            None,  # calibrated frame not available here
                        )
                    cv2.imshow("SALTE Realtime Demo", last_result.frame)
                    perf_monitor.on_display()
                
                # 4. Keyboard input (main thread only)
                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("c"):
                        force_calibrate_event.set()
                    elif key == ord("q"):
                        stop_event.set()
                        break
                else:
                    time.sleep(0.005)  # yield CPU
                
                # 5. Performance report (with REAL worker stats)
                perf_monitor.maybe_report(
                    grabber.stats,
                    worker.stats,  # ← agora stats reais do Worker
                )
        
        finally:
            worker.stop()
            grabber.stop()
            cap.release()
            if not headless:
                cv2.destroyAllWindows()
            # Memory cleanup (mesma lógica de hoje)
            summary = session.summary()
            print(f"[session] Resumo final: {summary}")
            op_store.save_session(_op_id, summary, session_started_at)
            op_store.close()
            feat_logger.close()
    
    else:
        # ── MAIN LOOP: SERIAL MODE (código atual, inalterado) ────────
        try:
            while True:
                # ... loop serial existente (342-676) sem mudanças ...
```

### Passo 3 — Atualizar `_draw_inference_overlay` para P2

A função `_draw_inference_overlay` recebe `calibrator` e `calibrated` para mostrar Z-norm no overlay. No modo Worker, esses objetos estão na thread worker. Duas opções:

**Opção escolhida:** Adicionar campos opcionais ao `PipelineResult` para transportar z-norm values:

```python
@dataclass
class PipelineResult:
    frame: np.ndarray
    raw_text: str
    status_text: str
    color: Tuple[int, int, int]
    overlay2: str
    output: Optional[Any] = None
    window_feats: Optional[Dict] = None
    is_warmup: bool = False
    warmup_progress: float = 0.0
    warmup_elapsed: float = 0.0
    # ★ Novos campos para transportar z-norm ao display
    znorm_text: str = ""       # "EAR_z:0.12 MAR_z:-0.34 Pitch_z:0.01"
    is_calibrated: bool = False
```

O worker preenche `znorm_text` quando calibrado:

```python
# Dentro do process_fn, após calibrate():
if calibrated is not None and calibrated.face_detected:
    znorm_text = (
        f"EAR_z:{calibrated.ear_avg_znorm:.2f} "
        f"MAR_z:{calibrated.mar_znorm:.2f} "
        f"Pitch_z:{calibrated.head_pitch_znorm:.2f}"
    )
else:
    znorm_text = ""
```

E a main thread usa `result.znorm_text` no overlay ao invés de acessar `calibrator`/`calibrated` diretamente.

### Passo 4 — Testes

**Novos testes em `test_parallel.py`:**

```python
class TestPipelineWorkerIntegrationP2:
    """Testes do PipelineWorker com process_fn simulando o pipeline real."""
    
    def test_worker_returns_warmup_result(self):
        """process_fn que simula warmup retorna PipelineResult(is_warmup=True)."""
        def fake_process(frame):
            return PipelineResult(
                frame=frame, raw_text="EAR:0.3", status_text="Calibrating...",
                color=(0, 200, 255), overlay2="",
                is_warmup=True, warmup_progress=0.5, warmup_elapsed=10.0,
            )
        worker = PipelineWorker(fake_process)
        worker.start()
        worker.put_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        time.sleep(0.1)
        result = worker.get_result(timeout=0.5)
        worker.stop()
        assert result is not None
        assert result.is_warmup is True
        assert result.warmup_progress == 0.5
    
    def test_worker_returns_inference_result(self):
        """process_fn que simula inferência retorna PipelineResult com output."""
        ...
    
    def test_force_calibrate_event(self):
        """threading.Event propaga corretamente entre threads."""
        event = threading.Event()
        assert not event.is_set()
        event.set()
        assert event.is_set()
        event.clear()
        assert not event.is_set()
    
    def test_worker_stats_reflect_real_processing(self):
        """worker.stats tem avg_process_ms > 0 após processar frames."""
        ...
```

---

## Diagrama de Threads — Antes vs Depois

### ANTES (--parallel atual, P2 não plugado)

```
Thread FrameGrabber:  camera.read() → Queue
Thread Main:          Queue.get() → [extract → calibrate → window → predict → guardrails
                                      → reflection → agents → memory → overlay → display]
                      ^^^ tudo na mesma thread ^^^
```

FPS de display = FPS de processamento (acoplados). Se ONNX demora 40ms, display roda a 25fps.

### DEPOIS (P2 plugado)

```
Thread FrameGrabber:  camera.read() → frame_queue
Thread PipelineWorker: frame_queue → [extract → calibrate → window → predict → guardrails
                                       → reflection → agents → memory] → result_queue
Thread Main:          result_queue.get() → display + keyboard
```

FPS de display ≈ 60fps (apenas `cv2.imshow` + `waitKey`). FPS de processamento = o que o hardware aguentar (25-30fps no RPi5). O display mostra o **último resultado disponível**, sem bloquear.

---

## Riscos e Mitigações

| Risco | Probabilidade | Mitigação |
|-------|:---:|-----------|
| Race condition em `session` | Baixa | Worker é a única thread que escreve em `session`. Main thread nunca acessa. |
| `calibrator` reassignment no worker (auto-recalibração) | Média | Usar `nonlocal` na closure. Testado: Python permite `nonlocal` para reassignment em closures. |
| `cv2.imshow` chamado de thread errada | Alta se bug | Garantido: main thread é a única com `cv2.imshow`. Worker NUNCA chama OpenCV display. |
| Worker morre silenciosamente (exceção não-tratada) | Média | Envolver `process_fn` em try/except dentro do `_worker_loop`, logando exceção e retornando `PipelineResult` de erro. |
| Latência extra pela fila | Baixa | Queue(maxsize=2) com drop-old garante < 2 frames de latência (~66ms a 30fps). Aceitável para detecção de fadiga. |
| `op_store.upsert_profile_from_calibration` é SQLite write do worker thread | Baixa | SQLite em modo `check_same_thread=False` ou acesso exclusivo (worker é a única thread que escreve). |

### Mitigação SQLite multi-thread

O `OperatorStore` precisa de uma pequena mudança para funcionar no worker thread:

```python
# Em memory.py, OperatorStore.__init__:
self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
```

Isso é seguro porque apenas o worker thread escreve no banco. O main thread nunca acessa o `op_store` durante a execução — só no `finally` (que roda após `worker.stop()`, logo o worker já parou).

---

## Checklist de Implementação

```
[ ] 1. Adicionar campos znorm_text e is_calibrated ao PipelineResult (parallel.py)
[ ] 2. Criar _build_process_fn() em run_realtime_demo.py
[ ] 3. Adicionar threading.Event (force_calibrate_event, stop_event)
[ ] 4. Criar bloco PARALLEL MODE no run_realtime()
[ ] 5. Manter bloco SERIAL MODE intacto (backward-compatible)
[ ] 6. Atualizar OperatorStore com check_same_thread=False
[ ] 7. Plugar worker.stats no perf_monitor.maybe_report()
[ ] 8. Adicionar error handling no _worker_loop (try/except + log)
[ ] 9. Adicionar ~6 testes de integração P2 em test_parallel.py
[ ] 10. Testar: --parallel (worker mode) vs sem flag (serial mode) — mesmos resultados
[ ] 11. Testar: RPi5 com --parallel — confirmar FPS de display > FPS de processamento
[ ] 12. Testar: 'c' (force calibrate) funciona via threading.Event no modo paralelo
```

---

## Métricas de Sucesso

| Métrica | Antes (P1 only) | Depois (P1+P2+P3) | Como medir |
|---------|-----------------|-------------------|------------|
| Display FPS (RPi5) | ~25fps (acoplado) | ~60fps (desacoplado) | `perf_monitor` disp_fps |
| Processing FPS (RPi5) | ~25fps | ~25fps (mesma) | `worker.stats["avg_process_ms"]` |
| worker stats no report | `avg_process_ms: 0` (dummy) | valor real (~35ms) | Verificar output do `maybe_report` |
| Modo serial intacto | Funciona | Funciona idêntico | Rodar sem `--parallel`, comparar logs |
| End-to-end latência | ~40ms | ~80ms (2 frames de fila) | Medir timestamp do frame vs display |
