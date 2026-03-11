# Changelog: Implementação Memory Management + Parallelization

**Data:** 2026-03-11
**Branch:** plano-de-migracao
**Baseado em:** Plano_Execucao_Memory_Parallelization.md

---

## Resumo

Implementação completa dos padrões agênticos **Memory Management (M1-M3)** e **Parallelization (P1-P3)** no pipeline de inferência DeteccaoFadiga2. O sistema ganha memória de curto e longo prazo (sessões e perfis de operador persistentes em SQLite), logging estruturado de features para data flywheel, e captura paralela de câmera com pipeline worker em threads separadas.

---

## Arquivos Criados

### `SALTE_INFERENCE/memory.py`
Módulo completo de memória com 3 componentes:

| Fase | Componente | Descrição |
|------|-----------|-----------|
| **M1** | `SessionMemory` | Dataclass de memória de curto prazo. Rastreia: total_windows, danger_ratio, max_consecutive_danger, prob_history (deque maxlen=120), perclos_history (maxlen=120), microsleep_timestamps, calibration_count, alerts. `perclos_trend_slope` via regressão linear (requer ≥10 pontos). `summary()` retorna dicionário completo da sessão. |
| **M2** | `OperatorStore` | Memória de longo prazo via SQLite. Tabelas: `operator_profiles` (EAR/MAR/head pose means + sessions_count + running averages) e `session_logs` (JSON blob por sessão). `upsert_profile_from_calibration()` usa EMA (alpha=0.3) para atualizar perfis existentes. `get_warm_start_baseline()` requer ≥2 sessões anteriores. `save_session()` atualiza médias corridas no perfil. |
| **M3** | `FeatureLogger` | Logger CSV opt-in (disabled por default). Header: 6 colunas de metadata (timestamp_ms, operator_id, label, prob_danger, alert_level, confidence) + 19 features. Flush automático a cada 100 linhas. Habilitado via `--log-features`. Gera arquivo nomeado com operator_id e timestamp para data flywheel. |

### `SALTE_INFERENCE/parallel.py`
Módulo completo de paralelização com 3 componentes:

| Fase | Componente | Descrição |
|------|-----------|-----------|
| **P1** | `FrameGrabber` | Captura de câmera em daemon thread dedicada. Queue(maxsize=2) com drop de frames antigos quando cheia (`drop_old_frames=True`). Garante que a câmera nunca para de capturar e que o consumidor sempre recebe o frame mais recente. Stats: frames_grabbed, frames_dropped, queue_size. |
| **P1** | `FrameGrabberConfig` | Dataclass configurável: queue_size (default 2), target_fps (default 30), drop_old_frames (default True). |
| **P2** | `PipelineWorker` | Executa `process_fn` em daemon thread separada. Padrão producer-consumer com input_queue e output_queue (ambas maxsize=2). Descarta frames antigos em ambas as filas para evitar backpressure. Stats: frames_processed, avg_process_ms, input/output queue sizes. |
| **P3** | `PerformanceMonitor` | Métricas de performance: FPS de captura/processamento/display via deques de timestamps (maxlen=100), latência avg + p95, taxa de drop, utilização de fila. Relatório periódico no console (default 30s) via `maybe_report()`. |

### `SALTE_INFERENCE/tests/test_memory.py`
26 testes unitários cobrindo M1-M3:
- `TestSessionMemoryM1` (15 testes): estado inicial, on_window safe/danger, danger_ratio, max_consecutive_danger, watch/critical counters, microsleep tracking, calibration tracking, alert tracking, auto-recalibration tracking, perclos_trend_slope (increasing/stable/insufficient data), avg_prob_danger, summary keys
- `TestOperatorStoreM2` (7 testes): profile not found, upsert creates profile, upsert updates with EMA, save/get session, warm start requires 2 sessions, multiple sessions ordered, profile sessions count
- `TestFeatureLoggerM3` (4 testes): disabled creates no file, enabled creates CSV, CSV header correct, log when disabled is noop

### `SALTE_INFERENCE/tests/test_parallel.py`
15 testes unitários cobrindo P1-P3:
- `TestFrameGrabberP1` (6 testes): start/stop, get_frame, stats tracking, drops when slow consumer, none when no frames, double start/stop safety
- `TestPipelineWorkerP2` (4 testes): start/stop, process frame, stats tracking, slow worker drops old input
- `TestPerformanceMonitorP3` (4 testes): no report before interval, report after interval, FPS computation, latency tracking

---

## Arquivos Modificados

### `SALTE_INFERENCE/run_realtime_demo.py`
Integração completa dos novos módulos no loop principal:

1. **Imports adicionados:** `datetime`, `memory` (SessionMemory, OperatorStore, FeatureLogger), `parallel` (FrameGrabber, FrameGrabberConfig, PipelineWorker, PerformanceMonitor)
2. **Novos parâmetros CLI:** `--log-features` (habilita FeatureLogger CSV), `--operator-id` (identifica operador para memória), `--parallel` (habilita captura paralela)
3. **Inicialização Memory:** `SessionMemory(operator_id)`, `OperatorStore(db_path)`, `FeatureLogger(output_dir, feature_names, operator_id, enabled)`
4. **Warm-start:** `op_store.get_warm_start_baseline(operator_id)` verifica se operador conhecido tem ≥2 sessões → reduz warm-up de 120s para 30s
5. **Pós-calibração Memory:** `session.on_calibration(ear_mean, verdict)` + `op_store.upsert_profile_from_calibration(operator_id, baseline)`
6. **Pós-inferência Memory:** `session.on_window(label, prob, alert_level, perclos, ear_mean, microsleep_count, microsleep_ms)` + `feat_logger.log(timestamp_ms, label, prob, alert_level, confidence, window_feats)`
7. **Parallel capture:** Quando `--parallel`, `FrameGrabber` captura em thread dedicada e `grabber.get_frame()` substitui `cap.read()`. `PerformanceMonitor` rastreia FPS e reporta periodicamente.
8. **Finally block:** `session.summary()` impresso, `op_store.save_session()` persiste sessão, cleanup de todos os recursos (op_store.close, feat_logger.close, grabber.stop)

### `SALTE_INFERENCE/__init__.py`
Docstring atualizada com descrição dos módulos `memory` e `parallel`.

---

## Bugs Encontrados e Corrigidos

### Deadlock reentrante em `PipelineWorker.stats` (P2)
- **Problema:** A property `stats` adquiria `self._lock` e depois chamava `self.avg_process_time_ms`, que também tentava adquirir `self._lock`. Como `threading.Lock()` não é reentrante, causava deadlock.
- **Sintoma:** Testes `test_stats_tracking` travavam indefinidamente.
- **Correção:** Inlinou o cálculo de avg diretamente na property `stats` em vez de chamar `avg_process_time_ms`:
```python
# ANTES (deadlock):
@property
def stats(self):
    with self._lock:
        return {"avg_process_ms": round(self.avg_process_time_ms, 1), ...}

# DEPOIS (corrigido):
@property
def stats(self):
    with self._lock:
        avg_ms = (self._total_process_time_ms / self._frames_processed
                  if self._frames_processed > 0 else 0.0)
        return {"avg_process_ms": round(avg_ms, 1), ...}
```

---

## Resultados dos Testes

```
83 passed in 3.02s
```

Todos os 83 testes passaram:
- 24 guardrails (G1-G4)
- 18 reflection (R1-R3)
- 26 memory (M1-M3)
- 15 parallel (P1-P3)

---

## Estrutura Final

```
SALTE_INFERENCE/
├── __init__.py                    (atualizado)
├── feature_extractor_rt.py        (sem alterações)
├── subject_calibrator_rt.py       (sem alterações)
├── window_factory_rt.py           (sem alterações)
├── model_loader.py                (sem alterações)
├── guardrails.py                  (Phase 1)
├── reflection.py                  (Phase 1)
├── memory.py                      ★ NOVO — M1/M2/M3
├── parallel.py                    ★ NOVO — P1/P2/P3
├── run_realtime_demo.py           (modificado — integração completa)
├── offline_eval.py                (sem alterações)
└── tests/
    ├── __init__.py
    ├── test_guardrails.py         (Phase 1 — 24 testes)
    ├── test_reflection.py         (Phase 1 — 18 testes)
    ├── test_memory.py             ★ NOVO — 26 testes
    └── test_parallel.py           ★ NOVO — 15 testes
```
