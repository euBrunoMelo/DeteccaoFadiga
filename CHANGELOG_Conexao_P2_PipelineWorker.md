# Changelog: Conexão P2 — Integração do PipelineWorker no Loop Principal

**Data:** 2026-03-11
**Branch:** plano-de-migracao
**Baseado em:** Conexao_P2_PipelineWorker.md

---

## Resumo

O `PipelineWorker` (P2), que já existia em `parallel.py` mas nunca era instanciado no loop principal, foi completamente integrado. Quando `--parallel` é ativado, o pipeline inteiro (feature extraction, calibração, windowing, MLP inference, guardrails, reflection, agents, memory) agora roda numa **thread worker separada**, liberando a main thread para apenas display + keyboard input. O modo serial (sem `--parallel`) permanece inalterado.

---

## Mudanças Implementadas

### 1. `SALTE_INFERENCE/parallel.py` — Novos campos em PipelineResult

Adicionados dois campos ao dataclass `PipelineResult` para transportar z-norm values e estado de calibração do worker para a main thread:

```python
znorm_text: str = ""          # "EAR_z:0.12 MAR_z:-0.34 Pitch_z:0.01"
is_calibrated: bool = False
```

Isso resolve o problema de a main thread (modo paralelo) não ter acesso direto ao `calibrator` e `calibrated` frame que vivem no worker.

### 2. `SALTE_INFERENCE/memory.py` — SQLite multi-thread

```python
self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
```

Necessário porque o `OperatorStore` é acessado pelo worker thread (durante execução) e pelo main thread (no `finally` após `worker.stop()`). Seguro porque apenas uma thread escreve por vez.

### 3. `SALTE_INFERENCE/run_realtime_demo.py` — Refatoração principal

#### `_build_process_fn()` (nova função)
Factory que constrói a `process_fn` via closure, capturando todos os objetos stateful:
- `extractor`, `calibrator`, `window_factory`, `model`, `config`, `neutralizer`
- `behavior_guard`, `drift_reflector`, `pred_reflector`, `recal_manager`
- `supervisor`, `session`, `feat_logger`, `op_store`
- `force_calibrate_event` (threading.Event para comunicação inter-thread)

Usa `nonlocal` para `calibrator`, `window_factory`, `drift_reflector`, `pred_reflector` (reassigned durante auto-recalibração).

O worker preenche `PipelineResult.znorm_text` e `PipelineResult.is_calibrated` para que a main thread possa desenhar o overlay sem acessar objetos do worker.

#### Loop principal — dois modos

| Aspecto | Serial (sem --parallel) | Paralelo (--parallel) |
|---------|------------------------|----------------------|
| Captura | `cap.read()` na main thread | `FrameGrabber` thread → queue |
| Processamento | Main thread (síncrono) | `PipelineWorker` thread → result queue |
| Display | Main thread | Main thread (apenas overlay + imshow) |
| Keyboard | Main thread (cv2.waitKey) | Main thread (cv2.waitKey) |
| Force calibrate | Direto no calibrator | Via `threading.Event` |
| Perf report | N/A | `worker.stats` (valores reais) |

#### `_draw_inference_overlay_parallel()` (nova função)
Versão do overlay para modo paralelo que recebe `znorm_text` e `is_calibrated` como strings/bools em vez de objetos `calibrator`/`calibrated`.

#### Dummy stats eliminado
```python
# ANTES:
perf_monitor.maybe_report(grabber.stats, {"avg_process_ms": 0, "frames_processed": 0})

# DEPOIS:
perf_monitor.maybe_report(grabber.stats, worker.stats)
```

### 4. `SALTE_INFERENCE/tests/test_parallel.py` — 6 novos testes

| Teste | O que valida |
|-------|-------------|
| `test_worker_returns_warmup_result` | Worker com process_fn de warmup retorna `is_warmup=True` |
| `test_worker_returns_inference_result` | Worker retorna `znorm_text` e `is_calibrated=True` |
| `test_force_calibrate_event` | `threading.Event` set/clear funciona corretamente |
| `test_worker_stats_reflect_real_processing` | `avg_process_ms > 0` após processar frames |
| `test_pipeline_result_new_fields` | Novos campos `znorm_text`/`is_calibrated` preenchidos |
| `test_pipeline_result_default_fields` | Defaults: `znorm_text=""`, `is_calibrated=False` |

---

## Diagrama de Threads — Antes vs Depois

### ANTES (P2 não plugado)
```
Thread FrameGrabber:  camera.read() → Queue
Thread Main:          Queue.get() → [extract → calibrate → window → predict
                                      → guardrails → reflection → agents
                                      → memory → overlay → display]
```

### DEPOIS (P2 plugado)
```
Thread FrameGrabber:   camera.read() → frame_queue
Thread PipelineWorker: frame_queue → [extract → calibrate → window → predict
                                       → guardrails → reflection → agents
                                       → memory] → result_queue
Thread Main:           result_queue.get() → display + keyboard
```

---

## Resultados dos Testes

```
119 passed in 3.40s
```

Todos os 119 testes passaram:
- 24 guardrails (G1-G4)
- 18 reflection (R1-R3)
- 26 memory (M1-M3)
- 21 parallel (P1-P3 + 6 integração P2)
- 30 agents (A1-A5)
