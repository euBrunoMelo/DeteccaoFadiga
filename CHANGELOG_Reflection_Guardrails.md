# Changelog: Implementação Reflection + Guardrails

**Data:** 2026-03-11
**Branch:** plano-de-migracao
**Baseado em:** Plano_Execucao_Reflection_Guardrails.md

---

## Resumo

Implementação completa dos padrões agênticos **Guardrails (G1-G4)** e **Reflection (R1-R3)** no pipeline de inferência DeteccaoFadiga2. O sistema evolui de um classificador binário reativo para um pipeline com auto-avaliação, validação estruturada e auto-correção.

---

## Arquivos Criados

### `SALTE_INFERENCE/guardrails.py`
Módulo completo de guardrails com 4 camadas:

| Fase | Componente | Descrição |
|------|-----------|-----------|
| **G1** | `FatigueOutput` | Dataclass validada que substitui a tupla `(float, str)`. Valida `prob_danger` em `[0,1]`, `label` em `{"Safe","Danger"}`, `window_quality` em `[0,1]` via `__post_init__`. |
| **G1** | `AlertLevel` | Enum com 4 níveis graduados: `SAFE`, `WATCH`, `DANGER`, `CRITICAL`. |
| **G1** | `validate_and_wrap()` | Função principal que recebe a predição raw e retorna `FatigueOutput` validado com alert_level, confidence e métricas. |
| **G2** | `FEATURE_RANGES` | Dicionário com ranges fisiológicos das 19 features (baseados em training_stats + 20% margem). |
| **G2** | `_check_feature_ranges()` | Valida se todas as features estão dentro dos ranges. Loga `[guardrail] WARN` para violações. |
| **G2** | `_compute_confidence()` | Determina confiança (`high`/`medium`/`low`) baseado em validação de features e atividade ocular. |
| **G3** | `BehaviorGuardRails` | Classe stateful que aplica restrições comportamentais: grace period pós-calibração (2 janelas), escalação para CRITICAL após 20 Dangers consecutivos, rate limiting de alertas sonoros (60s cooldown), watchdog de pipeline (30s timeout). |
| **G4** | `CalibrationVerdict` | Dataclass com resultado da validação de calibração. |
| **G4** | `validate_calibration()` | Valida `SubjectBaseline` contra critérios fisiológicos: EAR mean `[0.15, 0.40]`, EAR std `[0.015, 0.10]`, Pitch std `< 25°`, Yaw mean `< 30°`. Retorna `"accept"`, `"retry"` ou `"use_with_caution"`. |

### `SALTE_INFERENCE/reflection.py`
Módulo completo de reflection (Producer-Critic) com 3 componentes:

| Fase | Componente | Descrição |
|------|-----------|-----------|
| **R1** | `DriftReflector` | Monitora drift nas features ao longo do tempo. Mantém buffer circular das últimas 20 janelas. A cada 10 janelas, calcula z-score da média recente vs training_stats. Classifica em `STABLE`/`WARNING`/`DRIFTED`/`CRITICAL` com thresholds configuráveis (1.5/2.5/4.0). |
| **R1** | `DriftReport` | Dataclass com status, features driftadas, z-scores e recomendação (`continue`/`recalibrate`/`suspend`). |
| **R2** | `PredictionReflector` | Analisa padrões temporais nas predições. Detecta: `stuck_danger` (15+ Dangers consecutivos), `oscillating` (taxa de transição > 0.4), `sudden_transition` (delta > 0.4 em 3 janelas). Retorna `confidence_modifier` para ajustar confiança. |
| **R3** | `AutoRecalibrationManager` | Combina sinais de drift e predição para decidir auto-recalibração. Regras: CRITICAL → imediato, DRIFTED + stuck → imediato, DRIFTED sozinho → agendado, stuck sem drift → não recalibra (pode ser fadiga real). Safety caps: cooldown 5min, max 4/hora. |

### `SALTE_INFERENCE/tests/test_guardrails.py`
24 testes unitários cobrindo G1-G4:
- `TestFatigueOutputG1`: validação de saída, exceções para valores inválidos
- `TestFeatureRangesG2`: ranges, confiança, detecção de violações
- `TestValidateAndWrap`: integração G1+G2, WATCH zone, CRITICAL com microsleep
- `TestBehaviorGuardRailsG3`: grace period, escalação, cooldown, watchdog
- `TestCalibrationVerdictG4`: aceitação, rejeição, cautela para cada critério

### `SALTE_INFERENCE/tests/test_reflection.py`
18 testes unitários cobrindo R1-R3:
- `TestDriftReflectorR1`: stable, warning, drifted, critical, z-score math
- `TestPredictionReflectorR2`: stable, stuck_danger, oscillating, sudden_transition
- `TestAutoRecalibrationR3`: todas as regras de decisão, cooldown, rate limit

---

## Arquivos Modificados

### `SALTE_INFERENCE/run_realtime_demo.py`
Integração completa dos novos módulos no loop principal:

1. **Imports adicionados:** `json`, `guardrails`, `reflection` (e seus componentes)
2. **Inicialização** (após `load_best_model`): carrega `training_stats` do JSON, instancia `BehaviorGuardRails`, `DriftReflector`, `PredictionReflector`, `AutoRecalibrationManager`
3. **Pós-calibração** (G4): `validate_calibration()` valida o baseline antes de aceitar. Se `"retry"` → reinicia warm-up. Se `"use_with_caution"` → loga warnings. Chama `behavior_guard.on_calibration_complete()`
4. **Pós-inferência** (G1+G2+G3): `validate_and_wrap()` empacota a predição → `behavior_guard.process()` aplica restrições comportamentais
5. **Reflection** (R1+R2+R3): `drift_reflector.push_window()` + `pred_reflector.push()` a cada janela. Quando drift é reportado, `recal_manager.evaluate()` decide se recalibra. Se urgência `"immediate"` → reset completo do pipeline
6. **Alertas**: `behavior_guard.should_sound_alert()` controla rate limiting de alertas sonoros
7. **Overlay**: status_text agora inclui confidence level; cor diferenciada para CRITICAL (dark red) e WATCH (orange); overlay2 inclui `Alert:LEVEL`

### `SALTE_INFERENCE/__init__.py`
Docstring atualizada com descrição dos novos módulos `guardrails` e `reflection`.

---

## Resultados dos Testes

```
42 passed in 0.70s
```

Todos os 42 testes passaram (24 guardrails + 18 reflection).

---

## Estrutura Final

```
SALTE_INFERENCE/
├── __init__.py                    (atualizado)
├── feature_extractor_rt.py        (sem alterações)
├── subject_calibrator_rt.py       (sem alterações)
├── window_factory_rt.py           (sem alterações)
├── model_loader.py                (sem alterações)
├── guardrails.py                  ★ NOVO — G1/G2/G3/G4
├── reflection.py                  ★ NOVO — R1/R2/R3
├── run_realtime_demo.py           (modificado — integração completa)
├── offline_eval.py                (sem alterações)
└── tests/                         ★ NOVO
    ├── __init__.py
    ├── test_guardrails.py         ★ 24 testes
    └── test_reflection.py         ★ 18 testes
```
