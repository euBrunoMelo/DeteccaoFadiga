# Plano de Execução: Reflection + Guardrails para DeteccaoFadiga2

## Implementação dos Padrões Agênticos Prioritários

**Projeto:** DeteccaoFadiga2  
**Escopo:** 2 padrões — Reflection (Auto-Correção) + Guardrails (Safety)  
**Estimativa total:** ~5 sprints de 1 semana  
**Pré-requisito:** Python 3.10+, numpy, onnxruntime (já presentes no projeto)

---

## PARTE A — GUARDRAILS / SAFETY

Guardrails são mecanismos de proteção que garantem que o sistema opera dentro de limites seguros e previsíveis. No DeteccaoFadiga2, guardrails atuam em três camadas: validação de entrada (features), validação de saída (predições), e restrições de comportamento (alertas e sistema).

### Fase G1 — Validação Estruturada de Saída (Semana 1)

**Objetivo:** Garantir que toda predição emitida pelo sistema seja consistente, tipada e dentro dos ranges esperados. Hoje, `predict_fatigue()` retorna uma tupla simples `(float, str)` sem nenhuma validação.

**Arquivo novo:** `SALTE_INFERENCE/guardrails.py`

```python
"""
Guardrails para o pipeline de inferência DeteccaoFadiga2.

Três camadas de proteção:
  G1. Validação estruturada de saída (Pydantic-like, sem dependência)
  G2. Validação de entrada (features dentro de ranges fisiológicos)
  G3. Restrições comportamentais (rate limit de alertas, watchdog)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import time
import numpy as np


# ── G1: Validação Estruturada de Saída ────────────────────────────────────


class AlertLevel(Enum):
    """Níveis de alerta graduados (ao invés de binário Safe/Danger)."""
    SAFE = 0
    WATCH = 1       # prob entre low_thresh e threshold
    DANGER = 2      # prob >= threshold
    CRITICAL = 3    # Danger consecutivo ou microsleep


@dataclass
class FatigueOutput:
    """Saída validada de uma inferência. Substitui a tupla (float, str)."""
    label: str                          # "Safe" | "Danger"
    prob_danger: float                  # [0.0, 1.0]
    alert_level: AlertLevel
    features_valid: bool                # todas 19 features dentro do range?
    confidence: str                     # "high" | "medium" | "low"
    window_quality: float               # ratio de frames válidos na janela
    timestamp_ms: float
    
    # Métricas-chave para overlay e logging
    perclos: float = 0.0
    blink_count: float = 0.0
    microsleep_count: float = 0.0
    
    def __post_init__(self):
        """Validação automática no momento da criação."""
        errors = []
        if not (0.0 <= self.prob_danger <= 1.0):
            errors.append(
                f"prob_danger={self.prob_danger} fora de [0,1]"
            )
        if self.label not in ("Safe", "Danger"):
            errors.append(f"label='{self.label}' inválido")
        if not (0.0 <= self.window_quality <= 1.0):
            errors.append(
                f"window_quality={self.window_quality} fora de [0,1]"
            )
        if errors:
            raise ValueError(
                f"FatigueOutput inválido: {'; '.join(errors)}"
            )
```

**Integração no `run_realtime_demo.py`:**

Localizar o bloco onde `predict_fatigue()` é chamado (linha ~369-372) e envolver o resultado:

```python
# ANTES (linha 369-372 de run_realtime_demo.py):
prob_danger, label = predict_fatigue(
    vec, model, config, threshold_override=threshold
)

# DEPOIS:
prob_danger, label = predict_fatigue(
    vec, model, config, threshold_override=threshold
)
output = guardrails.validate_and_wrap(
    prob_danger=prob_danger,
    label=label,
    window_feats=window_feats,
    feature_names=feature_names,
    config=config,
    timestamp_ms=calibrated.timestamp_ms,
)
# output é um FatigueOutput validado, com alert_level e confidence
status_text = f"{output.label} ({output.prob_danger:.2f}) [{output.confidence}]"
```

**Função de validação e wrap no `guardrails.py`:**

```python
def validate_and_wrap(
    prob_danger: float,
    label: str,
    window_feats: Dict[str, float],
    feature_names: List[str],
    config,  # InferenceConfig
    timestamp_ms: float,
    threshold: float = 0.41,
    watch_threshold: float = 0.30,
) -> FatigueOutput:
    """Valida a predição e retorna FatigueOutput estruturado."""
    
    # Clamp de segurança (nunca deveria ser necessário, mas protege)
    prob_clamped = float(np.clip(prob_danger, 0.0, 1.0))
    
    # Determinar alert_level graduado
    if prob_clamped >= threshold:
        alert_level = AlertLevel.DANGER
    elif prob_clamped >= watch_threshold:
        alert_level = AlertLevel.WATCH
    else:
        alert_level = AlertLevel.SAFE
    
    # Checar se microsleep eleva para CRITICAL
    micros = window_feats.get("microsleep_count", 0.0)
    if micros > 0 and alert_level == AlertLevel.DANGER:
        alert_level = AlertLevel.CRITICAL
    
    # Validar ranges das features
    features_valid = _check_feature_ranges(window_feats, feature_names)
    
    # Determinar confiança
    confidence = _compute_confidence(window_feats, features_valid)
    
    return FatigueOutput(
        label=label,
        prob_danger=prob_clamped,
        alert_level=alert_level,
        features_valid=features_valid,
        confidence=confidence,
        window_quality=1.0,  # será calculado pelo window_factory
        timestamp_ms=timestamp_ms,
        perclos=window_feats.get("perclos_p80_mean", 0.0),
        blink_count=window_feats.get("blink_count", 0.0),
        microsleep_count=micros,
    )
```

**Critério de aceite G1:**

- `FatigueOutput.__post_init__` levanta `ValueError` se `prob_danger` estiver fora de `[0, 1]`
- Toda inferência passa por `validate_and_wrap()` antes de ser usada no overlay
- Teste unitário: criar outputs com valores inválidos e confirmar que a exceção é levantada

---

### Fase G2 — Validação de Entrada (Features) (Semana 1)

**Objetivo:** Detectar features fora dos ranges fisiológicos ou do range de treinamento antes que cheguem ao modelo. Hoje, o modelo recebe qualquer vetor de 19 floats sem validação.

**Função no `guardrails.py`:**

```python
# Ranges baseados em training_stats do inference_config.json
# Usa min/max do treino com margem de 20%
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "ear_mean":      (-18.0, 6.0),      # treino: [-14.9, 4.4]
    "ear_std":       (-2.5, 16.0),       # treino: [-1.7, 13.0]
    "ear_min":       (-10.0, 3.0),       # treino: [-8.0, 1.9]
    "ear_vel_mean":  (-33.0, 30.0),      # treino: [-27.5, 24.2]
    "ear_vel_std":   (-2.0, 16.0),       # treino: [-1.1, 13.2]
    "mar_mean":      (-2.0, 28.0),       # treino: [-1.3, 23.5]
    "pitch_mean":    (-11.0, 10.0),      # treino: [-8.6, 8.2]
    "pitch_std":     (-1.0, 31.0),       # treino: [-0.7, 25.3]
    "yaw_std":       (-1.0, 20.0),       # treino: [-0.6, 16.3]
    "roll_std":      (-1.0, 27.0),       # treino: [-0.6, 22.3]
    "blink_count":   (0.0, 35.0),        # treino: [0, 26]
    "blink_rate_per_min": (0.0, 130.0),  # treino: [0, 104]
    "blink_mean_dur_ms":  (0.0, 24000.0),# treino: [0, 19300]
    "perclos_p80_mean":   (0.0, 1.0),
    "perclos_p80_max":    (0.0, 1.0),
    "blink_closing_vel_mean": (-2.0, 7.0),  # treino: [-1.3, 5.0]
    "blink_opening_vel_mean": (-2.0, 11.0), # treino: [-1.4, 8.5]
    "long_blink_pct":     (0.0, 1.0),
    "blink_regularity":   (0.0, 3.0),       # treino: [0, 2.2]
}


def _check_feature_ranges(
    window_feats: Dict[str, float],
    feature_names: List[str],
) -> bool:
    """
    Retorna True se TODAS as features estão dentro dos ranges esperados.
    Loga warnings para features fora do range.
    """
    all_valid = True
    for name in feature_names:
        val = window_feats.get(name)
        if val is None:
            all_valid = False
            continue
        lo, hi = FEATURE_RANGES.get(name, (-np.inf, np.inf))
        if not (lo <= val <= hi):
            print(
                f"[guardrail] WARN: {name}={val:.4f} "
                f"fora do range [{lo}, {hi}]"
            )
            all_valid = False
    return all_valid


def _compute_confidence(
    window_feats: Dict[str, float],
    features_valid: bool,
) -> str:
    """
    Determina confiança da predição baseado na qualidade dos dados.
    
    high:   features válidas + blink_count > 0 (houve atividade ocular)
    medium: features válidas mas blink_count == 0 (pode ser estático)
    low:    alguma feature fora do range esperado
    """
    if not features_valid:
        return "low"
    blinks = window_feats.get("blink_count", 0.0)
    if blinks > 0:
        return "high"
    return "medium"
```

**Critério de aceite G2:**

- Feature fora do range loga `[guardrail] WARN` mas não bloqueia inferência (graceful degradation)
- `FatigueOutput.features_valid` reflete o status — overlay mostra `[low confidence]` se falso
- Teste: injetar `ear_mean = -20.0` (abaixo do min de treino) e verificar que o warning aparece

---

### Fase G3 — Restrições Comportamentais (Semana 2)

**Objetivo:** Proteger contra comportamentos indesejados do sistema como um todo: alertas excessivos, crash silencioso, e instabilidade pós-calibração.

**Classe no `guardrails.py`:**

```python
@dataclass
class BehaviorGuardrailConfig:
    """Configuração das restrições comportamentais."""
    alert_cooldown_sec: float = 60.0     # Min 60s entre alertas sonoros
    post_calibration_grace_windows: int = 2  # Ignorar 2 janelas pós-calibração
    watchdog_timeout_sec: float = 30.0   # Alerta se sem inferência por 30s
    max_consecutive_danger: int = 20     # Safety cap — forçar ação após 20 Dangers
    min_face_ratio_for_inference: float = 0.50  # Abaixo disso, suspender


class BehaviorGuardRails:
    """
    Guardrails comportamentais — opera sobre o fluxo de saídas ao longo
    do tempo, não sobre uma única predição.
    """
    
    def __init__(
        self, config: Optional[BehaviorGuardrailConfig] = None
    ) -> None:
        self.cfg = config or BehaviorGuardrailConfig()
        self._last_alert_time: float = 0.0
        self._last_inference_time: float = time.monotonic()
        self._windows_since_calibration: int = 0
        self._consecutive_danger: int = 0
        self._calibration_just_completed: bool = False
        self._total_outputs: int = 0
        self._total_suppressed: int = 0
    
    def on_calibration_complete(self) -> None:
        """Chamado quando calibração termina. Inicia grace period."""
        self._calibration_just_completed = True
        self._windows_since_calibration = 0
    
    def process(self, output: FatigueOutput) -> FatigueOutput:
        """
        Aplica guardrails comportamentais sobre um FatigueOutput.
        
        Pode:
        - Rebaixar alert_level durante grace period pós-calibração
        - Elevar para CRITICAL se consecutive_danger > max
        - Suprimir flag de alerta sonoro se dentro do cooldown
        
        Retorna FatigueOutput (possivelmente modificado).
        """
        now = time.monotonic()
        self._last_inference_time = now
        self._total_outputs += 1
        
        # (1) Grace period pós-calibração
        if self._calibration_just_completed:
            self._windows_since_calibration += 1
            if self._windows_since_calibration <= self.cfg.post_calibration_grace_windows:
                # Rebaixar qualquer Danger para WATCH durante grace
                if output.alert_level in (
                    AlertLevel.DANGER, AlertLevel.CRITICAL
                ):
                    output.alert_level = AlertLevel.WATCH
                    output.confidence = "low"
                    self._total_suppressed += 1
                return output
            else:
                self._calibration_just_completed = False
        
        # (2) Tracking de consecutive danger
        if output.label == "Danger":
            self._consecutive_danger += 1
        else:
            self._consecutive_danger = 0
        
        # (3) Elevar para CRITICAL se muitos Dangers consecutivos
        if self._consecutive_danger >= self.cfg.max_consecutive_danger:
            output.alert_level = AlertLevel.CRITICAL
        
        return output
    
    def should_sound_alert(self, output: FatigueOutput) -> bool:
        """
        Rate limiter: retorna True somente se o alerta sonoro 
        é permitido (respeitando cooldown).
        """
        if output.alert_level.value < AlertLevel.DANGER.value:
            return False
        
        now = time.monotonic()
        elapsed = now - self._last_alert_time
        if elapsed < self.cfg.alert_cooldown_sec:
            return False
        
        self._last_alert_time = now
        return True
    
    def check_watchdog(self) -> bool:
        """
        Retorna True se o pipeline está saudável.
        Retorna False se sem inferência por mais de watchdog_timeout_sec.
        Deve ser chamado periodicamente (ex: a cada 5s de um timer).
        """
        elapsed = time.monotonic() - self._last_inference_time
        if elapsed > self.cfg.watchdog_timeout_sec:
            print(
                f"[guardrail] WATCHDOG: sem inferência há "
                f"{elapsed:.1f}s (timeout={self.cfg.watchdog_timeout_sec}s)"
            )
            return False
        return True
    
    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_outputs": self._total_outputs,
            "total_suppressed": self._total_suppressed,
            "consecutive_danger": self._consecutive_danger,
        }
```

**Integração no `run_realtime_demo.py`:**

```python
# No início de run_realtime():
behavior_guard = BehaviorGuardRails()

# Após calibração (linha ~269):
if calibrator.is_calibrated:
    behavior_guard.on_calibration_complete()
    ...

# Após validate_and_wrap() (nova inserção):
output = guardrails.validate_and_wrap(...)
output = behavior_guard.process(output)

# No bloco de display:
if behavior_guard.should_sound_alert(output):
    # Acionar buzzer / som (GPIO ou beep)
    pass
```

**Critério de aceite G3:**

- As 2 primeiras janelas após calibração nunca emitem Danger (rebaixadas para WATCH)
- Alertas sonoros respeitam cooldown de 60s mesmo com Danger contínuo
- Se o loop principal travar, `check_watchdog()` retorna False após 30s
- Teste: simular 25 Dangers consecutivos e verificar que `alert_level` sobe para CRITICAL

---

### Fase G4 — Guardrail de Calibração (Semana 2)

**Objetivo:** Validar a qualidade da calibração antes de aceitar o baseline. Hoje, `_compute_baseline()` aceita qualquer resultado, inclusive baselines com EAR fisiologicamente impossível.

**Função no `guardrails.py`:**

```python
@dataclass
class CalibrationVerdict:
    """Resultado da validação da calibração."""
    is_acceptable: bool
    issues: List[str]
    recommendation: str  # "accept" | "retry" | "use_with_caution"


def validate_calibration(baseline) -> CalibrationVerdict:
    """
    Valida um SubjectBaseline contra critérios fisiológicos e estatísticos.
    
    Critérios:
    1. EAR mean entre [0.15, 0.40] — olhos humanos normais
    2. EAR std entre [0.015, 0.10] — se muito baixo, não houve piscadas
    3. Pitch std < 30° — se maior, dados corrompidos
    4. baseline.is_valid == True
    5. Segmento de baseline tem duração razoável
    """
    issues = []
    
    if not baseline.is_valid:
        issues.append("Baseline marcado como inválido pelo calibrador")
    
    # EAR mean fisiológico
    if baseline.ear_mean < 0.15:
        issues.append(
            f"EAR mean={baseline.ear_mean:.4f} muito baixo (<0.15). "
            f"Possível: olhos parcialmente fechados durante warm-up ou "
            f"má detecção de landmarks"
        )
    elif baseline.ear_mean > 0.40:
        issues.append(
            f"EAR mean={baseline.ear_mean:.4f} muito alto (>0.40). "
            f"Possível: artefato de landmark (rosto parcial)"
        )
    
    # EAR std — deve ter variação (blinks)
    if baseline.ear_std < 0.015:
        issues.append(
            f"EAR std={baseline.ear_std:.4f} muito baixo. "
            f"Operador pode não ter piscado durante warm-up"
        )
    elif baseline.ear_std > 0.10:
        issues.append(
            f"EAR std={baseline.ear_std:.4f} muito alto. "
            f"Instabilidade nos landmarks ou iluminação variável"
        )
    
    # Pitch std — sanity check pós-sanitizer
    if baseline.pitch_std > 25.0:
        issues.append(
            f"Pitch std={baseline.pitch_std:.2f}° excessivo. "
            f"HeadPoseSanitizer pode não estar corrigindo flip PnP"
        )
    
    # Yaw/Roll — operador deve estar relativamente frontal
    if abs(baseline.yaw_mean) > 30.0:
        issues.append(
            f"Yaw mean={baseline.yaw_mean:.2f}° — operador não "
            f"frontal à câmera durante warm-up"
        )
    
    # Determinar recomendação
    if not issues:
        return CalibrationVerdict(
            is_acceptable=True,
            issues=[],
            recommendation="accept",
        )
    
    critical = any(
        "muito baixo (<0.15)" in i or "inválido" in i for i in issues
    )
    if critical:
        return CalibrationVerdict(
            is_acceptable=False,
            issues=issues,
            recommendation="retry",
        )
    
    return CalibrationVerdict(
        is_acceptable=True,
        issues=issues,
        recommendation="use_with_caution",
    )
```

**Integração no `run_realtime_demo.py`:**

```python
# Após calibração (substituir bloco da linha ~269-313):
if calibrator.is_calibrated:
    b = calibrator.baseline
    verdict = guardrails.validate_calibration(b)
    
    if verdict.recommendation == "retry":
        print(f"[guardrail] Calibração REJEITADA: {verdict.issues}")
        print("[guardrail] Reiniciando warm-up...")
        # Reset calibrator e reiniciar warm-up
        calibrator = RTSubjectCalibrator(
            CalibrationConfig(fps=fps, search_sec=warmup_sec)
        )
        continue
    
    if verdict.recommendation == "use_with_caution":
        print(f"[guardrail] Calibração ACEITA com ressalvas:")
        for issue in verdict.issues:
            print(f"  ⚠ {issue}")
    
    behavior_guard.on_calibration_complete()
    ...
```

**Critério de aceite G4:**

- EAR mean de 0.08 (impossível) → calibração rejeitada, warm-up reiniciado
- EAR std de 0.005 (sem piscadas) → aceita com warning
- Pitch std de 131° (bug conhecido do solvePnP) → calibração rejeitada
- Teste: criar `SubjectBaseline` com valores extremos e rodar `validate_calibration()`

---

## PARTE B — REFLECTION (AUTO-CORREÇÃO)

O padrão Reflection cria um loop de feedback: o sistema gera um resultado, um componente "crítico" avalia esse resultado, e se houver problemas, o sistema se corrige. No contexto de visão computacional em tempo real, reflection não usa LLM — usa regras estatísticas determinísticas como "crítico".

### Fase R1 — Detector de Drift (Semana 3)

**Objetivo:** Detectar quando as distribuições de features se desviam significativamente do baseline de calibração. Drift acontece quando a iluminação muda, a câmera se move, ou o operador ajusta o assento.

**Arquivo novo:** `SALTE_INFERENCE/reflection.py`

```python
"""
Módulo de Reflection (Auto-Correção) para DeteccaoFadiga2.

Implementa o padrão Producer-Critic em três componentes:
  R1. DriftReflector   — detecta drift nas features ao longo do tempo
  R2. CalibrationCritic — avalia e critica a calibração com dados novos
  R3. PredictionReflector — detecta padrões anômalos nas predições

O "Producer" é o pipeline normal de inferência.
O "Critic" é este módulo, que opera a cada N janelas.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class DriftStatus(Enum):
    STABLE = "stable"
    WARNING = "warning"      # drift leve — logar
    DRIFTED = "drifted"      # drift severo — sugerir recalibração
    CRITICAL = "critical"    # drift extremo — suspender inferência


@dataclass
class DriftReport:
    """Relatório de drift para uma feature ou conjunto de features."""
    status: DriftStatus
    drifted_features: List[str]
    details: Dict[str, float]    # feature_name → zscore_from_baseline
    recommendation: str          # "continue" | "recalibrate" | "suspend"
    windows_analyzed: int


@dataclass 
class DriftReflectorConfig:
    """Configuração do detector de drift."""
    analysis_interval: int = 10     # Analisar a cada 10 janelas (~2.5 min)
    window_buffer_size: int = 20    # Manter últimas 20 janelas para análise
    warning_threshold: float = 1.5  # |z-score da média| > 1.5 → warning
    drift_threshold: float = 2.5    # |z-score da média| > 2.5 → drifted
    critical_threshold: float = 4.0 # |z-score da média| > 4.0 → critical
    # Features monitoradas para drift (as mais sensíveis a mudanças externas)
    monitored_features: Tuple[str, ...] = (
        "ear_mean", "ear_std", "mar_mean",
        "pitch_mean", "pitch_std", "yaw_std",
    )


class DriftReflector:
    """
    R1: Critic que monitora drift entre as features recentes e o baseline.
    
    Lógica:
    - Mantém buffer circular das últimas N janelas de features
    - A cada analysis_interval janelas, compara a média recente
      contra os training_stats do inference_config.json
    - Se a média recente se desviou além do threshold, reporta drift
    
    O pipeline (Producer) é responsável por agir sobre o DriftReport.
    
    Ref: Agentic Design Patterns Ch.4 — Producer-Critic model.
    """
    
    def __init__(
        self,
        training_stats: Dict[str, Dict[str, float]],
        config: Optional[DriftReflectorConfig] = None,
    ) -> None:
        """
        Args:
            training_stats: Dict do inference_config.json["training_stats"]
                Cada entry tem {"mean", "std", "min", "max"}.
            config: Configuração do reflector.
        """
        self.cfg = config or DriftReflectorConfig()
        self._training_stats = training_stats
        self._window_buffer: deque = deque(
            maxlen=self.cfg.window_buffer_size
        )
        self._windows_since_last_analysis: int = 0
        self._total_analyses: int = 0
        self._last_report: Optional[DriftReport] = None
    
    def push_window(
        self, window_feats: Dict[str, float]
    ) -> Optional[DriftReport]:
        """
        Alimenta uma janela de features. Retorna DriftReport quando é
        hora de analisar, ou None caso contrário.
        """
        self._window_buffer.append(window_feats)
        self._windows_since_last_analysis += 1
        
        if self._windows_since_last_analysis < self.cfg.analysis_interval:
            return None
        
        if len(self._window_buffer) < 5:
            return None  # dados insuficientes
        
        self._windows_since_last_analysis = 0
        self._total_analyses += 1
        
        report = self._analyze()
        self._last_report = report
        return report
    
    def _analyze(self) -> DriftReport:
        """
        Compara médias recentes vs training_stats usando z-score.
        
        Para cada feature monitorada:
          z = (mean_recente - mean_treino) / std_treino
        
        Se |z| > threshold para qualquer feature → drift.
        """
        drifted = []
        details = {}
        worst_status = DriftStatus.STABLE
        
        for feat_name in self.cfg.monitored_features:
            stats = self._training_stats.get(feat_name)
            if stats is None:
                continue
            
            train_mean = stats["mean"]
            train_std = stats["std"]
            if train_std < 1e-8:
                continue
            
            # Média recente desta feature
            recent_vals = [
                w[feat_name] for w in self._window_buffer
                if feat_name in w
            ]
            if not recent_vals:
                continue
            
            recent_mean = float(np.mean(recent_vals))
            z = (recent_mean - train_mean) / train_std
            details[feat_name] = round(z, 3)
            
            abs_z = abs(z)
            if abs_z >= self.cfg.critical_threshold:
                drifted.append(feat_name)
                if DriftStatus.CRITICAL.value > worst_status.value:
                    worst_status = DriftStatus.CRITICAL
            elif abs_z >= self.cfg.drift_threshold:
                drifted.append(feat_name)
                if worst_status != DriftStatus.CRITICAL:
                    worst_status = DriftStatus.DRIFTED
            elif abs_z >= self.cfg.warning_threshold:
                if worst_status in (
                    DriftStatus.STABLE,
                ):
                    worst_status = DriftStatus.WARNING
        
        # Definir recomendação
        if worst_status == DriftStatus.CRITICAL:
            rec = "suspend"
        elif worst_status == DriftStatus.DRIFTED:
            rec = "recalibrate"
        else:
            rec = "continue"
        
        return DriftReport(
            status=worst_status,
            drifted_features=drifted,
            details=details,
            recommendation=rec,
            windows_analyzed=len(self._window_buffer),
        )
    
    @property
    def last_report(self) -> Optional[DriftReport]:
        return self._last_report
```

**Integração no `run_realtime_demo.py`:**

```python
import json

# Na inicialização (após carregar config):
with open(config_path) as f:
    raw_config = json.load(f)
training_stats = raw_config.get("training_stats", {})
drift_reflector = DriftReflector(training_stats)

# Após cada window_feats (linha ~334):
if window_feats is not None:
    drift_report = drift_reflector.push_window(window_feats)
    
    if drift_report is not None:
        if drift_report.status != DriftStatus.STABLE:
            print(
                f"[reflection] Drift {drift_report.status.value}: "
                f"{drift_report.drifted_features}"
            )
            print(f"[reflection] Z-scores: {drift_report.details}")
            print(f"[reflection] Recomendação: {drift_report.recommendation}")
        
        if drift_report.recommendation == "recalibrate":
            # Agir: pode exibir no overlay, logar, ou auto-recalibrar
            pass
```

**Critério de aceite R1:**

- Com dados estáveis, `DriftReflector` reporta `STABLE` consistentemente
- Ao simular shift de iluminação (EAR cai 30%), detecta `DRIFTED` em `ear_mean` dentro de 3 análises
- Z-scores no report são matematicamente corretos (verificar com cálculo manual)

---

### Fase R2 — Crítico de Predição (Semana 4)

**Objetivo:** Detectar padrões anômalos nas sequências de predições que indicam falso alarme ou falsa segurança.

**Classe no `reflection.py`:**

```python
@dataclass
class PredictionReflection:
    """Resultado da reflexão sobre padrões de predição."""
    pattern: str           # "stable" | "oscillating" | "stuck_danger" | "sudden_transition"
    confidence_modifier: float  # Multiplicador de confiança: 1.0 = sem ajuste
    suggestion: str        # Texto explicando o padrão
    consecutive_danger: int
    consecutive_safe: int
    recent_prob_mean: float
    recent_prob_std: float


class PredictionReflector:
    """
    R2: Critic que analisa padrões temporais nas sequências de predições.
    
    Detecta:
    - Oscilação rápida Safe/Danger (indica threshold na zona de fronteira)
    - Stuck em Danger por muito tempo (possível falso positivo crônico)
    - Transição abrupta Safe→Danger sem features intermediárias
    """
    
    def __init__(self, buffer_size: int = 30) -> None:
        self._prob_buffer: deque = deque(maxlen=buffer_size)
        self._label_buffer: deque = deque(maxlen=buffer_size)
        self._consecutive_danger: int = 0
        self._consecutive_safe: int = 0
    
    def push(
        self, prob_danger: float, label: str
    ) -> PredictionReflection:
        """Alimenta uma predição e retorna reflexão."""
        self._prob_buffer.append(prob_danger)
        self._label_buffer.append(label)
        
        if label == "Danger":
            self._consecutive_danger += 1
            self._consecutive_safe = 0
        else:
            self._consecutive_safe += 1
            self._consecutive_danger = 0
        
        return self._reflect()
    
    def _reflect(self) -> PredictionReflection:
        if len(self._prob_buffer) < 5:
            return PredictionReflection(
                pattern="stable",
                confidence_modifier=1.0,
                suggestion="Dados insuficientes para reflexão",
                consecutive_danger=self._consecutive_danger,
                consecutive_safe=self._consecutive_safe,
                recent_prob_mean=0.0,
                recent_prob_std=0.0,
            )
        
        probs = np.array(self._prob_buffer)
        labels = list(self._label_buffer)
        prob_mean = float(probs.mean())
        prob_std = float(probs.std())
        
        # Detectar oscilação: muitas transições Safe<->Danger
        transitions = sum(
            1 for i in range(1, len(labels))
            if labels[i] != labels[i - 1]
        )
        oscillation_rate = transitions / max(len(labels) - 1, 1)
        
        # Detectar stuck em Danger
        if self._consecutive_danger >= 15:
            return PredictionReflection(
                pattern="stuck_danger",
                confidence_modifier=0.7,
                suggestion=(
                    f"Danger contínuo por {self._consecutive_danger} janelas. "
                    f"Verificar se é fadiga real ou drift de calibração. "
                    f"Considerar recalibração."
                ),
                consecutive_danger=self._consecutive_danger,
                consecutive_safe=self._consecutive_safe,
                recent_prob_mean=prob_mean,
                recent_prob_std=prob_std,
            )
        
        # Detectar oscilação
        if oscillation_rate > 0.4 and len(labels) >= 10:
            return PredictionReflection(
                pattern="oscillating",
                confidence_modifier=0.5,
                suggestion=(
                    f"Oscilação Safe/Danger detectada "
                    f"(taxa={oscillation_rate:.2f}). "
                    f"Operador pode estar no limiar de fadiga. "
                    f"prob_mean={prob_mean:.3f} — próximo do threshold."
                ),
                consecutive_danger=self._consecutive_danger,
                consecutive_safe=self._consecutive_safe,
                recent_prob_mean=prob_mean,
                recent_prob_std=prob_std,
            )
        
        # Detectar transição abrupta
        if len(probs) >= 3:
            last_3 = probs[-3:]
            delta = float(last_3[-1] - last_3[0])
            if abs(delta) > 0.4:
                return PredictionReflection(
                    pattern="sudden_transition",
                    confidence_modifier=0.8,
                    suggestion=(
                        f"Transição abrupta de probabilidade: "
                        f"delta={delta:+.3f} em 3 janelas. "
                        f"Verificar mudança brusca de condições."
                    ),
                    consecutive_danger=self._consecutive_danger,
                    consecutive_safe=self._consecutive_safe,
                    recent_prob_mean=prob_mean,
                    recent_prob_std=prob_std,
                )
        
        return PredictionReflection(
            pattern="stable",
            confidence_modifier=1.0,
            suggestion="Padrão de predição estável",
            consecutive_danger=self._consecutive_danger,
            consecutive_safe=self._consecutive_safe,
            recent_prob_mean=prob_mean,
            recent_prob_std=prob_std,
        )
```

**Integração no loop principal:**

```python
# Na inicialização:
pred_reflector = PredictionReflector()

# Após predict_fatigue():
reflection = pred_reflector.push(prob_danger, label)

if reflection.pattern != "stable":
    print(
        f"[reflection] Padrão: {reflection.pattern} | "
        f"Sugestão: {reflection.suggestion}"
    )

# Usar confidence_modifier para ajustar o overlay
if reflection.confidence_modifier < 1.0:
    output.confidence = "low"
```

**Critério de aceite R2:**

- 15+ Dangers consecutivos → detecta `stuck_danger`, loga sugestão de recalibração
- Sequência alternada Safe/Danger/Safe/Danger por 10+ janelas → detecta `oscillating`
- Salto de prob 0.20 → 0.75 em 3 janelas → detecta `sudden_transition`
- Teste unitário com sequências sintéticas para cada padrão

---

### Fase R3 — Auto-Recalibração Reflexiva (Semana 5)

**Objetivo:** Fechar o loop de reflexão — quando drift ou anomalias são detectados, o sistema se auto-corrige recalibrando sem intervenção humana.

**Classe no `reflection.py`:**

```python
@dataclass
class RecalibrationDecision:
    should_recalibrate: bool
    reason: str
    urgency: str  # "immediate" | "scheduled" | "none"


class AutoRecalibrationManager:
    """
    R3: Fecha o loop Reflection — decide quando auto-recalibrar.
    
    Combina sinais do DriftReflector e PredictionReflector para decidir
    se a recalibração é necessária.
    
    Regras:
    1. Drift CRITICAL → recalibração imediata
    2. Drift DRIFTED + stuck_danger → recalibração imediata
    3. Drift DRIFTED sozinho → recalibração agendada (próximo checkpoint)
    4. stuck_danger sem drift → sugerir (pode ser fadiga real)
    5. Cooldown: no mínimo 5 min entre recalibrações
    """
    
    def __init__(
        self,
        min_recal_interval_sec: float = 300.0,  # 5 min entre recalibrações
        max_recalibrations_per_hour: int = 4,
    ) -> None:
        self._min_interval = min_recal_interval_sec
        self._max_per_hour = max_recalibrations_per_hour
        self._recal_timestamps: List[float] = []
        self._last_recal_time: float = 0.0
    
    def evaluate(
        self,
        drift_report: Optional[DriftReport],
        pred_reflection: Optional[PredictionReflection],
    ) -> RecalibrationDecision:
        """Decide se deve recalibrar com base nos sinais combinados."""
        import time as _time
        now = _time.monotonic()
        
        # Cooldown check
        if now - self._last_recal_time < self._min_interval:
            return RecalibrationDecision(
                should_recalibrate=False,
                reason="Dentro do cooldown mínimo entre recalibrações",
                urgency="none",
            )
        
        # Rate limit check
        recent = [
            t for t in self._recal_timestamps
            if now - t < 3600
        ]
        if len(recent) >= self._max_per_hour:
            return RecalibrationDecision(
                should_recalibrate=False,
                reason=(
                    f"Limite de {self._max_per_hour} "
                    f"recalibrações/hora atingido"
                ),
                urgency="none",
            )
        
        # Avaliar sinais combinados
        drift_status = (
            drift_report.status if drift_report else DriftStatus.STABLE
        )
        pred_pattern = (
            pred_reflection.pattern if pred_reflection else "stable"
        )
        
        # Regra 1: drift critical → imediato
        if drift_status == DriftStatus.CRITICAL:
            return self._approve_recal(
                now,
                "Drift CRITICAL detectado — features completamente "
                "fora da distribuição de treino",
                "immediate",
            )
        
        # Regra 2: drift + stuck → imediato
        if (
            drift_status == DriftStatus.DRIFTED
            and pred_pattern == "stuck_danger"
        ):
            return self._approve_recal(
                now,
                "Drift DRIFTED + Danger contínuo — provável mudança "
                "de condições (não fadiga real)",
                "immediate",
            )
        
        # Regra 3: drift sozinho → agendado
        if drift_status == DriftStatus.DRIFTED:
            return RecalibrationDecision(
                should_recalibrate=True,
                reason="Drift DRIFTED detectado — agendar recalibração",
                urgency="scheduled",
            )
        
        # Regra 4: stuck_danger sem drift → apenas sugestão
        if pred_pattern == "stuck_danger":
            return RecalibrationDecision(
                should_recalibrate=False,
                reason=(
                    "Danger contínuo SEM drift — pode ser fadiga real. "
                    "Não recalibrar automaticamente."
                ),
                urgency="none",
            )
        
        return RecalibrationDecision(
            should_recalibrate=False,
            reason="Nenhum sinal de necessidade de recalibração",
            urgency="none",
        )
    
    def _approve_recal(
        self, now: float, reason: str, urgency: str
    ) -> RecalibrationDecision:
        self._last_recal_time = now
        self._recal_timestamps.append(now)
        return RecalibrationDecision(
            should_recalibrate=True,
            reason=reason,
            urgency=urgency,
        )
```

**Integração no loop principal:**

```python
# Na inicialização:
recal_manager = AutoRecalibrationManager()

# No bloco de análise (a cada N janelas):
if drift_report is not None:
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
            # Reset pipeline para recalibração
            calibrator = RTSubjectCalibrator(
                CalibrationConfig(fps=fps, search_sec=60.0)
            )
            window_factory = OnlineWindowFactory(RTWindowConfig(fps=fps))
            drift_reflector = DriftReflector(training_stats)
            pred_reflector = PredictionReflector()
            behavior_guard.on_calibration_complete()
```

**Critério de aceite R3:**

- Drift CRITICAL → recalibração dispara em ≤10s
- Drift DRIFTED + stuck_danger → recalibração dispara
- Stuck_danger sem drift → NÃO recalibra (pode ser fadiga real!)
- Máximo 4 recalibrações por hora (safety cap contra loops infinitos)
- Cooldown de 5 min respeitado entre recalibrações

---

## Cronograma Consolidado

```
Semana 1  ┃ G1: FatigueOutput validado          ┃ G2: Feature range checks
          ┃ guardrails.py criado                 ┃ validate_and_wrap() integrado
          ┃                                      ┃
Semana 2  ┃ G3: BehaviorGuardRails               ┃ G4: CalibrationVerdict 
          ┃ Rate limiter + Watchdog + Grace       ┃ validate_calibration()
          ┃                                      ┃
Semana 3  ┃ R1: DriftReflector                   ┃ Integração com loop principal
          ┃ reflection.py criado                 ┃ Testes com dados sintéticos
          ┃                                      ┃
Semana 4  ┃ R2: PredictionReflector              ┃ Integração + testes end-to-end
          ┃ Detecção de stuck/oscillation         ┃ Testes com vídeos reais
          ┃                                      ┃
Semana 5  ┃ R3: AutoRecalibrationManager         ┃ Teste integrado completo
          ┃ Fecha o loop Reflection               ┃ Validação em RPi5 + webcam
```

---

## Estrutura Final de Arquivos

```
SALTE_INFERENCE/
├── __init__.py                    # (atualizar exports)
├── feature_extractor_rt.py        # (sem alterações)
├── subject_calibrator_rt.py       # (sem alterações)
├── window_factory_rt.py           # (sem alterações)
├── model_loader.py                # (sem alterações)
├── guardrails.py                  # ★ NOVO — G1/G2/G3/G4
├── reflection.py                  # ★ NOVO — R1/R2/R3
├── run_realtime_demo.py           # (modificado — integrar guardrails + reflection)
├── offline_eval.py                # (sem alterações)
└── tests/                         # ★ NOVO — testes unitários
    ├── test_guardrails.py
    └── test_reflection.py
```

---

## Métricas de Sucesso

| Métrica | Antes | Depois | Como medir |
|---------|-------|--------|------------|
| Falsos alarmes pós-calibração | ~2-3 nas primeiras janelas | 0 (grace period) | Contar Dangers nos primeiros 30s |
| Tempo para detectar drift | Nunca detectava | < 5 min | Simular mudança de iluminação |
| Calibrações ruins aceitas | 100% (sem filtro) | 0% (rejeitadas) | Injetar baselines fisiologicamente impossíveis |
| Alertas sonoros excessivos | Sem limite | Max 1/min | Contar beeps em sessão de 10 min |
| Crash silencioso detectado | Nunca | < 30s via watchdog | Simular freeze do loop |
| Auto-recalibração por drift | Impossível | Funcional | Mover câmera durante sessão |
