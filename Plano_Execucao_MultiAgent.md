# Plano de Execução: Multi-Agent Collaboration para DeteccaoFadiga

## Implementação do Padrão Agêntico — Fase 3

**Projeto:** DeteccaoFadiga  
**Repo:** github.com/euBrunoMelo/DeteccaoFadiga  
**Pré-requisitos:** Guardrails (G1-G4) + Reflection (R1-R3) + Memory (M1-M3) + Parallelization (P1-P3) — 83 testes  
**Estimativa total:** ~5 sprints de 1 semana  
**Dependências novas:** nenhuma (usa apenas numpy, dataclasses, typing)

---

## Contexto e Motivação

Hoje o DeteccaoFadiga opera com um **único modelo MLP** que recebe 19 features e emite uma probabilidade de fadiga. Essa arquitetura monolítica tem limitações:

1. **Opacidade:** quando o modelo diz "Danger 0.78", não há como saber se o sinal vem de piscadas lentas, PERCLOS alto, ou head-nod — tudo é uma caixa preta.
2. **Fragilidade:** se o EAR está ruidoso (óculos escuros, iluminação ruim), todo o vetor de 19 features é contaminado e a predição inteira sofre.
3. **Threshold único:** o limiar 0.41 é o mesmo para sinais oculares e posturais, embora tenham dinâmicas completamente diferentes.
4. **Sem resposta graduada baseada em tipo de fadiga:** o sistema não diferencia entre "olhos fechando" (microsleep iminente) e "head drooping" (fadiga postural progressiva).

A arquitetura Multi-Agent resolve isso dividindo a análise entre **agentes especialistas** com domínios independentes, coordenados por um **Supervisor** que agrega suas opiniões e toma a decisão final.

---

## Arquitetura: Modelo Supervisor com Agentes Especialistas

```
                        ┌─────────────────────┐
                        │     Supervisor       │
                        │  (SupervisorAgent)   │
                        │                      │
                        │  Agrega opiniões     │
                        │  Voting ponderado    │
                        │  Decisão final       │
                        └──────────┬───────────┘
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                  ▼
       ┌──────────────┐  ┌──────────────┐  ┌───────────────┐
       │ OcularAgent  │  │ BlinkAgent   │  │ PosturalAgent │
       │              │  │              │  │               │
       │ EAR stats    │  │ Blink count  │  │ Pitch/Yaw/    │
       │ EAR velocity │  │ Blink rate   │  │ Roll stats    │
       │ PERCLOS      │  │ Blink dur.   │  │ MAR           │
       │              │  │ Closing vel  │  │               │
       │ 7 features   │  │ Opening vel  │  │ 5 features    │
       │              │  │ Long blink%  │  │               │
       │              │  │ Regularity   │  │               │
       │              │  │ 7 features   │  │               │
       └──────────────┘  └──────────────┘  └───────────────┘
```

Cada agente recebe o subconjunto de features relevante ao seu domínio, aplica regras específicas, e emite uma **opinião estruturada** (`AgentOpinion`). O Supervisor agrega as opiniões via voting ponderado e produz a decisão final.

**Por que este modelo e não Network/Hierárquico?** O modelo Supervisor é ideal aqui porque:
- Os agentes não precisam se comunicar entre si (análises independentes)
- Existe uma decisão final única (Safe/Danger) que requer agregação centralizada
- O Supervisor pode resolver conflitos (ex: OcularAgent diz Safe mas BlinkAgent diz Danger)
- É o modelo com menor overhead de comunicação — crítico para tempo real no RPi5

---

## Fase A1 — Protocolo de Comunicação: AgentOpinion (Semana 1)

**Objetivo:** Definir a interface de comunicação padronizada entre agentes e o Supervisor. Todo agente emite um `AgentOpinion` que o Supervisor consome.

**Arquivo novo:** `SALTE_INFERENCE/agents.py`

```python
"""
Multi-Agent Collaboration para DeteccaoFadiga.

Arquitetura Supervisor com 3 agentes especialistas:
  A1. AgentOpinion / AgentProtocol  — protocolo de comunicação
  A2. OcularAgent     — EAR + PERCLOS (microsleep iminente)
  A3. BlinkAgent      — padrões de piscada (degradação progressiva)
  A4. PosturalAgent   — head pose + MAR (fadiga postural)
  A5. SupervisorAgent — agrega opiniões, decisão final

Ref: Agentic Design Patterns Ch.7 — Multi-Agent Collaboration, Supervisor model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np


# ── A1: Protocolo de Comunicação ─────────────────────────────────────────


class FatigueSignal(Enum):
    """Nível de fadiga reportado por um agente especialista."""
    CLEAR = 0       # Sem sinal de fadiga neste domínio
    MILD = 1        # Sinal leve — monitorar
    MODERATE = 2    # Sinal moderado — atenção
    SEVERE = 3      # Sinal forte — intervenção recomendada
    CRITICAL = 4    # Sinal extremo — ação imediata


@dataclass
class AgentOpinion:
    """
    Opinião estruturada emitida por um agente especialista.
    
    É o contrato de comunicação entre agentes e Supervisor.
    Todo agente DEVE preencher todos os campos.
    """
    agent_name: str              # "OcularAgent" | "BlinkAgent" | "PosturalAgent"
    signal: FatigueSignal        # Nível de fadiga detectado
    confidence: float            # [0.0, 1.0] — quão confiante o agente está
    score: float                 # [0.0, 1.0] — score contínuo de fadiga
    reasoning: str               # Explicação textual curta do diagnóstico
    key_indicators: Dict[str, float]  # Features mais relevantes para a decisão
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence={self.confidence} fora de [0,1]")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score={self.score} fora de [0,1]")


class SpecialistAgent(Protocol):
    """Interface que todo agente especialista deve implementar."""
    
    @property
    def name(self) -> str: ...
    
    def analyze(self, window_feats: Dict[str, float]) -> AgentOpinion: ...
```

**Critério de aceite A1:**
- `AgentOpinion.__post_init__` valida confidence e score em [0,1]
- `FatigueSignal` tem 5 níveis ordenados (CLEAR < MILD < MODERATE < SEVERE < CRITICAL)
- `SpecialistAgent` protocol tem `name` e `analyze()`

---

## Fase A2 — OcularAgent: Especialista em EAR + PERCLOS (Semana 1)

**Objetivo:** Agente focado em sinais oculares de fechamento — o indicador mais imediato de microsleep iminente.

**Features monitoradas (7):** `ear_mean`, `ear_std`, `ear_min`, `ear_vel_mean`, `ear_vel_std`, `perclos_p80_mean`, `perclos_p80_max`

```python
class OcularAgent:
    """
    A2: Especialista em sinais oculares (EAR + PERCLOS).
    
    Detecta:
    - Olhos progressivamente fechando (ear_mean baixo)
    - Variabilidade reduzida (ear_std baixo — olhar fixo/vidroso)
    - PERCLOS alto (proporção de tempo com olhos fechados)
    - Microsleep patterns (ear_min extremo + PERCLOS spike)
    
    Domínio: indicador mais IMEDIATO de fadiga — microsleep iminente.
    Tempo de reação: segundos.
    """
    
    # Thresholds calibrados contra training_stats do inference_config.json
    # ear_mean Z-normed: <-1.0 = olhos mais fechados que baseline
    # perclos: >0.15 = padrão FHWA de alerta
    
    def __init__(self) -> None:
        self._name = "OcularAgent"
    
    @property
    def name(self) -> str:
        return self._name
    
    def analyze(self, window_feats: Dict[str, float]) -> AgentOpinion:
        ear_mean = window_feats.get("ear_mean", 0.0)
        ear_std = window_feats.get("ear_std", 0.0)
        ear_min = window_feats.get("ear_min", 0.0)
        ear_vel_mean = window_feats.get("ear_vel_mean", 0.0)
        ear_vel_std = window_feats.get("ear_vel_std", 0.0)
        perclos_mean = window_feats.get("perclos_p80_mean", 0.0)
        perclos_max = window_feats.get("perclos_p80_max", 0.0)
        
        # Score composto: combinação ponderada dos indicadores
        # ear_mean negativo = olhos mais fechados que baseline
        ear_score = np.clip(-ear_mean / 3.0, 0.0, 1.0)  # -3→1.0, 0→0.0
        perclos_score = np.clip(perclos_mean / 0.4, 0.0, 1.0)  # 0.4→1.0
        perclos_peak = np.clip(perclos_max / 0.5, 0.0, 1.0)
        ear_min_score = np.clip(-ear_min / 4.0, 0.0, 1.0)  # -4→1.0
        
        # Peso: PERCLOS e ear_min são os mais diagnósticos
        score = float(
            0.25 * ear_score
            + 0.30 * perclos_score
            + 0.25 * perclos_peak
            + 0.20 * ear_min_score
        )
        score = float(np.clip(score, 0.0, 1.0))
        
        # Determinar sinal
        if score >= 0.75:
            signal = FatigueSignal.CRITICAL
        elif score >= 0.55:
            signal = FatigueSignal.SEVERE
        elif score >= 0.35:
            signal = FatigueSignal.MODERATE
        elif score >= 0.15:
            signal = FatigueSignal.MILD
        else:
            signal = FatigueSignal.CLEAR
        
        # Confiança: alta se EAR tem variação normal (dados bons)
        # Baixa se ear_std é muito baixo (pode ser artefato)
        confidence = 0.9
        if ear_std < 0.1:
            confidence = 0.6  # pouca variação — dados questionáveis
        if perclos_mean > 0.0 and ear_mean < -1.0:
            confidence = 0.95  # sinais convergentes — alta confiança
        
        # Reasoning
        reasons = []
        if ear_mean < -1.0:
            reasons.append(f"EAR abaixo do baseline (z={ear_mean:.2f})")
        if perclos_mean > 0.15:
            reasons.append(f"PERCLOS elevado ({perclos_mean:.2f})")
        if perclos_max > 0.3:
            reasons.append(f"Pico PERCLOS {perclos_max:.2f}")
        if ear_min < -3.0:
            reasons.append(f"EAR mínimo extremo (z={ear_min:.2f})")
        reasoning = "; ".join(reasons) if reasons else "Sinais oculares normais"
        
        return AgentOpinion(
            agent_name=self._name,
            signal=signal,
            confidence=confidence,
            score=score,
            reasoning=reasoning,
            key_indicators={
                "ear_mean": ear_mean,
                "perclos_p80_mean": perclos_mean,
                "perclos_p80_max": perclos_max,
                "ear_min": ear_min,
            },
        )
```

**Critério de aceite A2:**
- `ear_mean=-3.0, perclos=0.5` → `CRITICAL` com `score > 0.75`
- `ear_mean=0.0, perclos=0.05` → `CLEAR` com `score < 0.15`
- `ear_std < 0.1` → `confidence` cai para 0.6
- `reasoning` lista os indicadores específicos que dispararam

---

## Fase A3 — BlinkAgent: Especialista em Padrões de Piscada (Semana 2)

**Objetivo:** Agente focado na degradação progressiva dos padrões de piscada — indicador de fadiga crônica que precede o microsleep.

**Features monitoradas (7):** `blink_count`, `blink_rate_per_min`, `blink_mean_dur_ms`, `blink_closing_vel_mean`, `blink_opening_vel_mean`, `long_blink_pct`, `blink_regularity`

```python
class BlinkAgent:
    """
    A3: Especialista em padrões de piscada.
    
    Detecta:
    - Blinks longos (>300ms) indicando fadiga progressiva
    - Velocidade de fechamento reduzida (músculos lentos)
    - Taxa de blinks anormal (muito alta ou muito baixa)
    - Irregularidade de piscadas (perda de ritmo)
    
    Domínio: indicador de fadiga PROGRESSIVA — degradação ao longo
    de minutos/horas. Complementa o OcularAgent que é imediato.
    Tempo de reação: minutos.
    """
    
    # Referência fisiológica:
    # - Blink rate normal: 15-20/min (< 10 ou > 30 = anormal)
    # - Blink duration normal: 100-400ms (> 500ms = slow blink)
    # - Closing velocity normal: 1-3 EAR/s (< 0.5 = lento)
    # - Long blink pct: < 20% normal (> 40% = fadiga)
    
    def __init__(self) -> None:
        self._name = "BlinkAgent"
    
    @property
    def name(self) -> str:
        return self._name
    
    def analyze(self, window_feats: Dict[str, float]) -> AgentOpinion:
        blink_count = window_feats.get("blink_count", 0.0)
        blink_rate = window_feats.get("blink_rate_per_min", 0.0)
        blink_dur_ms = window_feats.get("blink_mean_dur_ms", 0.0)
        closing_vel = window_feats.get("blink_closing_vel_mean", 0.0)
        opening_vel = window_feats.get("blink_opening_vel_mean", 0.0)
        long_blink_pct = window_feats.get("long_blink_pct", 0.0)
        blink_reg = window_feats.get("blink_regularity", 0.0)
        
        # Caso especial: sem piscadas detectadas
        if blink_count == 0:
            return AgentOpinion(
                agent_name=self._name,
                signal=FatigueSignal.MILD,
                confidence=0.4,
                score=0.2,
                reasoning="Nenhuma piscada detectada — dados insuficientes ou olhar fixo",
                key_indicators={"blink_count": 0.0},
            )
        
        # Sub-scores individuais
        # Long blink percentage: > 40% é sinal forte
        long_blink_score = np.clip(long_blink_pct / 0.6, 0.0, 1.0)
        
        # Blink duration: > 500ms é slow blink
        dur_score = np.clip((blink_dur_ms - 200) / 800, 0.0, 1.0)
        
        # Closing velocity (Z-normed): negativo = mais lento que treino
        closing_score = np.clip(-closing_vel / 2.0, 0.0, 1.0)
        
        # Blink rate: muito alta (> 25/min = compensação) ou muito baixa (< 8/min = supressão)
        if blink_rate > 25:
            rate_score = np.clip((blink_rate - 25) / 30, 0.0, 0.6)
        elif blink_rate < 8:
            rate_score = np.clip((8 - blink_rate) / 8, 0.0, 0.6)
        else:
            rate_score = 0.0
        
        # Irregularidade: > 0.8 = ritmo desorganizado
        irreg_score = np.clip(blink_reg / 1.5, 0.0, 1.0)
        
        score = float(
            0.30 * long_blink_score
            + 0.25 * dur_score
            + 0.20 * closing_score
            + 0.10 * rate_score
            + 0.15 * irreg_score
        )
        score = float(np.clip(score, 0.0, 1.0))
        
        # Sinal
        if score >= 0.70:
            signal = FatigueSignal.SEVERE
        elif score >= 0.45:
            signal = FatigueSignal.MODERATE
        elif score >= 0.20:
            signal = FatigueSignal.MILD
        else:
            signal = FatigueSignal.CLEAR
        
        # Confiança: proporcional ao número de blinks (mais dados = mais confiável)
        confidence = float(np.clip(blink_count / 8.0, 0.3, 0.95))
        
        # Reasoning
        reasons = []
        if long_blink_pct > 0.3:
            reasons.append(f"Long blinks {long_blink_pct:.0%}")
        if blink_dur_ms > 400:
            reasons.append(f"Blinks lentos ({blink_dur_ms:.0f}ms)")
        if closing_vel < -0.5:
            reasons.append(f"Fechamento lento (z={closing_vel:.2f})")
        if blink_reg > 0.8:
            reasons.append(f"Ritmo irregular ({blink_reg:.2f})")
        reasoning = "; ".join(reasons) if reasons else "Padrão de piscadas normal"
        
        return AgentOpinion(
            agent_name=self._name,
            signal=signal,
            confidence=confidence,
            score=score,
            reasoning=reasoning,
            key_indicators={
                "long_blink_pct": long_blink_pct,
                "blink_mean_dur_ms": blink_dur_ms,
                "blink_closing_vel_mean": closing_vel,
                "blink_regularity": blink_reg,
            },
        )
```

**Critério de aceite A3:**
- `long_blink_pct=0.7, blink_dur_ms=800` → `SEVERE`
- `blink_count=0` → `MILD` com `confidence=0.4` (dados insuficientes)
- `blink_count=10, long_blink_pct=0.1, dur=200` → `CLEAR`
- Confidence escala com `blink_count` (mais blinks = mais confiável)

---

## Fase A4 — PosturalAgent: Especialista em Head Pose + MAR (Semana 2)

**Objetivo:** Agente focado em sinais posturais — head-nods, desvios de posição, e bocejo.

**Features monitoradas (5):** `pitch_mean`, `pitch_std`, `yaw_std`, `roll_std`, `mar_mean`

```python
class PosturalAgent:
    """
    A4: Especialista em sinais posturais (head pose + MAR).
    
    Detecta:
    - Head-nod (pitch negativo = cabeça caindo para frente)
    - Variabilidade postural alta (pitch_std, yaw_std = instabilidade)
    - Bocejo (mar_mean elevado)
    - Desvio lateral (roll_std alto)
    
    Domínio: indicador de fadiga POSTURAL — complementa sinais oculares.
    Nota: em produção, HeadPoseNeutralizer (C33) zera pitch/yaw/roll.
    Este agente é mais relevante em modo lab (--no-neutralize-pose).
    Em modo produção, baseia-se primariamente no MAR (bocejo).
    Tempo de reação: minutos.
    """
    
    def __init__(self, pose_neutralized: bool = True) -> None:
        self._name = "PosturalAgent"
        self._pose_neutralized = pose_neutralized
    
    @property
    def name(self) -> str:
        return self._name
    
    def analyze(self, window_feats: Dict[str, float]) -> AgentOpinion:
        pitch_mean = window_feats.get("pitch_mean", 0.0)
        pitch_std = window_feats.get("pitch_std", 0.0)
        yaw_std = window_feats.get("yaw_std", 0.0)
        roll_std = window_feats.get("roll_std", 0.0)
        mar_mean = window_feats.get("mar_mean", 0.0)
        
        # Se pose está neutralizada, os valores de pitch/yaw/roll são 0
        # após o scaler. O agente se baseia apenas no MAR.
        if self._pose_neutralized:
            return self._analyze_mar_only(mar_mean)
        
        return self._analyze_full(
            pitch_mean, pitch_std, yaw_std, roll_std, mar_mean
        )
    
    def _analyze_mar_only(self, mar_mean: float) -> AgentOpinion:
        """Análise quando pose está neutralizada (produção)."""
        # MAR Z-normed: > 1.0 = boca mais aberta que treino (bocejo?)
        mar_score = float(np.clip(mar_mean / 3.0, 0.0, 1.0))
        
        if mar_score >= 0.5:
            signal = FatigueSignal.MODERATE
        elif mar_score >= 0.2:
            signal = FatigueSignal.MILD
        else:
            signal = FatigueSignal.CLEAR
        
        reasons = []
        if mar_mean > 1.0:
            reasons.append(f"MAR elevado (z={mar_mean:.2f}) — possível bocejo")
        
        return AgentOpinion(
            agent_name=self._name,
            signal=signal,
            confidence=0.5,  # baixa: apenas MAR disponível
            score=mar_score,
            reasoning="; ".join(reasons) if reasons else "MAR normal, pose neutralizada",
            key_indicators={"mar_mean": mar_mean},
        )
    
    def _analyze_full(
        self,
        pitch_mean: float, pitch_std: float,
        yaw_std: float, roll_std: float,
        mar_mean: float,
    ) -> AgentOpinion:
        """Análise completa com pose (modo lab)."""
        # Pitch negativo = cabeça caindo (head-nod)
        nod_score = float(np.clip(-pitch_mean / 3.0, 0.0, 1.0))
        
        # Instabilidade postural
        instab_score = float(np.clip(
            (pitch_std + yaw_std + roll_std) / 6.0, 0.0, 1.0
        ))
        
        # MAR (bocejo)
        mar_score = float(np.clip(mar_mean / 3.0, 0.0, 1.0))
        
        score = float(
            0.40 * nod_score
            + 0.30 * instab_score
            + 0.30 * mar_score
        )
        score = float(np.clip(score, 0.0, 1.0))
        
        if score >= 0.65:
            signal = FatigueSignal.SEVERE
        elif score >= 0.40:
            signal = FatigueSignal.MODERATE
        elif score >= 0.18:
            signal = FatigueSignal.MILD
        else:
            signal = FatigueSignal.CLEAR
        
        reasons = []
        if pitch_mean < -1.0:
            reasons.append(f"Head-nod (pitch z={pitch_mean:.2f})")
        if pitch_std > 1.5:
            reasons.append(f"Instabilidade postural (pitch_std={pitch_std:.2f})")
        if mar_mean > 1.0:
            reasons.append(f"Bocejo (MAR z={mar_mean:.2f})")
        
        confidence = 0.85
        if abs(pitch_mean) < 0.3 and pitch_std < 0.5:
            confidence = 0.7  # pouco movimento — dados limitados
        
        return AgentOpinion(
            agent_name=self._name,
            signal=signal,
            confidence=confidence,
            score=score,
            reasoning="; ".join(reasons) if reasons else "Postura estável",
            key_indicators={
                "pitch_mean": pitch_mean,
                "pitch_std": pitch_std,
                "mar_mean": mar_mean,
            },
        )
```

**Critério de aceite A4:**
- Com `pose_neutralized=True`: apenas MAR influencia, confidence=0.5
- Com `pose_neutralized=False`: pitch_mean=-2.5 → `SEVERE` (head-nod)
- `mar_mean=2.0` → `MODERATE` (bocejo), tanto com pose neutralizada ou não
- Com `pitch_mean=0, pitch_std=0, mar_mean=0` → `CLEAR`

---

## Fase A5 — SupervisorAgent: Agregador de Decisão (Semana 3)

**Objetivo:** Agente supervisor que recebe as opiniões dos 3 especialistas, agrega via voting ponderado, e produz a decisão final. Também detecta convergência e divergência entre agentes.

```python
@dataclass
class SupervisorDecision:
    """Decisão final do Supervisor."""
    label: str                         # "Safe" | "Danger"
    combined_score: float              # [0.0, 1.0] — score agregado
    alert_level_suggestion: str        # "SAFE" | "WATCH" | "DANGER" | "CRITICAL"
    dominant_agent: str                # agente com maior contribuição
    agent_agreement: float             # [0.0, 1.0] — grau de concordância
    opinions: List[AgentOpinion]       # opiniões originais
    reasoning: str                     # explicação agregada
    fatigue_type: str                  # "ocular" | "behavioral" | "postural" | "mixed" | "none"


@dataclass
class SupervisorConfig:
    """Pesos dos agentes e thresholds do Supervisor."""
    # Pesos base dos agentes (somam 1.0)
    ocular_weight: float = 0.45       # Mais peso: indicador mais imediato
    blink_weight: float = 0.35        # Segundo: degradação progressiva
    postural_weight: float = 0.20     # Menor: neutralizado em produção
    
    # Thresholds de decisão
    danger_threshold: float = 0.45    # score >= threshold → Danger
    watch_threshold: float = 0.25     # score >= threshold → Watch
    critical_threshold: float = 0.70  # score >= threshold → Critical
    
    # Boost de convergência: se 2+ agentes concordam, amplificar
    convergence_boost: float = 0.10   # +10% ao score se convergência


class SupervisorAgent:
    """
    A5: Supervisor — agrega opiniões dos especialistas.
    
    Estratégia de agregação:
    1. Weighted average dos scores, ponderado por (weight × confidence)
    2. Convergence boost: se 2+ agentes reportam MODERATE+, soma +10%
    3. Dominant agent: identifica qual agente contribuiu mais
    4. Fatigue type: classifica o tipo dominante de fadiga
    
    O Supervisor NÃO substitui o modelo MLP — opera em paralelo.
    O resultado final combina a predição do MLP com a análise multi-agente.
    """
    
    def __init__(
        self,
        ocular: OcularAgent,
        blink: BlinkAgent,
        postural: PosturalAgent,
        config: Optional[SupervisorConfig] = None,
    ) -> None:
        self._ocular = ocular
        self._blink = blink
        self._postural = postural
        self.cfg = config or SupervisorConfig()
    
    def decide(
        self, window_feats: Dict[str, float]
    ) -> SupervisorDecision:
        """
        Consulta os 3 agentes e agrega suas opiniões.
        
        Pipeline:
        1. Cada agente analisa window_feats independentemente
        2. Weighted average dos scores × confidences
        3. Convergence boost se houver concordância
        4. Classificação final
        """
        # 1. Coletar opiniões
        op_ocular = self._ocular.analyze(window_feats)
        op_blink = self._blink.analyze(window_feats)
        op_postural = self._postural.analyze(window_feats)
        opinions = [op_ocular, op_blink, op_postural]
        
        # 2. Weighted average (score × confidence × weight)
        weights = {
            op_ocular.agent_name: self.cfg.ocular_weight,
            op_blink.agent_name: self.cfg.blink_weight,
            op_postural.agent_name: self.cfg.postural_weight,
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        contributions = {}
        
        for op in opinions:
            w = weights[op.agent_name]
            effective_w = w * op.confidence
            contribution = op.score * effective_w
            weighted_sum += contribution
            total_weight += effective_w
            contributions[op.agent_name] = contribution
        
        combined_score = weighted_sum / max(total_weight, 1e-8)
        
        # 3. Convergence boost
        agents_above_moderate = sum(
            1 for op in opinions
            if op.signal.value >= FatigueSignal.MODERATE.value
        )
        if agents_above_moderate >= 2:
            combined_score = min(
                combined_score + self.cfg.convergence_boost, 1.0
            )
        
        combined_score = float(np.clip(combined_score, 0.0, 1.0))
        
        # 4. Dominant agent
        dominant = max(contributions, key=contributions.get)
        
        # 5. Fatigue type
        fatigue_type = self._classify_fatigue_type(
            op_ocular, op_blink, op_postural
        )
        
        # 6. Agreement (1.0 se todos iguais, 0.0 se totalmente divergentes)
        signals = [op.signal.value for op in opinions]
        signal_range = max(signals) - min(signals)
        agreement = 1.0 - min(signal_range / 4.0, 1.0)
        
        # 7. Label e alert level
        if combined_score >= self.cfg.critical_threshold:
            label = "Danger"
            alert = "CRITICAL"
        elif combined_score >= self.cfg.danger_threshold:
            label = "Danger"
            alert = "DANGER"
        elif combined_score >= self.cfg.watch_threshold:
            label = "Safe"
            alert = "WATCH"
        else:
            label = "Safe"
            alert = "SAFE"
        
        # 8. Reasoning agregado
        agent_reasons = [
            f"[{op.agent_name}:{op.signal.name}] {op.reasoning}"
            for op in opinions
            if op.signal.value >= FatigueSignal.MILD.value
        ]
        if not agent_reasons:
            reasoning = "Todos os agentes reportam sinais normais"
        else:
            reasoning = " | ".join(agent_reasons)
        
        return SupervisorDecision(
            label=label,
            combined_score=combined_score,
            alert_level_suggestion=alert,
            dominant_agent=dominant,
            agent_agreement=agreement,
            opinions=opinions,
            reasoning=reasoning,
            fatigue_type=fatigue_type,
        )
    
    @staticmethod
    def _classify_fatigue_type(
        ocular: AgentOpinion,
        blink: AgentOpinion,
        postural: AgentOpinion,
    ) -> str:
        """Classifica o tipo dominante de fadiga."""
        above = {
            "ocular": ocular.signal.value >= FatigueSignal.MODERATE.value,
            "behavioral": blink.signal.value >= FatigueSignal.MODERATE.value,
            "postural": postural.signal.value >= FatigueSignal.MODERATE.value,
        }
        active = [k for k, v in above.items() if v]
        
        if len(active) == 0:
            return "none"
        if len(active) >= 2:
            return "mixed"
        return active[0]
```

**Critério de aceite A5:**
- Todos os agentes `CLEAR` → `label="Safe"`, `fatigue_type="none"`
- Ocular `SEVERE` + Blink `MODERATE` + Postural `CLEAR` → `label="Danger"`, `fatigue_type="mixed"`
- Convergence boost: 2 agentes MODERATE → score sobe ~10%
- `agreement=1.0` quando todos no mesmo nível, `agreement=0.0` quando CLEAR vs CRITICAL
- `dominant_agent` identifica corretamente quem mais contribuiu

---

## Fase A6 — Integração com Pipeline Existente (Semana 4)

**Objetivo:** O sistema multi-agente opera **em paralelo** ao modelo MLP existente, não o substitui. Ambas as avaliações alimentam o `FatigueOutput` com informação adicional.

**Modificação no `run_realtime_demo.py`:**

```python
# Na inicialização:
from .agents import (
    OcularAgent, BlinkAgent, PosturalAgent,
    SupervisorAgent, SupervisorConfig,
)

ocular_agent = OcularAgent()
blink_agent = BlinkAgent()
postural_agent = PosturalAgent(pose_neutralized=neutralize_pose)
supervisor = SupervisorAgent(
    ocular_agent, blink_agent, postural_agent,
    SupervisorConfig(),
)
print("[init] Multi-Agent: Supervisor + 3 specialists initialized")

# Após o MLP predict_fatigue() e validate_and_wrap():
supervisor_decision = supervisor.decide(window_feats)

# Enriquecer o output com informação multi-agente
if debug:
    print(f"[agents] MLP: {output.label} ({output.prob_danger:.3f})")
    print(f"[agents] Supervisor: {supervisor_decision.label} "
          f"({supervisor_decision.combined_score:.3f})")
    print(f"[agents] Type: {supervisor_decision.fatigue_type}")
    print(f"[agents] Dominant: {supervisor_decision.dominant_agent}")
    print(f"[agents] Agreement: {supervisor_decision.agent_agreement:.2f}")
    for op in supervisor_decision.opinions:
        print(f"[agents]   {op.agent_name}: {op.signal.name} "
              f"(score={op.score:.3f}, conf={op.confidence:.2f})")
        print(f"[agents]     → {op.reasoning}")

# Log do supervisor na SessionMemory
session.on_window(
    ...
    # novos campos opcionais:
    # fatigue_type=supervisor_decision.fatigue_type,
    # agent_agreement=supervisor_decision.agent_agreement,
)

# Overlay enriquecido: mostrar tipo de fadiga
if supervisor_decision.fatigue_type != "none":
    overlay2 += f" Type:{supervisor_decision.fatigue_type}"
```

**Novo CLI flag:**

```python
parser.add_argument(
    "--no-agents", action="store_true",
    help="Desativar análise multi-agente (usar apenas MLP)",
)
```

**Critério de aceite A6:**
- Com `--debug`, log mostra MLP + Supervisor + 3 opiniões lado a lado
- Overlay mostra `Type:ocular` ou `Type:mixed` quando fadiga detectada
- Com `--no-agents`, o pipeline funciona igual ao anterior (backward-compatible)
- Overhead: < 1ms por janela (regras simples, sem ONNX adicional)

---

## Fase A7 — Testes Unitários (Semana 5)

**Arquivo:** `SALTE_INFERENCE/tests/test_agents.py`

Cobertura planejada (~30 testes):

```
TestAgentOpinionA1 (4 testes):
  - test_valid_opinion
  - test_invalid_confidence_raises
  - test_invalid_score_raises
  - test_fatigue_signal_ordering

TestOcularAgentA2 (6 testes):
  - test_clear_normal_eyes
  - test_critical_closed_eyes_high_perclos
  - test_moderate_perclos_only
  - test_low_confidence_low_ear_std
  - test_reasoning_lists_indicators
  - test_score_bounds

TestBlinkAgentA3 (6 testes):
  - test_clear_normal_blinks
  - test_severe_long_slow_blinks
  - test_no_blinks_insufficient_data
  - test_confidence_scales_with_count
  - test_high_rate_compensation
  - test_irregular_rhythm

TestPosturalAgentA4 (6 testes):
  - test_clear_stable_posture
  - test_neutralized_mar_only
  - test_neutralized_low_confidence
  - test_full_head_nod
  - test_full_yawn
  - test_full_instability

TestSupervisorAgentA5 (8 testes):
  - test_all_clear_is_safe
  - test_ocular_severe_is_danger
  - test_convergence_boost
  - test_dominant_agent_correct
  - test_agreement_all_same
  - test_agreement_divergent
  - test_fatigue_type_mixed
  - test_fatigue_type_single
```

---

## Cronograma Consolidado

```
Semana 1  ┃ A1: AgentOpinion + Protocol           ┃ A2: OcularAgent
          ┃ Contrato de comunicação                ┃ EAR + PERCLOS specialist
          ┃                                        ┃
Semana 2  ┃ A3: BlinkAgent                         ┃ A4: PosturalAgent
          ┃ Padrões de piscada specialist          ┃ Head pose + MAR specialist
          ┃                                        ┃
Semana 3  ┃ A5: SupervisorAgent                    ┃ Weighted voting + convergence
          ┃ Agregação + decisão final              ┃ Fatigue type classification
          ┃                                        ┃
Semana 4  ┃ A6: Integração no pipeline             ┃ Overlay + debug + CLI flags
          ┃ MLP + Multi-Agent em paralelo          ┃ Backward-compatible
          ┃                                        ┃
Semana 5  ┃ A7: Testes unitários (~30)             ┃ Validação end-to-end
          ┃ Cobertura de todos os agents           ┃ RPi5 + PC
```

---

## Estrutura Final de Arquivos

```
SALTE_INFERENCE/
├── __init__.py                    (atualizar)
├── feature_extractor_rt.py        (sem alterações)
├── subject_calibrator_rt.py       (sem alterações)
├── window_factory_rt.py           (sem alterações)
├── model_loader.py                (sem alterações)
├── guardrails.py                  (Fase 1)
├── reflection.py                  (Fase 1)
├── memory.py                      (Fase 2)
├── parallel.py                    (Fase 2)
├── agents.py                      ★ NOVO — A1/A2/A3/A4/A5
├── run_realtime_demo.py           (modificado — integrar multi-agent)
├── offline_eval.py                (sem alterações)
└── tests/
    ├── test_guardrails.py         (24 testes)
    ├── test_reflection.py         (18 testes)
    ├── test_memory.py             (26 testes)
    ├── test_parallel.py           (15 testes)
    └── test_agents.py             ★ NOVO — ~30 testes
```

---

## Decisões Arquiteturais e Justificativas

**Por que agentes baseados em regras e não em LLM?** O sistema roda em tempo real no Raspberry Pi 5 a 30fps. Chamadas a LLM adicionariam latência de 200ms-2s por janela — inaceitável. Os agentes usam regras calibradas contra os `training_stats` do modelo, oferecendo interpretabilidade com overhead de < 1ms.

**Por que o multi-agente não substitui o MLP?** O MLP é o modelo treinado com validação cruzada (71.5% balanced accuracy) — é a ground truth estatística. O multi-agente adiciona **interpretabilidade** e **resposta graduada por tipo de fadiga**, mas a decisão binária Safe/Danger continua vindo do MLP. O supervisor enriquece a decisão com diagnóstico, não a substitui.

**Por que pesos fixos e não aprendidos?** Na Fase 3, os pesos do Supervisor são heurísticos (0.45/0.35/0.20). Em uma Fase futura, os pesos poderiam ser aprendidos via regressão logística sobre os features logs (M3), otimizando os pesos para maximizar concordância com o MLP. O `FeatureLogger` já grava os dados necessários para isso.

**Por que `confidence` por agente?** Sem confidence, o Supervisor trata todas as opiniões igualmente — mesmo quando um agente tem dados ruins (ex: 0 blinks, ou pose neutralizada). Com confidence ponderada, o Supervisor dá menos peso a opiniões incertas, melhorando a robustez.

**Por que `fatigue_type`?** É o diferencial principal do multi-agente. Saber que a fadiga é "ocular" (microsleep iminente) vs "behavioral" (degradação progressiva) permite respostas diferentes: alerta sonoro imediato para ocular, sugestão de pausa para behavioral.

---

## Métricas de Sucesso

| Métrica | Antes | Depois | Como medir |
|---------|-------|--------|------------|
| Interpretabilidade | "Danger 0.78" (caixa preta) | "Danger: ocular (PERCLOS 0.45) + blinks lentos" | Verificar reasoning no log |
| Tipo de fadiga | Inexistente | ocular / behavioral / postural / mixed / none | `fatigue_type` no SupervisorDecision |
| Concordância MLP↔Supervisor | N/A | > 80% | Comparar labels em 100 janelas |
| Overhead por janela | 0ms | < 1ms | `PerformanceMonitor` |
| Conflito detectável | Impossível | agreement < 0.5 → conflito | `agent_agreement` no log |
| Backward-compatible | N/A | `--no-agents` funciona igual | Teste de regressão |
