# Padrões Agênticos Aplicáveis ao DeteccaoFadiga2

## Análise de Oportunidades de Evolução com Design Patterns de IA

**Projeto:** DeteccaoFadiga2 — Sistema de detecção de fadiga em tempo real via visão computacional  
**Repositório:** github.com/euBrunoMelo/DeteccaoFadiga2  
**Data da Análise:** Março 2026

---

## Contexto do Projeto

O DeteccaoFadiga2 é um sistema de detecção de fadiga em tempo real que utiliza visão computacional e uma rede MLP para classificar o estado de motoristas/operadores como **Safe** ou **Danger**. O pipeline atual opera de forma determinística e sequencial:

```
Camera → ONNXFaceMesh → FeatureExtractor → Calibração Z-Norm → Janela 15s → MLP → Safe/Danger
```

O sistema já possui qualidades arquiteturais sólidas (modularidade, constraints bem definidos como C5, C6-V2, C13, C22, C32, C33, C34), mas opera de forma estática — sem adaptação, sem auto-avaliação e sem coordenação inteligente entre componentes.

Este documento analisa como cada um dos 11 padrões agênticos pode ser aplicado para transformar o DeteccaoFadiga2 de um pipeline determinístico em um **sistema agêntico** capaz de auto-correção, adaptação e coordenação inteligente.

---

## 1. Prompt Chaining (Pipeline)

### Status Atual: ✅ Já implementado

O projeto já opera como um pipeline encadeado de 5 estágios com interfaces estruturadas entre cada etapa:

| Estágio | Módulo | Entrada → Saída |
|---------|--------|-----------------|
| 1 | `feature_extractor_rt.py` | `frame_bgr` → `RTFrameFeatures` |
| 2 | `subject_calibrator_rt.py` | `RTFrameFeatures` → `CalibratedFrame` |
| 3 | `window_factory_rt.py` | `CalibratedFrame` → `Dict[str, float]` (19 features) |
| 4 | `model_loader.py` (scale + neutralize) | `Dict` → `np.ndarray` escalado |
| 5 | `model_loader.py` (predict) | `np.ndarray` → `(prob_danger, label)` |

### Oportunidade de Evolução

Transformar o pipeline estático em um **pipeline adaptativo** onde cada estágio pode sinalizar anomalias para o próximo. Exemplo: se o `FeatureExtractor` detectar que a face saiu do quadro por mais de 5 segundos, ele poderia emitir um sinal estruturado de "face_lost_prolonged" que o estágio de inferência interpretaria como alerta independente, sem esperar a janela de 15s.

```
Proposta: RTFrameFeatures ganha campo "metadata: Dict" com sinais 
inter-estágio como face_lost_duration_ms, lighting_quality, landmark_confidence.
Cada estágio downstream consome esses metadados para ajustar seu comportamento.
```

---

## 2. Routing (Roteamento Condicional)

### Status Atual: ⚠️ Parcial

O projeto tem um roteamento básico entre backends (`ONNXFaceMeshBackend` vs `DummyBackend`) e entre modos de câmera (`PiCamera2Capture` vs `cv2.VideoCapture`), mas essas decisões são estáticas (definidas na inicialização).

### Oportunidade de Evolução

Implementar **roteamento dinâmico** baseado em contexto em múltiplas dimensões:

**a) Roteamento de Modelo por Confiança:**
Se a confiança da detecção facial cair abaixo de um limiar (ex: `min_face_score < 0.65`), rotear para um modelo secundário mais robusto (ou menor resolução). Se a confiança for alta, usar o pipeline completo de 19 features.

```python
# Pseudo-código
if face_confidence > 0.8:
    route = "full_pipeline"      # MLP 19 features, alta precisão
elif face_confidence > 0.5:
    route = "reduced_pipeline"   # Modelo simplificado, apenas EAR + PERCLOS
else:
    route = "alert_no_face"      # Alerta direto — face não detectável
```

**b) Roteamento por Ambiente:**
Classificar condições de iluminação (dia/noite/contraluz) e rotear para modelos ou thresholds específicos por condição. O EAR calibrado à noite com câmera IR pode ter ranges diferentes do dia.

**c) Roteamento por Perfil de Operador:**
Após calibração, classificar o perfil do operador (olhos naturalmente pequenos vs grandes baseado no `ear_mean` do baseline) e rotear para thresholds personalizados.

---

## 3. Parallelization (Paralelização)

### Status Atual: ❌ Não implementado

Todo o pipeline roda sequencialmente em um único thread. O loop principal em `run_realtime_demo.py` processa captura → extração → calibração → janela → inferência de forma serial.

### Oportunidade de Evolução

**a) Captura e Processamento em Paralelo:**
Usar um padrão producer-consumer com fila thread-safe. A câmera captura frames em uma thread dedicada enquanto o processamento de features roda em outra. Isso elimina drops de frame quando a inferência é lenta (especialmente no Raspberry Pi).

```
Thread 1 (Captura):   Camera → Queue(maxsize=2)
Thread 2 (Pipeline):  Queue → FeatureExtractor → Calibrator → Window → Model
Thread 3 (Overlay):   Results → Display/Alerta
```

**b) Feature Extraction Paralela:**
Os cálculos de EAR, MAR e HeadPose são independentes entre si após a extração de landmarks. Podem ser computados em paralelo:

```
landmarks → ┬→ compute_ear (LEFT_EYE + RIGHT_EYE)  ──┐
             ├→ compute_mar (MOUTH)                   ├→ RTFrameFeatures
             └→ compute_head_pose (solvePnP)         ──┘
```

**c) Multi-Sensor Fusion Paralela:**
Em cenários com múltiplos sensores (câmera visível + câmera IR + sensor de pressão no volante), cada sensor seria processado em paralelo e os resultados agregados para decisão final, aumentando a robustez.

---

## 4. Reflection (Auto-Correção)

### Status Atual: ❌ Não implementado

O sistema atual faz uma inferência one-shot: gera um vetor de features → prediz → emite label. Não há mecanismo de auto-avaliação da qualidade da predição.

### Oportunidade de Evolução

Este é talvez o padrão com **maior potencial de impacto** no projeto.

**a) Agente Crítico de Qualidade de Calibração:**
Após a calibração automática (120s warm-up), um agente "Crítico" avalia a qualidade do baseline:

```
Produtor: RTSubjectCalibrator → SubjectBaseline
Crítico:  Verifica:
  - ear_std está dentro de [0.015, 0.10]? (muito baixo = pessoa não piscou)
  - pitch_std < 30°? (variação excessiva = dados ruins)
  - Proporção de frames válidos > 90%?
  - EAR mean está em range fisiológico [0.15, 0.40]?
  
Se falhar → solicitar recalibração ou ajustar thresholds automaticamente
```

**b) Detecção de Drift com Loop de Reflexão:**
A cada N janelas (ex: 10 janelas = ~2.5 minutos), um módulo de reflexão compara as distribuições recentes com o baseline de calibração. Se houver drift significativo (ex: iluminação mudou, câmera se moveu), o sistema sugere recalibração.

```python
class DriftReflector:
    """Compara distribuição running vs baseline a cada N janelas."""
    def reflect(self, recent_windows: List[Dict], baseline: SubjectBaseline) -> str:
        ear_drift = abs(np.mean([w['ear_mean'] for w in recent_windows]))
        if ear_drift > 2.0:  # mais de 2 sigma de desvio
            return "RECALIBRATE"  # drift detectado
        return "OK"
```

**c) Reflexão sobre Falsos Alarmes:**
Se o sistema emitir 5 "Danger" consecutivos seguidos de uma rápida transição para "Safe" (sem intervenção), isso pode indicar falso alarme. O reflexor registra o padrão e ajusta o threshold de forma conservadora.

---

## 5. Tool Use (Function Calling)

### Status Atual: ⚠️ Parcial (ferramentas internas)

O sistema usa "ferramentas" internas — modelos ONNX (BlazeFace, FaceMesh, MLP) — mas não tem capacidade de chamar ferramentas externas de forma dinâmica.

### Oportunidade de Evolução

**a) Integração com APIs Externas via Function Calling:**

| Ferramenta | Propósito | Trigger |
|------------|-----------|---------|
| API de Frota | Enviar alerta ao dispatcher | `label == "Danger"` por 3+ janelas |
| API de Telemetria | Cruzar fadiga com velocidade/frenagem | Cada janela |
| API Meteorológica | Ajustar thresholds em condições adversas | A cada 30 min |
| Buzzer/LED GPIO | Alerta sonoro/visual no Raspberry Pi | `label == "Danger"` |
| Logger Remoto | Enviar features para retraining futuro | Cada janela |

**b) Formato Estruturado de Tool Calling:**
O sistema poderia emitir JSONs estruturados de "ação sugerida" que uma camada de orquestração consome:

```json
{
  "action": "send_alert",
  "tool": "fleet_api",
  "params": {
    "driver_id": "op-042",
    "severity": "high",
    "prob_danger": 0.87,
    "perclos": 0.45,
    "microsleeps": 2,
    "timestamp": "2026-03-11T14:32:00Z"
  }
}
```

**c) Execução Condicional de Ferramentas:**
Ao invés de alertar sempre que `label == "Danger"`, o agente avalia se deve chamar cada ferramenta. Por exemplo: se o veículo está parado (velocidade = 0 via telemetria), o sistema suspende alertas de fadiga pois o motorista pode estar descansando legitimamente.

---

## 6. Planning (Planejamento)

### Status Atual: ❌ Não implementado

O sistema opera de forma reativa — processa cada frame conforme chega, sem planejamento antecipado.

### Oportunidade de Evolução

**a) Planejador de Sessão de Monitoramento:**
Antes de iniciar o monitoramento, um agente planejador avalia as condições e gera um plano de sessão:

```
Entrada: condições de câmera, iluminação, perfil do operador, duração prevista
Plano gerado:
  1. Warm-up: 120s (ou 60s se perfil já calibrado anteriormente)
  2. Fase inicial: threshold conservador (0.51) por 5 min
  3. Fase operacional: threshold ajustado (0.41) após confirmar estabilidade
  4. Checkpoints de recalibração a cada 30 min
  5. Escalation: se 3+ Danger em 5 min → acionar pausa obrigatória
```

**b) Plano de Resposta a Fadiga (ReAct):**
Implementar um loop Reason-Act-Observe para resposta graduada:

```
Observação: prob_danger = 0.62 (1ª janela Danger)
Raciocínio: "Primeira ocorrência. Pode ser ruído. Monitorar."
Ação: Incrementar alerta_level para 1 (visual sutil)

Observação: prob_danger = 0.78 (3ª janela Danger consecutiva)
Raciocínio: "Padrão consistente. Fadiga provável."
Ação: Alerta sonoro + notificar frota

Observação: prob_danger = 0.91 + microsleep_count > 0
Raciocínio: "Microsleep detectado. Risco iminente."
Ação: Alerta máximo + solicitar parada imediata
```

---

## 7. Multi-Agent Collaboration

### Status Atual: ❌ Não implementado

O sistema é monolítico — um único pipeline de detecção.

### Oportunidade de Evolução

Transformar o sistema em uma **arquitetura multi-agente com supervisor**:

```
                    ┌──────────────┐
                    │  Supervisor  │
                    │  (Decisor)   │
                    └──────┬───────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    ┌─────────────┐ ┌──────────┐ ┌──────────────┐
    │ Agente      │ │ Agente   │ │ Agente       │
    │ Ocular      │ │ Postural │ │ Comportamental│
    │ (EAR/PERCLOS│ │ (Head    │ │ (Blink       │
    │  Microsleep)│ │  Pose)   │ │  Patterns)   │
    └─────────────┘ └──────────┘ └──────────────┘
```

**Agente Ocular:** Especialista em EAR, PERCLOS, microsleeps. Usa threshold e modelo otimizados para sinais oculares.

**Agente Postural:** Especialista em head pose. Detecta head-nods (acenos de cabeça típicos de sonolência) e desvios posturais. Pode usar um modelo separado treinado especificamente para padrões de pose.

**Agente Comportamental:** Especialista em padrões temporais de piscadas — regularidade, velocidade de fechamento/abertura, long blinks. Detecta degradação progressiva ao longo do turno.

**Supervisor:** Agrega as opiniões dos 3 agentes com pesos (ex: voting ponderado ou meta-modelo) e toma a decisão final. Pode resolver conflitos — por exemplo, se o agente ocular diz "Safe" mas o comportamental detecta degradação progressiva, o supervisor pode elevar o nível de alerta preventivamente.

---

## 8. Memory Management

### Status Atual: ⚠️ Parcial (curto prazo apenas)

O sistema tem memória de curto prazo via buffers: `_warmup_buffer` no calibrador, `_buffer` no window factory, e ring buffers no `HeadPoseSanitizer`. Não há memória de longo prazo.

### Oportunidade de Evolução

**a) Memória de Curto Prazo Enriquecida (Estado de Sessão):**
Manter um estado de sessão que acumula informações ao longo da operação:

```python
class SessionMemory:
    total_danger_windows: int = 0
    total_safe_windows: int = 0
    consecutive_danger: int = 0
    max_consecutive_danger: int = 0
    last_recalibration: float = 0.0
    microsleep_history: List[float] = []  # timestamps de microsleeps
    perclos_trend: List[float] = []       # PERCLOS ao longo do tempo
    alert_level: int = 0                   # escalation state
    operator_profile: Dict = {}            # aprendido na calibração
```

**b) Memória de Longo Prazo (Banco Vetorial / SQLite):**
Armazenar histórico de sessões anteriores do mesmo operador para:

- Usar baselines de calibrações anteriores como "warm start" (reduzir tempo de warm-up de 120s para 30s se o perfil já é conhecido)
- Detectar degradação crônica de atenção ao longo de dias/semanas
- Construir perfis de risco individuais (ex: operador X tende a apresentar fadiga após 4h de turno)

**c) Memória para Retraining (Data Flywheel):**
Gravar features + labels + condições ambientais para uso futuro em retraining do modelo. Isso fecha o ciclo de aprendizado contínuo:

```
Operação → Features + Labels → Storage → Curadoria → Retraining → Modelo atualizado
```

---

## 9. Model Context Protocol (MCP)

### Status Atual: ❌ Não implementado

Os modelos ONNX são carregados diretamente via `onnxruntime.InferenceSession`. Não há protocolo padronizado de descoberta ou acesso a ferramentas.

### Oportunidade de Evolução

Se o sistema evoluir para uma arquitetura de frota com múltiplos veículos e um servidor central, o MCP se torna relevante:

**a) MCP Server para Ferramentas de Monitoramento:**
Cada veículo/estação expõe suas capacidades via um MCP Server local:

```json
{
  "tools": [
    {"name": "get_fatigue_status", "description": "Retorna status atual Safe/Danger com probabilidade"},
    {"name": "get_calibration_status", "description": "Retorna se o operador está calibrado"},
    {"name": "force_recalibrate", "description": "Força nova calibração do operador"},
    {"name": "get_session_stats", "description": "Retorna estatísticas da sessão atual"},
    {"name": "update_threshold", "description": "Ajusta threshold de decisão remotamente"}
  ],
  "resources": [
    {"name": "baseline://current", "description": "Baseline de calibração atual"},
    {"name": "features://latest_window", "description": "Última janela de 19 features"}
  ]
}
```

**b) Agente Central de Frota via MCP Client:**
Um agente central de gestão de frota descobre automaticamente todas as estações de monitoramento disponíveis e consulta cada uma uniformemente via MCP, sem precisar conhecer detalhes de implementação de cada veículo.

---

## 10. Knowledge Retrieval (RAG)

### Status Atual: ❌ Não implementado

O modelo MLP opera com pesos fixos treinados offline. Não há mecanismo de consulta a bases de conhecimento.

### Oportunidade de Evolução

**a) RAG para Contextualização de Alertas:**
Quando o sistema detecta fadiga, um módulo RAG consulta uma base de conhecimento para gerar recomendações contextuais:

```
Query: "Operador com PERCLOS 0.45, 2 microsleeps, turno de 6h, 23h local"
Base de Conhecimento: Regulações trabalhistas, protocolos de segurança, histórico do operador

Resposta gerada: "Alerta de fadiga severa. Regulamento [CTB Art. 329] 
recomenda pausa mínima de 30 min após 5.5h contínuas. Histórico mostra 
que este operador apresentou padrão similar ontem às 22h. Sugestão: 
encaminhar para pausa obrigatória."
```

**b) Agentic RAG para Refinamento de Busca:**
Se a primeira busca não encontrar regulação específica, o agente reformula a query e busca novamente (ex: procurar por "NR-35" se o operador trabalha em altura, ou "ANTT Resolução 5.232" se é motorista de carga).

**c) GraphRAG para Relações Complexas:**
Modelar relações entre operadores, turnos, veículos, rotas e incidentes em um grafo de conhecimento. Isso permite queries como: "Qual a taxa de fadiga em turnos noturnos na rota BR-153?" ou "Operadores que dirigem veículos do tipo X têm mais incidentes de fadiga?"

---

## 11. Guardrails / Safety

### Status Atual: ✅ Parcialmente implementado (nível de sinal)

O projeto já possui guardrails robustos no nível de processamento de sinais:

| Constraint | Guardrail | Módulo |
|------------|-----------|--------|
| C5 | Z-Norm sempre per-subject | `subject_calibrator_rt.py` |
| C13 | PERCLOS sobre EAR raw (nunca Z-norm) | `window_factory_rt.py` |
| C22 | Blink velocity clampada em [0.01, 5] EAR/s | `window_factory_rt.py` |
| C32 | Z-Score Clamp [-3, +3] | `window_factory_rt.py` |
| C33 | HeadPoseNeutralizer zera 4 features | `run_realtime_demo.py` |
| C34 | Filtro fisiológico na calibração | `subject_calibrator_rt.py` |

### Oportunidade de Evolução

**a) Guardrails de Saída com Validação Estruturada:**
Usar validação tipo Pydantic para garantir que a saída do sistema seja sempre consistente:

```python
from pydantic import BaseModel, validator

class FatigueOutput(BaseModel):
    label: Literal["Safe", "Danger"]
    prob_danger: float  # [0.0, 1.0]
    confidence: Literal["high", "medium", "low"]
    features_valid: bool
    
    @validator('prob_danger')
    def prob_in_range(cls, v):
        assert 0.0 <= v <= 1.0, "Probabilidade fora do range"
        return v
```

**b) Guardrails de Comportamento do Sistema:**

- **Rate limiter de alertas:** Máximo de 1 alerta sonoro a cada 60 segundos para evitar dessensibilização do operador
- **Cooldown pós-recalibração:** Após recalibrar, ignorar as primeiras 2 janelas (30s) pois podem ser instáveis
- **Watchdog de sistema:** Se o pipeline não produzir inferência por mais de 30s (crash silencioso), acionar alerta de "sistema indisponível"
- **Limite de confiança mínima:** Se `face_confidence < 0.3` por mais de 10 frames, suspender inferência e avisar que as condições não permitem monitoramento confiável

**c) Guardrails Éticos e de Privacidade:**

- Frames brutos nunca são armazenados — apenas features numéricas
- Dados de calibração são vinculados a IDs anônimos (não nomes)
- Log de features é opt-in e criptografado
- Sistema não pode ser usado para avaliação de desempenho — apenas segurança

---

## Matriz de Prioridade

A tabela abaixo prioriza os padrões por **impacto vs esforço de implementação**:

| Prioridade | Padrão | Impacto | Esforço | Justificativa |
|:---:|--------|:---:|:---:|---------------|
| 🥇 | **Reflection** | Alto | Médio | Detectar drift e calibração ruim melhora diretamente a acurácia |
| 🥇 | **Guardrails (expandido)** | Alto | Baixo | Validação Pydantic + watchdog são rápidos de implementar |
| 🥈 | **Memory Management** | Alto | Médio | SessionMemory + warm-start reduzem tempo operacional |
| 🥈 | **Parallelization** | Médio | Médio | Crítico no Raspberry Pi para manter FPS estável |
| 🥈 | **Planning** | Médio | Médio | Resposta graduada (ReAct) é mais segura que binário |
| 🥉 | **Routing** | Médio | Médio | Routing por confiança/ambiente melhora robustez |
| 🥉 | **Tool Use** | Médio | Médio | Integração com frota agrega valor operacional |
| 🥉 | **Multi-Agent** | Alto | Alto | Arquitetura elegante mas requer refatoração significativa |
| 4️⃣ | **MCP** | Médio | Alto | Só justifica em cenário de frota multi-veículo |
| 4️⃣ | **RAG** | Baixo | Alto | Mais útil para dashboard/relatórios que para RT |
| — | **Prompt Chaining** | — | — | Já implementado. Evoluir com metadados inter-estágio |

---

## Composição de Padrões: Visão do Sistema Evoluído

A verdadeira potência emerge da **composição de múltiplos padrões**. A arquitetura alvo combinaria:

```
┌─────────────────────────────────────────────────────────────┐
│                    PLANNER (Sessão)                          │
│  Define thresholds, checkpoints, estratégia de escalation   │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│              PIPELINE PARALELO (Chaining)                     │
│  Camera(thread1) → Features(thread2) → Window → Model        │
│                                                               │
│  ┌─────────┐  ┌──────────┐  ┌─────────────┐                 │
│  │ Routing  │→ │ Agentes  │→ │ Supervisor  │                 │
│  │ (confiança)│ │ Especial.│  │ (Multi-Agent)│                │
│  └─────────┘  └──────────┘  └──────┬──────┘                 │
│                                     │                         │
│  ┌──────────────┐  ┌───────────────┐│                        │
│  │ Reflection   │  │ Guardrails    ││                        │
│  │ (Drift Det.) │  │ (Pydantic +   ││                        │
│  │              │  │  Rate Limit)  ││                        │
│  └──────────────┘  └───────────────┘│                        │
└──────────────────────────────────────┼───────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    TOOL USE (Ações)                            │
│  Fleet API │ GPIO Buzzer │ Logger │ Retraining Pipeline       │
└──────────────────────────────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────┐
│                 MEMORY (Curto + Longo Prazo)                  │
│  SessionMemory │ OperatorProfiles (SQLite) │ FeatureLog       │
└──────────────────────────────────────────────────────────────┘
```

---

## Conclusão

O DeteccaoFadiga2 já possui uma base sólida como pipeline de visão computacional. A aplicação dos padrões agênticos o transformaria de um classificador binário reativo em um **sistema inteligente adaptativo** capaz de: se auto-avaliar (Reflection), planejar respostas graduadas (Planning), coordenar múltiplas análises especializadas (Multi-Agent), aprender com o histórico (Memory), e integrar-se ao ecossistema operacional (Tool Use + MCP).

A recomendação é adotar uma abordagem incremental: começar pelos padrões de maior impacto e menor esforço (Guardrails expandidos → Reflection → Memory), e progressivamente evoluir para os padrões mais arquiteturais (Multi-Agent → MCP) conforme o sistema se consolida em produção.
