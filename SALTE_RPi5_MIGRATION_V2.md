# SALTE — Guia de Migração: salte_inference V1 → V2 (Raspberry Pi 5)

**Autor:** Principal Engineer — CV/ML/Edge AI  
**Arquiteto:** Antonio Giulii  
**Data:** 26/02/2026  
**Base:** Conversa de deploy RPi 24/02 + TEV7 Agentic V3 (26/02) + SALTE_COMPLETE_HISTORY.md  
**Classificação:** Guia de Migração — Sprint 5.1 (Edge Deploy Hardened)

---

## 1 · Contexto: Por Que V2 Existe

O pacote `salte_inference` V1, criado em 24/02/2026, foi projetado para o **TEV7 original** (21 features, `mlp_v3.onnx`, `scaler_v7.pkl`). Durante a tentativa de deploy no Raspberry Pi 5, três bloqueios foram identificados:

### 1.1 Bloqueio #1 — MediaPipe Não Instala no RPi 5

**Erro real:** O Raspberry Pi OS Bookworm vem com **Python 3.13 + aarch64**. O pacote `mediapipe` do PyPI não tem wheel para essa combinação. `pip install mediapipe` falha com erro de compatibilidade.

**Opções avaliadas:**

| Opção | Descrição | Veredicto |
|-------|-----------|-----------|
| A — Compilar Python 3.12 | Instalar Python 3.12 from source, criar venv isolado | ❌ Fragilidade permanente, updates do OS podem quebrar |
| B — Rodar em x86 | Desenvolver no PC, RPi só coleta dados | ❌ Derrota o propósito de edge deployment |
| **C — ONNXFaceMeshBackend** | Rodar BlazeFace + FaceMesh como ONNX direto, sem pacote mediapipe | ✅ **Escolhido** — mesmos landmarks, zero dependência extra |

**Decisão:** Opção C. O MediaPipe internamente roda dois modelos (BlazeFace para detecção, FaceMesh para 468 landmarks). Convertê-los para ONNX e rodá-los via `onnxruntime` (que já instala no ARM64) elimina o pacote `mediapipe` inteiro.

### 1.2 Bloqueio #2 — Modelo TEV7 Evoluiu para Agentic V3

Em 26/02/2026, o TEV7 foi re-treinado com o framework agêntico V3. Mudanças:

| Aspecto | TEV7 Original (V1) | TEV7 Agentic V3 (V2) |
|---------|--------------------|-----------------------|
| Features | 21 (FEATURE_NAMES_21) | **19** (Grupo 8 microsleep removido por ablation) |
| Features removidas | — | `microsleep_count`, `microsleep_total_ms` |
| Modelo | FatigueMLP_V3 (21→128→64→2), ~13K params | **Wide MLP com skip + GELU** (19→?→?→2), 12.2K params |
| Input ONNX | `features_21` | `features_19` |
| Output ONNX | `logits` [batch, 2] | `logits` [batch, 2] |
| Threshold | 0.51 (em `mlp_v3_artifacts.json`) | **0.41** (em `inference_config.json`) |
| Scaler | `scaler_v7.pkl` (pickle de SelectiveScalerV2) | **Embutido no JSON** (scaler_mean + scaler_scale + scaler_indices) |
| BAcc (CV) | ~0.65 | **0.678** |
| Safe Recall | ~0.55 | **0.795** |
| Gates | G1–G5 | **G1–G7** (inclui EquivalenceTestHarness) |
| Arquivo ONNX | `mlp_v3.onnx` (~60KB) | **`best_model.onnx` + `.data`** (50.6KB) |

### 1.3 Bloqueio #3 — Dependências Desnecessárias

O V1 importava PyTorch como dependência hard (~2GB no ARM64) apenas para ter a classe `FatigueMLP_V3`. Como a inferência é 100% via ONNX Runtime, o PyTorch é desnecessário no RPi.

O `scaler_v7.pkl` exigia a classe `SelectiveScalerV2` presente em memória para desserializar (pickle). No V2, o scaler está como arrays numéricos no JSON — zero dependência de classe.

---

## 2 · Arquitetura Comparada

### 2.1 Pacote V1 (24/02/2026)

```
~/salte/
├── models/
│   ├── mlp_v3.onnx              # 21 features input
│   ├── mlp_v3.onnx.data         # pesos externos
│   ├── scaler_v7.pkl            # pickle (requer SelectiveScalerV2 class)
│   ├── mlp_v3_artifacts.json    # threshold 0.51
│   └── tev7_summary.json
├── salte_inference/
│   ├── __init__.py
│   ├── feature_extractor_rt.py  # MediapipeBackend (import mediapipe ❌)
│   ├── model_loader.py          # import torch ❌, pickle.load(scaler)
│   ├── window_factory_rt.py     # FEATURE_NAMES_21 (21 features)
│   ├── subject_calibrator_rt.py
│   ├── run_realtime_demo.py     # cv2.VideoCapture (sem picamera2)
│   └── offline_eval.py
└── requirements.txt             # mediapipe, torch, ...
```

**Problemas:**
- `mediapipe` não instala no Python 3.13 aarch64
- `torch` ~2GB desnecessário
- `pickle` do scaler requer classe em memória
- `cv2.VideoCapture` não funciona com AI Camera IMX500
- 21 features — modelo desatualizado

### 2.2 Pacote V2 (26/02/2026) — Target

```
~/salte/
├── models/
│   ├── best_model.onnx            # 19 features input (TEV7 Agentic V3)
│   ├── best_model.onnx.data       # pesos externos
│   ├── inference_config.json      # scaler + features + threshold 0.41
│   ├── tev7_agentic_v3_summary.json
│   ├── blazeface_detector.onnx    # BlazeFace (face detection) ~200KB
│   └── face_mesh_landmark.onnx    # FaceMesh 468 landmarks ~3MB
├── salte_inference/
│   ├── __init__.py
│   ├── feature_extractor_rt.py    # ONNXFaceMeshBackend (onnxruntime only ✅)
│   ├── model_loader.py            # ONNX-only, JSON scaler (sem torch, sem pickle)
│   ├── window_factory_rt.py       # FEATURE_NAMES_19 (19 features)
│   ├── subject_calibrator_rt.py   # (sem alterações)
│   ├── run_realtime_demo.py       # picamera2 + headless + argparse
│   └── offline_eval.py
└── requirements.txt               # onnxruntime, opencv-python-headless, numpy, picamera2
```

---

## 3 · Mudanças Arquivo por Arquivo

### 3.1 `inference_config.json` (NOVO — substitui scaler_v7.pkl + mlp_v3_artifacts.json)

Este arquivo agora é o **single source of truth** para o pipeline no RPi. Contém:

```
{
  "model_type": "mlp",
  "threshold": 0.41,                           ← threshold de classificação
  "feature_names": [...19 nomes...],           ← ordem das features no input
  "n_features": 19,
  "scaler_mean": [...12 valores...],           ← média do StandardScaler (só features contínuas)
  "scaler_scale": [...12 valores...],          ← desvio-padrão do StandardScaler
  "scaler_indices": [0,1,2,3,4,5,6,7,8,9,15,16], ← índices das features que são escaladas
  "scaler_feature_names": [...12 nomes...],    ← nomes das features escaladas
  "training_stats": {...},                     ← ranges esperados (para validação C31)
  "balanced_accuracy": 0.7148,
  "safe_recall": 0.7946,
  "c28_pass": true
}
```

**Scaler Logic:**
- 12 features **contínuas** (EAR, velocities, MAR, head pose, blink velocities) → `(x - mean) / scale`
- 7 features **passthrough** (blink_count, blink_rate_per_min, blink_mean_dur_ms, perclos_p80_mean, perclos_p80_max, long_blink_pct, blink_regularity) → sem transformação

**Vantagem vs pickle:** Não precisa da classe `SelectiveScalerV2` em memória. Qualquer linguagem/plataforma lê JSON.

### 3.2 `model_loader.py` — Reescrita Completa

**V1 (problemas):**
```python
import torch                          # ❌ 2GB desnecessários
import pickle                         # ❌ requer classe SelectiveScalerV2
from selective_scaler import ...      # ❌ arquivo não incluído no pacote

class FatigueMLP_V3(nn.Module): ...   # ❌ só serve para PyTorch
```

**V2 (correções):**
```python
import onnxruntime as ort             # ✅ já instalado para o MLP
import json                           # ✅ stdlib
import numpy as np                    # ✅ já instalado

# Scaler via JSON
config = json.load(open('inference_config.json'))
mean = np.array(config['scaler_mean'])
scale = np.array(config['scaler_scale'])
indices = config['scaler_indices']
threshold = config['threshold']

# Inferência
def scale_features(raw: np.ndarray) -> np.ndarray:
    scaled = raw.copy()
    scaled[indices] = (scaled[indices] - mean) / scale
    return scaled

def predict(session, features_19: np.ndarray) -> tuple[str, float]:
    logits = session.run(None, {"features_19": features_19})[0]
    probs = softmax(logits[0])
    return ("DANGER" if probs[1] >= threshold else "SAFE"), probs[1]
```

**Dependências eliminadas:** `torch`, `pickle`, `selective_scaler.py`

### 3.3 `feature_extractor_rt.py` — Backend Swap

**V1:** Usava `MediapipeBackend` que importa `mediapipe`.

**V2:** Adiciona `ONNXFaceMeshBackend` que roda os modelos do MediaPipe via `onnxruntime`:

```python
class ONNXFaceMeshBackend:
    """
    Roda BlazeFace + FaceMesh como modelos ONNX puros.
    Retorna os mesmos 468 landmarks normalizados do MediaPipe.
    Mesmos índices (LEFT_EYE, RIGHT_EYE, MOUTH, POSE).
    """
    def __init__(self, detector_path, mesh_path):
        self.detector = ort.InferenceSession(detector_path)  # BlazeFace
        self.mesh = ort.InferenceSession(mesh_path)          # FaceMesh

    def get_landmarks(self, frame_bgr):
        # 1. BlazeFace: detecta face bbox
        # 2. Crop + align face
        # 3. FaceMesh: 468 landmarks 3D normalizados
        return landmarks, has_face
```

**Interface idêntica** ao `MediapipeBackend` — `LandmarkBackend` Protocol mantido. Nenhuma mudança no pipeline downstream.

**Modelos necessários:**
- `blazeface_detector.onnx` (~200KB) — detecção de face
- `face_mesh_landmark.onnx` (~3MB) — 468 landmarks 3D

Ambos convertidos dos TFLite oficiais do Google via `tf2onnx` no Colab.

### 3.4 `window_factory_rt.py` — 21 → 19 Features

**V1:**
```python
FEATURE_NAMES_21 = [
    'ear_mean', 'ear_std', 'ear_min',
    'ear_vel_mean', 'ear_vel_std',
    'mar_mean',
    'pitch_mean', 'pitch_std', 'yaw_std', 'roll_std',
    'blink_count', 'blink_rate_per_min', 'blink_mean_dur_ms',
    'perclos_p80_mean', 'perclos_p80_max',
    'blink_closing_vel_mean', 'blink_opening_vel_mean',
    'long_blink_pct', 'blink_regularity',
    'microsleep_count', 'microsleep_total_ms',     # ← REMOVIDAS
]
```

**V2:**
```python
FEATURE_NAMES_19 = [
    'ear_mean', 'ear_std', 'ear_min',              # Grupo 1: EAR stats (3)
    'ear_vel_mean', 'ear_vel_std',                  # Grupo 2: EAR velocity (2)
    'mar_mean',                                     # Grupo 3: MAR (1)
    'pitch_mean', 'pitch_std', 'yaw_std', 'roll_std', # Grupo 4: Head pose (4)
    'blink_count', 'blink_rate_per_min', 'blink_mean_dur_ms', # Grupo 5: Blink (3)
    'perclos_p80_mean', 'perclos_p80_max',          # Grupo 6: PERCLOS (2)
    'blink_closing_vel_mean', 'blink_opening_vel_mean', # Grupo 7: Blink morph (4)
    'long_blink_pct', 'blink_regularity',
]
# Grupo 8 (microsleep) REMOVIDO — ablation Δ5 no TEV7 Agentic V3
```

**Código de agregação do microsleep:** Pode permanecer no `aggregate_window()` para logging/telemetria, mas **não deve ser incluído** no vetor de features enviado ao modelo.

### 3.5 `run_realtime_demo.py` — picamera2 + Headless

**V1:** `cv2.VideoCapture(source)` — não funciona com AI Camera IMX500.

**V2:**
```python
# Suporta 3 modos de captura:
# 1. picamera2 (AI Camera / CSI) — default no RPi
# 2. cv2.VideoCapture (USB webcam / IP stream) — fallback
# 3. Arquivo de vídeo (offline)

if args.source == 'picamera2':
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    ))
    cam.start()
```

**Headless mode:** `--headless` flag desabilita `cv2.imshow()` (necessário via SSH sem display). Output vai para stdout/log.

**Argparse completo:**
```
python3 run_realtime_demo.py \
    --source picamera2 \
    --model-dir ~/salte/models/ \
    --headless \
    --log-file ~/salte/logs/session.log
```

### 3.6 `subject_calibrator_rt.py` — Sem Alterações

A calibração per-subject (C5, C6-V2) não depende do número de features nem do backend de landmarks. O warm-up de 120s com best-segment strategy permanece idêntico.

**Melhoria menor (P2):** Adicionar log periódico durante warm-up headless para que o operador via SSH saiba que o sistema está calibrando.

### 3.7 `offline_eval.py` — Atualizar FEATURE_NAMES

Mesma mudança do `window_factory_rt.py`: 21 → 19 features. Remover import de `FatigueMLP_V3` e `_infer_device` (PyTorch). Usar ONNX Runtime.

### 3.8 `requirements.txt` — Simplificado

**V1:**
```
mediapipe>=0.10.0        # ❌ não instala no RPi
torch>=2.0               # ❌ 2GB
onnxruntime>=1.18
opencv-python-headless
numpy
picamera2
```

**V2:**
```
onnxruntime>=1.18.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
picamera2>=0.3.12
```

**Total de dependências:** 4 pacotes (~50MB vs ~2.5GB do V1).

---

## 4 · Conversão dos Modelos BlazeFace + FaceMesh (Colab)

### 4.1 Por Que Converter

O MediaPipe distribui seus modelos internos em formato TFLite. O RPi 5 não tem o pacote `mediapipe` (que contém o runtime TFLite embutido), então precisamos converter para ONNX e rodar via `onnxruntime`.

### 4.2 Notebook de Conversão (rodar no Colab)

```python
# Cell 1: Install
!pip install -q tf2onnx onnx onnxruntime

# Cell 2: Download TFLite models from Google's MediaPipe repository
import urllib.request

# BlazeFace Short Range (face detection)
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "blazeface_short.tflite"
)

# FaceMesh V2 (468 landmarks)
urllib.request.urlretrieve(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "face_landmarker.task"
)

# Cell 3: Extract individual TFLite from .task bundle
# O .task é um ZIP contendo múltiplos TFLite
import zipfile
with zipfile.ZipFile("face_landmarker.task", 'r') as z:
    z.extractall("face_landmarker_extracted/")
    for f in z.namelist():
        print(f"  {f} ({z.getinfo(f).file_size/1024:.1f} KB)")

# Cell 4: Convert BlazeFace to ONNX
!python -m tf2onnx.convert \
    --tflite blazeface_short.tflite \
    --output blazeface_detector.onnx \
    --opset 17

# Cell 5: Convert FaceMesh to ONNX
# (O modelo exato depende do conteúdo extraído do .task)
!python -m tf2onnx.convert \
    --tflite face_landmarker_extracted/face_landmarks_detector.tflite \
    --output face_mesh_landmark.onnx \
    --opset 17

# Cell 6: Validate
import onnxruntime as ort
for name in ['blazeface_detector.onnx', 'face_mesh_landmark.onnx']:
    sess = ort.InferenceSession(name)
    for inp in sess.get_inputs():
        print(f"{name}: input {inp.name} shape={inp.shape} dtype={inp.type}")
    for out in sess.get_outputs():
        print(f"{name}: output {out.name} shape={out.shape}")
    print(f"  Size: {os.path.getsize(name)/1024:.1f} KB\n")

# Cell 7: Download para Google Drive
from google.colab import drive
drive.mount('/content/drive')
import shutil
dest = '/content/drive/MyDrive/salte_models/'
os.makedirs(dest, exist_ok=True)
shutil.copy('blazeface_detector.onnx', dest)
shutil.copy('face_mesh_landmark.onnx', dest)
print("Models saved to Google Drive")
```

> **NOTA:** O bundle `.task` pode ter estrutura interna diferente dependendo da versão. Verificar os nomes dos TFLite extraídos. Projetos de referência: `face-detection-onnx` do IntelliProve, `py-feat/mp_facemesh_v2` no HuggingFace.

### 4.3 Re-export do best_model.onnx (Opcional)

O `best_model.onnx` atual usa pesos externos (`.data` file). Para simplificar o deploy, pode-se re-exportar com pesos embutidos:

```python
# No Colab (onde o modelo foi treinado)
import onnx

model = onnx.load("best_model.onnx")
onnx.save(model, "best_model_embedded.onnx",
          save_as_external_data=False)  # ← pesos dentro do .onnx
# Resultado: um único arquivo ~50KB
```

Isso elimina o `.data` e garante que um único arquivo contém tudo.

---

## 5 · Scaler: Pickle → JSON (Detalhe Técnico)

### 5.1 O Problema do Pickle

O V1 usava `scaler_v7.pkl`, que é um pickle da classe `SelectiveScalerV2`. Para desserializar:

```python
# V1 — FALHA se SelectiveScalerV2 não está em sys.modules
with open('scaler_v7.pkl', 'rb') as f:
    scaler = pickle.load(f)  # → AttributeError: Can't get attribute 'SelectiveScalerV2'
```

Isso criou o bug P1 da auditoria V1: o `selective_scaler.py` nunca foi incluído no pacote.

### 5.2 A Solução do V2

O `inference_config.json` já contém os parâmetros numéricos do scaler. A lógica é trivial:

```python
import json
import numpy as np

with open('inference_config.json') as f:
    cfg = json.load(f)

SCALER_MEAN   = np.array(cfg['scaler_mean'], dtype=np.float32)    # shape: (12,)
SCALER_SCALE  = np.array(cfg['scaler_scale'], dtype=np.float32)   # shape: (12,)
SCALER_IDX    = cfg['scaler_indices']                              # [0,1,2,...,15,16]
THRESHOLD     = cfg['threshold']                                   # 0.41
FEATURE_NAMES = cfg['feature_names']                               # 19 nomes

def scale_features(raw_19: np.ndarray) -> np.ndarray:
    """C11: Selective scaling — contínuas normalizadas, passthrough intactas."""
    scaled = raw_19.copy()
    scaled[SCALER_IDX] = (scaled[SCALER_IDX] - SCALER_MEAN) / SCALER_SCALE
    return scaled
```

### 5.3 Mapeamento Scaler ↔ Features

```
Index | Feature Name              | Escalada? | scaler_indices pos
------|---------------------------|-----------|--------------------
  0   | ear_mean                  |    ✅     | 0
  1   | ear_std                   |    ✅     | 1
  2   | ear_min                   |    ✅     | 2
  3   | ear_vel_mean              |    ✅     | 3
  4   | ear_vel_std               |    ✅     | 4
  5   | mar_mean                  |    ✅     | 5
  6   | pitch_mean                |    ✅     | 6
  7   | pitch_std                 |    ✅     | 7
  8   | yaw_std                   |    ✅     | 8
  9   | roll_std                  |    ✅     | 9
 10   | blink_count               |    ❌     | — (passthrough)
 11   | blink_rate_per_min        |    ❌     | — (passthrough)
 12   | blink_mean_dur_ms         |    ❌     | — (passthrough)
 13   | perclos_p80_mean          |    ❌     | — (passthrough)
 14   | perclos_p80_max           |    ❌     | — (passthrough)
 15   | blink_closing_vel_mean    |    ✅     | 10
 16   | blink_opening_vel_mean    |    ✅     | 11
 17   | long_blink_pct            |    ❌     | — (passthrough)
 18   | blink_regularity          |    ❌     | — (passthrough)
```

As 7 features passthrough já estão em escalas naturais (contagens, percentuais, ms). As 12 features contínuas precisam de Z-norm porque suas escalas variam enormemente (EAR ~0.2–0.4 vs pitch ~-30°–+30° vs velocities ~-5–+5 EAR/s).

---

## 6 · Modelo best_model.onnx — Análise Detalhada

### 6.1 Arquitetura (Wide MLP + Skip Connection + GELU)

```
Input: features_19 [batch, 19]
  │
  ├──────────────────────────────────────────┐
  │                                          │ (skip connection)
  ▼                                          │
Gemm: wide.0 (19 → hidden)                  │
  ▼                                          │
GELU activation                              │
  ▼                                          │
Gemm: wide.4 (hidden → hidden)              │
  ▼                                          │
GELU activation                              │
  ▼                                          │
  + ◄───────── Gemm: skip (19 → hidden) ────┘
  │
  ▼
GELU activation
  ▼
Gemm: head.2 (hidden → 2)
  ▼
Output: logits [batch, 2]    →    softmax    →    P(danger) >= 0.41?
```

### 6.2 Especificações

| Propriedade | Valor |
|-------------|-------|
| Input name | `features_19` |
| Input shape | `[batch, 19]` (dynamic batch) |
| Output name | `logits` |
| Output shape | `[batch, 2]` (safe, danger) |
| Parameters | 12,229 |
| File size | 50.6 KB (com external data) |
| Opset | 17 |
| Activation | GELU |
| Classificação | Softmax → threshold 0.41 |

### 6.3 Benchmark de Inferência

| Plataforma | Latência (1000 runs) | Provider |
|------------|---------------------|----------|
| x86 (Colab/PC) | **0.015ms** mean, 0.020ms P95 | CPUExecutionProvider |
| ARM Cortex-A76 (RPi 5) | **~0.1–0.3ms** estimado | CPUExecutionProvider |

A inferência do MLP é **irrelevante no budget de latência**. O bottleneck é 100% o face mesh (~35–50ms/frame).

---

## 7 · Pipeline de Inferência Completo no RPi 5

### 7.1 Fluxo por Frame

```
┌─────────────────────────────────────────────────────────────┐
│  picamera2 → frame BGR (640×480)                            │
│  Latência: ~2ms (CSI DMA)                                   │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  ONNXFaceMeshBackend                                        │
│  1. BlazeFace ONNX → bbox da face (~5ms)                    │
│  2. Crop + align → face normalizada                         │
│  3. FaceMesh ONNX → 468 landmarks 3D (~25-40ms)            │
│  Latência total: ~30-50ms                                   │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  feature_extractor_rt.py                                    │
│  compute_ear(landmarks) → EAR L/R/avg                       │
│  compute_mar(landmarks) → MAR                               │
│  solvePnP(landmarks) → pitch/yaw/roll                       │
│  BlinkDetectorV3 → onset/offset/velocities                  │
│  PERCLOS P80 → rolling 30s                                  │
│  Latência: ~1-2ms                                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  subject_calibrator_rt.py (primeiros 120s)                  │
│  Best-segment → Z-Norm per-subject (C5, C6-V2)             │
│  Após calibração: Z-Norm em tempo real                      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  window_factory_rt.py                                       │
│  Ring buffer 15s (450 frames @ 30fps)                       │
│  A cada stride (2s): aggregate → 19 features                │
│  Latência: ~0.5ms                                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  model_loader.py (scale + predict)                          │
│  1. scale_features(raw_19) — C11 SelectiveScaler via JSON   │
│  2. session.run({"features_19": scaled})                    │
│  3. softmax(logits) → P(danger)                             │
│  4. P(danger) >= 0.41 → DANGER                              │
│  Latência: ~0.3ms                                           │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Alerting (GPIO)                                            │
│  SAFE  → LED verde                                          │
│  DANGER → LED vermelho + buzzer + MQTT                      │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Budget de Latência Total

| Etapa | Latência (RPi 5) |
|-------|----------------:|
| Captura (CSI) | ~2ms |
| BlazeFace ONNX | ~5ms |
| FaceMesh ONNX | ~25–40ms |
| Feature extraction | ~1ms |
| Calibration | ~0.1ms |
| Window aggregation | ~0.5ms |
| MLP inference | ~0.3ms |
| **Total** | **~34–49ms** |
| **FPS efetivo** | **~20–29 FPS** |

---

## 8 · Plano de Implementação

### Fase 1 — Preparação no Colab (30 min)

| # | Ação | Prioridade |
|---|------|-----------|
| 1.1 | Converter BlazeFace + FaceMesh TFLite → ONNX (Seção 4.2) | P0 |
| 1.2 | Re-exportar `best_model.onnx` com `save_as_external_data=False` | P1 |
| 1.3 | Validar conversão: comparar landmarks ONNX vs MediaPipe original | P0 |

### Fase 2 — Implementação do Pacote V2 (4–6h)

| # | Ação | Prioridade | Dependência |
|---|------|-----------|-------------|
| 2.1 | Reescrever `model_loader.py` (ONNX-only, JSON scaler) | P0 | 1.2 |
| 2.2 | Implementar `ONNXFaceMeshBackend` em `feature_extractor_rt.py` | P0 | 1.1 |
| 2.3 | Atualizar `window_factory_rt.py` para 19 features | P0 | — |
| 2.4 | Adaptar `run_realtime_demo.py` para picamera2 + headless | P0 | 2.1–2.3 |
| 2.5 | Atualizar `offline_eval.py` | P1 | 2.1, 2.3 |
| 2.6 | Atualizar `requirements.txt` | P0 | — |
| 2.7 | Atualizar `__init__.py` | P0 | — |

### Fase 3 — Deploy no RPi 5 (2h)

| # | Ação | Prioridade |
|---|------|-----------|
| 3.1 | `scp` dos models/ + salte_inference/ para `~/salte/` | P0 |
| 3.2 | `pip install -r requirements.txt --break-system-packages` | P0 |
| 3.3 | Smoke test: `python3 -c "import onnxruntime; print('OK')"` | P0 |
| 3.4 | Benchmark FaceMesh ONNX standalone (FPS) | P0 |
| 3.5 | Pipeline E2E: câmera → landmarks → features → MLP → output | P0 |
| 3.6 | Stress test 1h contínuo | P1 |

### Gate de Validação

| Critério | Meta |
|----------|------|
| FPS pipeline E2E | ≥ 15 FPS sustentado |
| Landmarks idênticos ao MediaPipe | Diff < 1e-3 (validação no Colab) |
| Scaler JSON produz mesmas features que scaler_v7.pkl | Diff < 1e-5 |
| 19 features no vetor de input | Schema validation |
| Threshold 0.41 usado | Config validation |
| CPU temp sustentada | < 75°C com active cooler |
| RAM steady-state | < 1.5GB |
| Zero dependência de mediapipe/torch | `pip list` audit |

---

## 9 · Changelog TEV7 V1 → TEV7 Agentic V3

Referência do `tev7_agentic_v3_summary.json`:

| Change | Detalhe |
|--------|---------|
| Δ5 | 21→19 features (Grupo 8 microsleep removido por ablation) |
| Δ6 | XGB interactions 3→2 (microsleep_x_long_blink removido) |
| Δ7 | EquivalenceTestHarness + Gate G7 (training↔edge parity) |
| Δ8 | FEATURE_NAMES alias (sem número hardcoded no nome) |

**Padrões Agênticos implementados:** Prompt Chaining, Parallelization, Reflection, Tool Use, Memory Management, Exception Handling, Guardrails, Resource-Aware, Eval & Monitoring, Human-in-the-Loop, Equivalence Testing.

---

## 10 · Referências

| # | Documento | Uso |
|---|-----------|-----|
| 1 | `inference_config.json` (upload 26/02) | Scaler, features, threshold |
| 2 | `tev7_agentic_v3_summary.json` (upload 26/02) | Métricas, gates, changelog |
| 3 | `best_model.onnx` + `.data` (upload 26/02) | Modelo ONNX (19 features) |
| 4 | Conversa deploy RPi 24/02 (c82c5ed7) | Auditoria V1, decisão Opção C |
| 5 | `SALTE_RPi5_DEPLOY.md` (23/02) | Arquitetura de hardware RPi 5 |
| 6 | `tev7.py` (projeto) | TEV7 original (21 features, constraints C1–C31) |
| 7 | `ffv5.py` (projeto) | FeatureFactory V5 (pipeline de extração) |
| 8 | `SALTE_COMPLETE_HISTORY.md` (projeto) | Histórico completo do projeto |

---

*Documento criado em 26/02/2026 — Guia de migração salte_inference V1→V2 para Raspberry Pi 5 com TEV7 Agentic V3.*
