# Plano de Execução: Memory Management + Parallelization para DeteccaoFadiga

## Implementação dos Padrões Agênticos — Fase 2

**Projeto:** DeteccaoFadiga  
**Repo:** github.com/euBrunoMelo/DeteccaoFadiga  
**Pré-requisitos implementados:** Guardrails (G1-G4) + Reflection (R1-R3) — 42 testes passando  
**Estimativa total:** ~5 sprints de 1 semana  
**Dependências novas:** nenhuma (usa apenas stdlib: `threading`, `queue`, `sqlite3`, `json`)

---

## PARTE A — MEMORY MANAGEMENT

O Memory Management dota o sistema de memória de curto prazo (estado da sessão em execução) e de longo prazo (persistência entre sessões via SQLite). Hoje, cada reinicialização do sistema perde toda informação acumulada — calibrações, histórico de alertas, perfis de operador. Memory resolve isso.

### Fase M1 — Memória de Curto Prazo: SessionMemory (Semana 1)

**Objetivo:** Criar um estado de sessão enriquecido que acumula informações ao longo da operação. Hoje, o estado está disperso entre variáveis locais no loop principal (`consecutive_danger` no `BehaviorGuardRails`, buffers no `DriftReflector`, etc.) sem uma visão unificada.

**Arquivo novo:** `SALTE_INFERENCE/memory.py`

```python
"""
Memory Management para DeteccaoFadiga.

Dois componentes:
  M1. SessionMemory    — estado de curto prazo (sessão atual em RAM)
  M2. OperatorStore    — memória de longo prazo (SQLite, persiste entre sessões)
  M3. FeatureLogger    — gravação de features para retraining futuro
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np


@dataclass
class SessionMemory:
    """
    M1: Memória de curto prazo — estado completo da sessão em execução.
    
    Centraliza toda informação acumulada durante uma sessão de monitoramento.
    Substitui variáveis dispersas no loop principal por um objeto coerente
    e consultável. Vive exclusivamente em RAM — destruído ao encerrar.
    
    Ref: Agentic Design Patterns Ch.8 — Short-term memory / Session State.
    """
    
    # ── Identificação ──
    operator_id: str = "unknown"
    session_start_time: float = field(default_factory=time.monotonic)
    
    # ── Contadores globais ──
    total_frames: int = 0
    total_windows: int = 0
    total_danger_windows: int = 0
    total_safe_windows: int = 0
    total_watch_windows: int = 0
    total_critical_windows: int = 0
    
    # ── Streaks ──
    max_consecutive_danger: int = 0
    current_consecutive_danger: int = 0
    
    # ── Calibração ──
    calibration_count: int = 0
    calibration_timestamps: List[float] = field(default_factory=list)
    last_calibration_ear_mean: float = 0.0
    last_calibration_verdict: str = ""   # "accept" | "retry" | "use_with_caution"
    
    # ── Trends (janela deslizante para análise temporal) ──
    prob_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=120)  # últimas 120 janelas (~30 min)
    )
    perclos_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    ear_mean_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=120)
    )
    
    # ── Alertas ──
    total_alerts_triggered: int = 0
    total_alerts_suppressed: int = 0
    total_recalibrations_auto: int = 0
    
    # ── Microsleeps ──
    microsleep_timestamps: List[float] = field(default_factory=list)
    total_microsleep_ms: float = 0.0
    
    def on_window(
        self,
        label: str,
        prob_danger: float,
        alert_level_name: str,
        perclos: float,
        ear_mean: float,
        microsleep_count: float,
        microsleep_total_ms: float,
    ) -> None:
        """Registra uma nova janela de inferência na memória."""
        self.total_windows += 1
        self.prob_history.append(prob_danger)
        self.perclos_history.append(perclos)
        self.ear_mean_history.append(ear_mean)
        
        if label == "Danger":
            self.total_danger_windows += 1
            self.current_consecutive_danger += 1
            self.max_consecutive_danger = max(
                self.max_consecutive_danger,
                self.current_consecutive_danger,
            )
        else:
            self.current_consecutive_danger = 0
            self.total_safe_windows += 1
        
        if alert_level_name == "WATCH":
            self.total_watch_windows += 1
        elif alert_level_name == "CRITICAL":
            self.total_critical_windows += 1
        
        if microsleep_count > 0:
            now = time.monotonic()
            for _ in range(int(microsleep_count)):
                self.microsleep_timestamps.append(now)
            self.total_microsleep_ms += microsleep_total_ms
    
    def on_calibration(
        self, ear_mean: float, verdict: str
    ) -> None:
        """Registra uma calibração."""
        self.calibration_count += 1
        self.calibration_timestamps.append(time.monotonic())
        self.last_calibration_ear_mean = ear_mean
        self.last_calibration_verdict = verdict
    
    def on_auto_recalibration(self) -> None:
        self.total_recalibrations_auto += 1
    
    def on_alert(self, triggered: bool) -> None:
        if triggered:
            self.total_alerts_triggered += 1
        else:
            self.total_alerts_suppressed += 1
    
    @property
    def session_duration_sec(self) -> float:
        return time.monotonic() - self.session_start_time
    
    @property
    def danger_ratio(self) -> float:
        if self.total_windows == 0:
            return 0.0
        return self.total_danger_windows / self.total_windows
    
    @property
    def avg_prob_danger(self) -> float:
        if not self.prob_history:
            return 0.0
        return float(np.mean(self.prob_history))
    
    @property
    def perclos_trend_slope(self) -> float:
        """Slope do PERCLOS ao longo do tempo. Positivo = piorando."""
        if len(self.perclos_history) < 10:
            return 0.0
        y = np.array(self.perclos_history)
        x = np.arange(len(y))
        # Regressão linear simples
        n = len(x)
        slope = (n * np.dot(x, y) - x.sum() * y.sum()) / (
            n * np.dot(x, x) - x.sum() ** 2 + 1e-12
        )
        return float(slope)
    
    def summary(self) -> Dict[str, object]:
        """Resumo da sessão para logging ou display."""
        return {
            "operator_id": self.operator_id,
            "duration_min": round(self.session_duration_sec / 60, 1),
            "total_windows": self.total_windows,
            "danger_ratio": round(self.danger_ratio, 3),
            "avg_prob": round(self.avg_prob_danger, 3),
            "max_consec_danger": self.max_consecutive_danger,
            "total_microsleeps": len(self.microsleep_timestamps),
            "total_microsleep_ms": round(self.total_microsleep_ms, 0),
            "perclos_trend": round(self.perclos_trend_slope, 6),
            "calibrations": self.calibration_count,
            "auto_recalibrations": self.total_recalibrations_auto,
            "alerts_triggered": self.total_alerts_triggered,
        }
```

**Integração no `run_realtime_demo.py`:**

```python
# Na inicialização (após guardrails + reflection):
from .memory import SessionMemory
session = SessionMemory(operator_id=f"op-{camera_index}")

# Após calibração (G4):
session.on_calibration(b.ear_mean, verdict.recommendation)

# Após cada inferência (já dentro do bloco window_feats):
session.on_window(
    label=output.label,
    prob_danger=output.prob_danger,
    alert_level_name=output.alert_level.name,
    perclos=output.perclos,
    ear_mean=window_feats.get("ear_mean", 0.0),
    microsleep_count=output.microsleep_count,
    microsleep_total_ms=window_feats.get("microsleep_total_ms", 0.0),
)

# Após should_sound_alert:
session.on_alert(triggered=alert_triggered)

# No finally (ao encerrar):
print(f"[session] Resumo final: {session.summary()}")
```

**Critério de aceite M1:**

- `session.summary()` retorna dict completo ao encerrar sessão
- `danger_ratio` calculado corretamente após 50 janelas mistas
- `perclos_trend_slope > 0` quando PERCLOS cresce ao longo do tempo
- `max_consecutive_danger` rastreia o pico corretamente

---

### Fase M2 — Memória de Longo Prazo: OperatorStore (Semana 2)

**Objetivo:** Persistir perfis de operadores entre sessões via SQLite. Permite: warm-start de calibração (reduzir 120s → 30s), detecção de degradação crônica, e perfis de risco individuais.

**Classe no `memory.py`:**

```python
import sqlite3
import json
from pathlib import Path


class OperatorStore:
    """
    M2: Memória de longo prazo — persiste perfis entre sessões via SQLite.
    
    Armazena:
    - Baselines de calibração (ear_mean, ear_std, etc.) por operador
    - Resumos de sessões anteriores (danger_ratio, microsleeps, duração)
    - Timestamps de última sessão para cálculo de descanso
    
    Ref: Agentic Design Patterns Ch.8 — Long-term memory / Persistent Storage.
    """
    
    DB_NAME = "operator_memory.db"
    
    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or self.DB_NAME
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()
    
    def _init_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS operator_profiles (
                operator_id   TEXT PRIMARY KEY,
                ear_mean      REAL,
                ear_std       REAL,
                mar_mean      REAL,
                pitch_mean    REAL,
                yaw_mean      REAL,
                sessions_count INTEGER DEFAULT 0,
                total_danger_ratio REAL DEFAULT 0.0,
                avg_session_minutes REAL DEFAULT 0.0,
                last_session_end TEXT,
                created_at    TEXT DEFAULT (datetime('now')),
                updated_at    TEXT DEFAULT (datetime('now'))
            );
            
            CREATE TABLE IF NOT EXISTS session_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                operator_id   TEXT NOT NULL,
                started_at    TEXT NOT NULL,
                ended_at      TEXT DEFAULT (datetime('now')),
                duration_min  REAL,
                total_windows INTEGER,
                danger_ratio  REAL,
                avg_prob      REAL,
                max_consec_danger INTEGER,
                total_microsleeps INTEGER,
                total_microsleep_ms REAL,
                perclos_trend REAL,
                calibrations  INTEGER,
                summary_json  TEXT,
                FOREIGN KEY (operator_id) REFERENCES operator_profiles(operator_id)
            );
        """)
        self._conn.commit()
    
    def get_profile(self, operator_id: str) -> Optional[Dict]:
        """Retorna perfil do operador ou None se não existe."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT * FROM operator_profiles WHERE operator_id = ?",
            (operator_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    
    def upsert_profile_from_calibration(
        self,
        operator_id: str,
        baseline,  # SubjectBaseline
    ) -> None:
        """
        Atualiza ou cria perfil com dados da calibração atual.
        Usa média exponencial com o baseline anterior para suavizar.
        """
        existing = self.get_profile(operator_id)
        alpha = 0.3  # peso da calibração nova vs histórica
        
        if existing and existing["ear_mean"] is not None:
            ear_mean = alpha * baseline.ear_mean + (1 - alpha) * existing["ear_mean"]
            ear_std = alpha * baseline.ear_std + (1 - alpha) * existing["ear_std"]
            mar_mean = alpha * baseline.mar_mean + (1 - alpha) * existing["mar_mean"]
            pitch_mean = alpha * baseline.pitch_mean + (1 - alpha) * existing["pitch_mean"]
            yaw_mean = alpha * baseline.yaw_mean + (1 - alpha) * existing["yaw_mean"]
        else:
            ear_mean = baseline.ear_mean
            ear_std = baseline.ear_std
            mar_mean = baseline.mar_mean
            pitch_mean = baseline.pitch_mean
            yaw_mean = baseline.yaw_mean
        
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO operator_profiles 
                (operator_id, ear_mean, ear_std, mar_mean, pitch_mean, yaw_mean)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(operator_id) DO UPDATE SET
                ear_mean = ?, ear_std = ?, mar_mean = ?,
                pitch_mean = ?, yaw_mean = ?,
                updated_at = datetime('now')
        """, (
            operator_id, ear_mean, ear_std, mar_mean, pitch_mean, yaw_mean,
            ear_mean, ear_std, mar_mean, pitch_mean, yaw_mean,
        ))
        self._conn.commit()
    
    def save_session(
        self,
        operator_id: str,
        session_summary: Dict,
        started_at: str,
    ) -> None:
        """Persiste o resumo de uma sessão encerrada."""
        cur = self._conn.cursor()
        cur.execute("""
            INSERT INTO session_logs
                (operator_id, started_at, duration_min, total_windows,
                 danger_ratio, avg_prob, max_consec_danger,
                 total_microsleeps, total_microsleep_ms, perclos_trend,
                 calibrations, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            operator_id,
            started_at,
            session_summary.get("duration_min", 0),
            session_summary.get("total_windows", 0),
            session_summary.get("danger_ratio", 0),
            session_summary.get("avg_prob", 0),
            session_summary.get("max_consec_danger", 0),
            session_summary.get("total_microsleeps", 0),
            session_summary.get("total_microsleep_ms", 0),
            session_summary.get("perclos_trend", 0),
            session_summary.get("calibrations", 0),
            json.dumps(session_summary),
        ))
        # Atualizar contadores no perfil
        cur.execute("""
            UPDATE operator_profiles SET
                sessions_count = sessions_count + 1,
                total_danger_ratio = (
                    total_danger_ratio * (sessions_count - 1) + ?
                ) / sessions_count,
                avg_session_minutes = (
                    avg_session_minutes * (sessions_count - 1) + ?
                ) / sessions_count,
                last_session_end = datetime('now'),
                updated_at = datetime('now')
            WHERE operator_id = ?
        """, (
            session_summary.get("danger_ratio", 0),
            session_summary.get("duration_min", 0),
            operator_id,
        ))
        self._conn.commit()
    
    def get_recent_sessions(
        self, operator_id: str, limit: int = 10
    ) -> List[Dict]:
        """Retorna as últimas N sessões do operador."""
        cur = self._conn.cursor()
        cur.execute("""
            SELECT * FROM session_logs
            WHERE operator_id = ?
            ORDER BY ended_at DESC LIMIT ?
        """, (operator_id, limit))
        return [dict(row) for row in cur.fetchall()]
    
    def get_warm_start_baseline(
        self, operator_id: str
    ) -> Optional[Dict]:
        """
        Retorna baseline suavizado do operador para warm-start.
        
        Se o operador já tem perfil com pelo menos 2 sessões,
        retorna os valores históricos como ponto de partida.
        Isso permite reduzir warm-up de 120s para 30s.
        """
        profile = self.get_profile(operator_id)
        if profile is None:
            return None
        if profile.get("sessions_count", 0) < 2:
            return None
        return {
            "ear_mean": profile["ear_mean"],
            "ear_std": profile["ear_std"],
            "mar_mean": profile["mar_mean"],
            "pitch_mean": profile["pitch_mean"],
            "yaw_mean": profile["yaw_mean"],
        }
    
    def close(self) -> None:
        self._conn.close()
```

**Integração no `run_realtime_demo.py`:**

```python
# Na inicialização:
from .memory import OperatorStore
op_store = OperatorStore(db_path=str(model_dir / "operator_memory.db"))

# Checar warm-start:
warm_baseline = op_store.get_warm_start_baseline(session.operator_id)
if warm_baseline is not None:
    print(f"[memory] Warm-start disponível para {session.operator_id}")
    print(f"[memory]   EAR histórico: {warm_baseline['ear_mean']:.4f}")
    # Reduzir warm-up
    warmup_sec = 30.0
    calibrator = RTSubjectCalibrator(
        CalibrationConfig(fps=fps, search_sec=warmup_sec)
    )

# Após calibração:
op_store.upsert_profile_from_calibration(session.operator_id, b)

# No finally (encerramento):
from datetime import datetime
op_store.save_session(
    session.operator_id,
    session.summary(),
    started_at=datetime.now().isoformat(),
)
op_store.close()
print(f"[memory] Sessão salva para {session.operator_id}")
```

**Critério de aceite M2:**

- Arquivo `operator_memory.db` criado automaticamente na primeira execução
- Após 2 sessões, `get_warm_start_baseline()` retorna valores
- Warm-start reduz `warmup_sec` de 120 para 30 quando perfil existe
- `get_recent_sessions()` retorna lista ordenada corretamente
- `total_danger_ratio` no perfil é a média ponderada entre todas as sessões

---

### Fase M3 — Feature Logger para Retraining (Semana 3)

**Objetivo:** Gravar features + labels + metadados de cada janela para uso futuro em retraining do modelo. Isso fecha o ciclo de aprendizado contínuo (data flywheel).

**Classe no `memory.py`:**

```python
import csv
from pathlib import Path


class FeatureLogger:
    """
    M3: Gravação de features para retraining futuro (data flywheel).
    
    Grava um CSV com 19 features + label + prob + metadados por janela.
    O CSV fica no mesmo diretório dos modelos.
    Opt-in: desativado por padrão (--log-features para ativar).
    
    Formato: um arquivo por sessão, nomeado com timestamp.
    Não grava frames brutos (privacidade — apenas features numéricas).
    """
    
    def __init__(
        self,
        output_dir: str,
        feature_names: List[str],
        operator_id: str = "unknown",
        enabled: bool = False,
    ) -> None:
        self._enabled = enabled
        self._feature_names = feature_names
        self._operator_id = operator_id
        self._file = None
        self._writer = None
        self._count = 0
        
        if not enabled:
            return
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        filepath = out_path / f"features_{operator_id}_{ts}.csv"
        
        self._file = open(filepath, "w", newline="", encoding="utf-8")
        header = (
            ["timestamp_ms", "operator_id", "label", "prob_danger",
             "alert_level", "confidence"]
            + feature_names
        )
        self._writer = csv.writer(self._file)
        self._writer.writerow(header)
        print(f"[feature_log] Logging to {filepath}")
    
    def log(
        self,
        timestamp_ms: float,
        label: str,
        prob_danger: float,
        alert_level: str,
        confidence: str,
        window_feats: Dict[str, float],
    ) -> None:
        """Grava uma linha no CSV."""
        if not self._enabled or self._writer is None:
            return
        
        row = [
            f"{timestamp_ms:.1f}",
            self._operator_id,
            label,
            f"{prob_danger:.6f}",
            alert_level,
            confidence,
        ]
        for name in self._feature_names:
            row.append(f"{window_feats.get(name, 0.0):.6f}")
        
        self._writer.writerow(row)
        self._count += 1
        
        # Flush a cada 100 linhas para não perder dados em crash
        if self._count % 100 == 0:
            self._file.flush()
    
    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            print(f"[feature_log] {self._count} janelas gravadas")
```

**Integração:**

```python
# CLI: adicionar --log-features flag
parser.add_argument("--log-features", action="store_true",
                    help="Gravar features de cada janela em CSV para retraining")

# Inicialização:
feat_logger = FeatureLogger(
    output_dir=str(model_dir / "logs"),
    feature_names=feature_names,
    operator_id=session.operator_id,
    enabled=args.log_features,
)

# Após cada inferência:
feat_logger.log(
    timestamp_ms=output.timestamp_ms,
    label=output.label,
    prob_danger=output.prob_danger,
    alert_level=output.alert_level.name,
    confidence=output.confidence,
    window_feats=window_feats,
)

# No finally:
feat_logger.close()
```

**Critério de aceite M3:**

- Com `--log-features`, CSV criado em `MODELS/logs/`
- Header tem 19 features + 6 colunas de metadados
- Sem `--log-features`, nenhum arquivo é criado (zero overhead)
- Flush a cada 100 linhas — dados sobrevivem a crash

---

## PARTE B — PARALLELIZATION

O pipeline atual é 100% serial: captura → extrai → calibra → agrega → infere → display, tudo na mesma thread. No Raspberry Pi 5 com FaceMesh ONNX, cada frame leva ~30-50ms. Se a inferência da janela demorar mais que 33ms (1/30fps), frames são dropados. Parallelization resolve isso separando captura, processamento e display em threads independentes.

### Fase P1 — Captura Paralela: FrameGrabber (Semana 4)

**Objetivo:** Separar a captura de câmera em uma thread dedicada com fila thread-safe. A thread principal consome frames da fila sem bloquear.

**Arquivo novo:** `SALTE_INFERENCE/parallel.py`

```python
"""
Parallelization para DeteccaoFadiga.

Componentes:
  P1. FrameGrabber       — captura de câmera em thread dedicada
  P2. PipelineWorker     — processamento em thread separada
  P3. PerformanceMonitor — métricas de FPS, latência e drops
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from queue import Queue, Empty, Full
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameGrabberConfig:
    """Configuração do capturador paralelo."""
    queue_size: int = 2          # Fila pequena: sempre pegar o frame mais recente
    target_fps: int = 30
    drop_old_frames: bool = True # Se fila cheia, descarta o mais antigo


class FrameGrabber:
    """
    P1: Captura de câmera em thread dedicada.
    
    A thread de captura roda em loop contínuo e coloca frames numa Queue.
    O consumidor (pipeline) pega frames sem bloquear a câmera.
    
    Se a fila está cheia e drop_old_frames=True, descarta o frame mais
    antigo para garantir que o consumidor sempre processe dados recentes.
    
    Benefícios:
    - Câmera nunca para de capturar (evita timeout de driver)
    - Pipeline pode demorar mais que 1/FPS sem dropar frames
    - Frame mais recente sempre disponível (baixa latência end-to-end)
    """
    
    def __init__(
        self,
        capture: Any,  # cv2.VideoCapture ou PiCamera2Capture
        config: Optional[FrameGrabberConfig] = None,
    ) -> None:
        self.cfg = config or FrameGrabberConfig()
        self._capture = capture
        self._queue: Queue = Queue(maxsize=self.cfg.queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frames_grabbed: int = 0
        self._frames_dropped: int = 0
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Inicia a thread de captura."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="FrameGrabber",
        )
        self._thread.start()
    
    def stop(self) -> None:
        """Para a thread de captura e libera recursos."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Retorna o frame mais recente, ou None se não há frame disponível.
        Não bloqueia por mais que `timeout` segundos.
        """
        try:
            frame = self._queue.get(timeout=timeout)
            return frame
        except Empty:
            return None
    
    def _capture_loop(self) -> None:
        """Loop de captura na thread dedicada."""
        interval = 1.0 / max(self.cfg.target_fps, 1)
        
        while self._running:
            ret, frame = self._capture.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            with self._lock:
                self._frames_grabbed += 1
            
            if self._queue.full() and self.cfg.drop_old_frames:
                try:
                    self._queue.get_nowait()  # descarta o mais antigo
                    with self._lock:
                        self._frames_dropped += 1
                except Empty:
                    pass
            
            try:
                self._queue.put_nowait(frame)
            except Full:
                with self._lock:
                    self._frames_dropped += 1
    
    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "frames_grabbed": self._frames_grabbed,
                "frames_dropped": self._frames_dropped,
                "queue_size": self._queue.qsize(),
            }
```

**Integração no `run_realtime_demo.py`:**

```python
# ANTES (serial):
#   while True:
#       ret, frame = cap.read()
#       ...

# DEPOIS (paralelo):
from .parallel import FrameGrabber, FrameGrabberConfig

grabber = FrameGrabber(cap, FrameGrabberConfig(target_fps=fps))
grabber.start()

try:
    while True:
        frame = grabber.get_frame(timeout=0.1)
        if frame is None:
            continue
        
        feats = extractor.process_frame(frame)
        # ... resto do pipeline
finally:
    grabber.stop()
    cap.release()
```

**Critério de aceite P1:**

- Thread de captura inicia e para sem deadlock
- Com processamento lento (sleep 50ms), frames antigos são descartados
- `stats["frames_dropped"]` reflete frames perdidos com precisão
- Sem `FrameGrabber`, comportamento idêntico ao original (fallback)

---

### Fase P2 — Pipeline Worker: Processamento Separado (Semana 4)

**Objetivo:** Mover o processamento pesado (FeatureExtractor + Calibrator + WindowFactory + Model) para uma thread worker, liberando a thread principal para captura + display.

**Classe no `parallel.py`:**

```python
@dataclass
class PipelineResult:
    """Resultado do processamento de um frame pelo worker."""
    frame: np.ndarray                  # frame original (para overlay)
    raw_text: str                      # EAR/MAR/Face para display
    status_text: str                   # label + prob + confidence
    color: Tuple[int, int, int]        # cor do overlay
    overlay2: str                      # PERCLOS + blinks + alert
    output: Optional[Any] = None       # FatigueOutput (se disponível)
    window_feats: Optional[Dict] = None
    is_warmup: bool = False
    warmup_progress: float = 0.0
    warmup_elapsed: float = 0.0


class PipelineWorker:
    """
    P2: Executa o pipeline de inferência em thread separada.
    
    Arquitetura:
      Thread Principal (Main):  FrameGrabber → display/overlay + key input
      Thread Worker:            frame_queue → Feature → Calibrate → Window → Model → result_queue
    
    A main thread alimenta frames via put_frame() e consome
    resultados via get_result(). O worker processa na velocidade
    que conseguir — se for mais lento que o FPS da câmera,
    frames intermediários são descartados pelo FrameGrabber.
    """
    
    def __init__(
        self,
        process_fn: Callable[[np.ndarray], PipelineResult],
        queue_size: int = 2,
    ) -> None:
        self._process_fn = process_fn
        self._input_queue: Queue = Queue(maxsize=queue_size)
        self._output_queue: Queue = Queue(maxsize=queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frames_processed: int = 0
        self._total_process_time_ms: float = 0.0
        self._lock = threading.Lock()
    
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="PipelineWorker",
        )
        self._thread.start()
    
    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def put_frame(self, frame: np.ndarray) -> bool:
        """Envia frame para processamento. Retorna False se fila cheia."""
        if self._input_queue.full():
            try:
                self._input_queue.get_nowait()  # descarta antigo
            except Empty:
                pass
        try:
            self._input_queue.put_nowait(frame)
            return True
        except Full:
            return False
    
    def get_result(self, timeout: float = 0.001) -> Optional[PipelineResult]:
        """Retorna resultado processado ou None."""
        try:
            return self._output_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def _worker_loop(self) -> None:
        while self._running:
            try:
                frame = self._input_queue.get(timeout=0.1)
            except Empty:
                continue
            
            t0 = time.monotonic()
            result = self._process_fn(frame)
            dt = (time.monotonic() - t0) * 1000
            
            with self._lock:
                self._frames_processed += 1
                self._total_process_time_ms += dt
            
            # Colocar resultado na output queue
            if self._output_queue.full():
                try:
                    self._output_queue.get_nowait()
                except Empty:
                    pass
            try:
                self._output_queue.put_nowait(result)
            except Full:
                pass
    
    @property
    def avg_process_time_ms(self) -> float:
        with self._lock:
            if self._frames_processed == 0:
                return 0.0
            return self._total_process_time_ms / self._frames_processed
    
    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "frames_processed": self._frames_processed,
                "avg_process_ms": round(self.avg_process_time_ms, 1),
                "input_queue": self._input_queue.qsize(),
                "output_queue": self._output_queue.qsize(),
            }
```

**Refatoração do loop principal:**

O loop em `run_realtime_demo.py` seria refatorado para separar a lógica de processamento de frame numa função `_process_single_frame()` que o `PipelineWorker` chama. A thread principal fica responsável apenas por: pegar frame do `FrameGrabber`, enviar ao `PipelineWorker`, pegar resultado, fazer overlay e display.

```python
# Loop principal refatorado:
grabber = FrameGrabber(cap)
grabber.start()

worker = PipelineWorker(process_fn=process_single_frame)
worker.start()

try:
    while True:
        # 1. Pegar frame da câmera
        frame = grabber.get_frame(timeout=0.1)
        if frame is not None:
            worker.put_frame(frame)
        
        # 2. Pegar resultado processado
        result = worker.get_result()
        if result is not None:
            if not headless:
                # Overlay e display na thread principal
                _draw_overlay(result)
                cv2.imshow("SALTE", result.frame)
        
        # 3. Inputs de teclado
        if not headless:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        else:
            time.sleep(frame_interval)
finally:
    worker.stop()
    grabber.stop()
    cap.release()
```

**Critério de aceite P2:**

- Pipeline roda em thread separada sem race conditions
- Se o processamento demora 60ms (abaixo de 30fps), o display continua fluído mostrando o último resultado disponível
- `stats["avg_process_ms"]` reporta tempo médio de processamento
- Encerramento limpo: `stop()` não causa deadlock ou exceções

---

### Fase P3 — Performance Monitor (Semana 5)

**Objetivo:** Métricas de performance em tempo real para validar o impacto da paralelização e diagnosticar gargalos.

**Classe no `parallel.py`:**

```python
class PerformanceMonitor:
    """
    P3: Métricas de performance do pipeline paralelo.
    
    Rastreia:
    - FPS real de captura, processamento e display
    - Latência end-to-end (captura → resultado disponível)
    - Taxa de frame drop
    - Utilização de fila (indicador de backpressure)
    
    Imprime relatório periódico no console.
    """
    
    def __init__(self, report_interval_sec: float = 30.0) -> None:
        self._report_interval = report_interval_sec
        self._last_report_time = time.monotonic()
        
        # FPS tracking
        self._capture_timestamps: Deque[float] = deque(maxlen=100)
        self._process_timestamps: Deque[float] = deque(maxlen=100)
        self._display_timestamps: Deque[float] = deque(maxlen=100)
        
        # Latência
        self._latencies_ms: Deque[float] = deque(maxlen=100)
    
    def on_capture(self) -> None:
        self._capture_timestamps.append(time.monotonic())
    
    def on_process_complete(self, latency_ms: float) -> None:
        self._process_timestamps.append(time.monotonic())
        self._latencies_ms.append(latency_ms)
    
    def on_display(self) -> None:
        self._display_timestamps.append(time.monotonic())
    
    def _compute_fps(self, timestamps: Deque[float]) -> float:
        if len(timestamps) < 2:
            return 0.0
        dt = timestamps[-1] - timestamps[0]
        if dt < 1e-6:
            return 0.0
        return (len(timestamps) - 1) / dt
    
    def maybe_report(
        self,
        grabber_stats: Dict,
        worker_stats: Dict,
    ) -> Optional[str]:
        """Gera relatório se o intervalo foi atingido."""
        now = time.monotonic()
        if now - self._last_report_time < self._report_interval:
            return None
        
        self._last_report_time = now
        
        cap_fps = self._compute_fps(self._capture_timestamps)
        proc_fps = self._compute_fps(self._process_timestamps)
        disp_fps = self._compute_fps(self._display_timestamps)
        
        avg_lat = (
            float(np.mean(self._latencies_ms))
            if self._latencies_ms else 0.0
        )
        p95_lat = (
            float(np.percentile(list(self._latencies_ms), 95))
            if len(self._latencies_ms) >= 5 else 0.0
        )
        
        total_grabbed = grabber_stats.get("frames_grabbed", 0)
        total_dropped = grabber_stats.get("frames_dropped", 0)
        drop_rate = (
            total_dropped / max(total_grabbed, 1) * 100
        )
        
        report = (
            f"[perf] FPS: capture={cap_fps:.1f} "
            f"process={proc_fps:.1f} display={disp_fps:.1f} | "
            f"Latency: avg={avg_lat:.0f}ms p95={p95_lat:.0f}ms | "
            f"Drops: {total_dropped}/{total_grabbed} ({drop_rate:.1f}%) | "
            f"Worker: {worker_stats.get('avg_process_ms', 0):.0f}ms/frame"
        )
        print(report)
        return report
```

**Integração:**

```python
# Inicialização:
perf = PerformanceMonitor(report_interval_sec=30.0)

# No loop principal:
frame = grabber.get_frame()
if frame is not None:
    perf.on_capture()
    worker.put_frame(frame)

result = worker.get_result()
if result is not None:
    perf.on_process_complete(latency_ms=...)
    if not headless:
        perf.on_display()

# A cada iteração:
perf.maybe_report(grabber.stats, worker.stats)
```

**Critério de aceite P3:**

- Relatório impresso a cada 30s com FPS, latência e drop rate
- `p95_lat` calculado corretamente (verificar com sleep artificial)
- No RPi5, report mostra se processamento está abaixo de 30fps (gargalo)
- No PC com webcam, drop rate próximo de 0%

---

## Cronograma Consolidado

```
Semana 1  ┃ M1: SessionMemory                    ┃ Integração no loop principal
          ┃ Estado unificado da sessão em RAM     ┃ summary() ao encerrar
          ┃                                       ┃
Semana 2  ┃ M2: OperatorStore (SQLite)            ┃ Warm-start de calibração
          ┃ Perfis persistentes entre sessões     ┃ Histórico de sessões
          ┃                                       ┃
Semana 3  ┃ M3: FeatureLogger                     ┃ Testes unitários M1-M3
          ┃ CSV de features para retraining       ┃ Testes de integração
          ┃                                       ┃
Semana 4  ┃ P1: FrameGrabber                      ┃ P2: PipelineWorker
          ┃ Captura em thread dedicada            ┃ Processamento em thread worker
          ┃                                       ┃
Semana 5  ┃ P3: PerformanceMonitor                ┃ Testes em RPi5 + PC
          ┃ Métricas de FPS/latência/drops        ┃ Validação end-to-end
```

---

## Estrutura Final de Arquivos

```
SALTE_INFERENCE/
├── __init__.py                    (atualizar exports)
├── feature_extractor_rt.py        (sem alterações)
├── subject_calibrator_rt.py       (sem alterações)
├── window_factory_rt.py           (sem alterações)
├── model_loader.py                (sem alterações)
├── guardrails.py                  (sem alterações — fase anterior)
├── reflection.py                  (sem alterações — fase anterior)
├── memory.py                      ★ NOVO — M1/M2/M3
├── parallel.py                    ★ NOVO — P1/P2/P3
├── run_realtime_demo.py           (modificado — integrar memory + parallel)
├── offline_eval.py                (sem alterações)
└── tests/
    ├── __init__.py
    ├── test_guardrails.py         (sem alterações — 24 testes)
    ├── test_reflection.py         (sem alterações — 18 testes)
    ├── test_memory.py             ★ NOVO — ~20 testes
    └── test_parallel.py           ★ NOVO — ~15 testes

MODELS/
├── operator_memory.db             ★ NOVO (criado automaticamente pelo OperatorStore)
└── logs/                          ★ NOVO (criado pelo FeatureLogger quando --log-features)
    └── features_op-0_20260311_143200.csv
```

---

## Interação entre Memory e os Padrões Anteriores

A memória integra-se com Guardrails e Reflection criando um ciclo completo:

```
┌────────────────────────────────────────────────────────────────┐
│                    SessionMemory (M1)                          │
│  Acumula: windows, danger_ratio, perclos_trend, microsleeps   │
│                                                                │
│  Alimenta:                                                     │
│  ├─ BehaviorGuardRails (G3): consecutive_danger tracking       │
│  ├─ PredictionReflector (R2): prob_history para detecção       │
│  ├─ PerformanceMonitor (P3): contagem de frames/windows        │
│  └─ Overlay: session duration + alert count no display         │
└────────────────────────┬───────────────────────────────────────┘
                         │ ao encerrar
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    OperatorStore (M2)                           │
│  Persiste: perfil do operador, histórico de sessões            │
│                                                                │
│  Alimenta:                                                     │
│  ├─ Warm-start: reduz warm-up de 120s → 30s                   │
│  ├─ validate_calibration (G4): compara com histórico           │
│  └─ Dashboard futuro: relatórios de risco por operador         │
└────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                    FeatureLogger (M3)                           │
│  CSV com features + labels → retraining futuro                 │
│  Fecha o data flywheel: deploy → coleta → treino → deploy     │
└────────────────────────────────────────────────────────────────┘
```

---

## Métricas de Sucesso

| Métrica | Antes | Depois | Como medir |
|---------|-------|--------|------------|
| Tempo de warm-up (operador conhecido) | 120s sempre | 30s com warm-start | Medir tempo até calibração em 2ª sessão |
| Estado ao encerrar | Perdido | Resumo completo via `summary()` | Verificar log no `finally` |
| Histórico entre sessões | Inexistente | SQLite com perfis e sessões | `get_recent_sessions()` retorna dados |
| Frame drop rate (RPi5) | ~15-20% (estimado) | < 5% com paralelização | `PerformanceMonitor.maybe_report()` |
| Latência end-to-end | ~50ms (serial) | ~35ms (paralelo) | `p95_lat` no PerformanceMonitor |
| FPS de display | Acoplado ao processamento | Desacoplado, constante | `disp_fps` no report |
| Data para retraining | Nenhum | CSV por sessão opt-in | Contar linhas no CSV após sessão |

---

## Decisões Arquiteturais e Justificativas

**Por que SQLite e não banco vetorial para M2?** O DeteccaoFadiga lida com perfis estruturados (ear_mean, danger_ratio, etc.), não com embeddings semânticos. SQLite é zero-config, roda no RPi5 sem servidor, e o volume de dados é baixo (dezenas de operadores, centenas de sessões). Um banco vetorial seria over-engineering neste contexto.

**Por que `threading` e não `multiprocessing` para Parallelization?** O gargalo principal é I/O (câmera, display) não CPU-bound. Threading elimina o overhead de serialização entre processos e compartilha memória (o frame numpy não precisa ser copiado). O GIL do Python não é problema porque OpenCV e ONNX Runtime liberam o GIL durante suas operações C++.

**Por que `Queue(maxsize=2)` e não unbounded?** Fila unbounded acumula frames quando o processamento é lento, gerando latência crescente. Com maxsize=2 e `drop_old_frames=True`, o sistema sempre processa dados recentes, sacrificando completude por baixa latência — trade-off correto para detecção de fadiga em tempo real.

**Por que FeatureLogger é opt-in?** Gravar a cada janela (~15s) gera ~5.7K linhas/dia para operação contínua. No RPi5 com SD card, escrita frequente pode degradar o cartão. O flag `--log-features` permite ativar apenas quando necessário (ex: sessões de coleta de dados para retraining).
