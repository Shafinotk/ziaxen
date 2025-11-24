import os
import json
import time
import hashlib
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# --- DB (SQLite via SQLAlchemy) ---
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

# ================== CONFIG ==================
ART_DIR = os.getenv("ARTIFACT_DIR", "./artifacts")
DB_PATH = os.getenv("DB_PATH", "./db/events.db")
CSV_MIRROR = os.getenv("CSV_MIRROR", "false").lower() == "true"
CSV_PATH = os.getenv("CSV_PATH", "./data/events.csv")

EGOMOTION_MODEL_PATH = os.getenv("EGOMOTION_MODEL", os.path.join(ART_DIR, "egomotion_model.h5"))
EGOMOTION_SCALER_PATH = os.getenv("EGOMOTION_SCALER", os.path.join(ART_DIR, "egomotion_scaler.pkl"))
EGOMOTION_ENCODER_PATH = os.getenv("EGOMOTION_ENCODER", os.path.join(ART_DIR, "egomotion_label_encoder.pkl"))  # optional

LIDAR_RADAR_MODEL_PATH = os.getenv("LIDAR_RADAR_MODEL", os.path.join(ART_DIR, "lidar_radar_model.h5"))
LIDAR_RADAR_SCALER_PATH = os.getenv("LIDAR_RADAR_SCALER", os.path.join(ART_DIR, "lidar_radar_scaler.pkl"))
LIDAR_RADAR_ENCODER_PATH = os.getenv("LIDAR_RADAR_ENCODER", os.path.join(ART_DIR, "lidar_radar_label_encoder.pkl"))  # optional

# --- Labels (YOUR models) ---
EGO_LABELS = ["Acceleration_Anomaly", "Normal", "Sensor_Spoofing", "Steering_Mismatch"]
LR_LABELS  = ["Normal", "intensity_manipulation", "jamming", "spoofing"]

# --- Human-readable descriptions ---
EGO_DESC = {
    "Acceleration_Anomaly": "Longitudinal acceleration inconsistent with throttle/brake/velocity dynamics.",
    "Normal": "No anomalous ego-motion detected.",
    "Sensor_Spoofing": "IMU/GNSS/pose signals inconsistent with vehicle kinematics (potential spoofing).",
    "Steering_Mismatch": "Steering angle vs yaw/heading response inconsistent (possible EPS command injection/sensor fault)."
}
LR_DESC = {
    "Normal": "No anomaly in LiDAR/Radar signatures.",
    "intensity_manipulation": "Return intensities/power inconsistent with range/geometry; potential fake returns.",
    "jamming": "Wideband noise or saturation typical of jamming.",
    "spoofing": "Coherent false targets/ghosts suggesting sensor spoofing."
}

# --- Threat thresholds (probability of top class) ---
# You can tune in .env: e.g., EGO_LOW=0.50, EGO_HIGH=0.90, LR_LOW=0.50, LR_HIGH=0.90
EGO_LOW  = float(os.getenv("EGO_LOW", 0.50))   # < low -> normal
EGO_HIGH = float(os.getenv("EGO_HIGH", 0.90))  # >= high -> attack
LR_LOW   = float(os.getenv("LR_LOW", 0.50))
LR_HIGH  = float(os.getenv("LR_HIGH", 0.90))

# --- For combined decision (both models) ---
# If either model is "attack", combined is "attack"; else if any "suspicious", combined is "suspicious".
COMBINED_POLICY = os.getenv("COMBINED_POLICY", "max_severity")  # only policy implemented for now

# ================== LOGGING ==================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("av-hack-agent")

# ================== DB SETUP ==================
Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    model = Column(String(32), index=True)             # egomotion | lidar_radar | combined
    label = Column(String(64))
    confidence = Column(Float)
    threat_level = Column(String(16))                  # normal | suspicious | attack
    action = Column(String(256))
    details = Column(Text)                              # JSON blob with probs, inputs hash, etc.

Base.metadata.create_all(engine)

def write_event(model: str, label: str, confidence: float, threat_level: str, action: str, details: Dict[str, Any]):
    """Persist event to SQLite; optionally mirror to CSV."""
    with SessionLocal() as s:
        row = Event(
            model=model,
            label=label,
            confidence=confidence,
            threat_level=threat_level,
            action=action,
            details=json.dumps(details)
        )
        s.add(row)
        s.commit()
    if CSV_MIRROR:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        header_needed = not os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("created_at,model,label,confidence,threat_level,action,details\n")
            f.write(f"{datetime.utcnow().isoformat()},{model},{label},{confidence:.6f},{threat_level},{action},{json.dumps(details)}\n")

# ================== MODELS & UTILS ==================
class Bundle:
    def __init__(self, model_path: str, scaler_path: Optional[str], encoder_path: Optional[str]):
        self.model = load_model(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
        self.encoder = joblib.load(encoder_path) if encoder_path and os.path.exists(encoder_path) else None
        self.input_shape = self.model.input_shape

def softmax_if_needed(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float).squeeze()
    # assume logits if not summing ~1
    s = x.sum()
    if s < 0.999 or s > 1.001:
        e = np.exp(x - np.max(x))
        x = e / np.sum(e)
    return x

def ensure_2d(matrix: List[List[float]]) -> np.ndarray:
    X = np.asarray(matrix, dtype=float)
    if X.ndim != 2:
        raise HTTPException(status_code=400, detail=f"'data' must be a 2D array, got shape {X.shape}")
    return X

def reshape_to_model(X: np.ndarray, input_shape) -> np.ndarray:
    """
    Try to adapt (B,F) or (T,F) to model input. We keep it simple & explicit:
    - If model expects (None, T, F): treat the given rows as timesteps -> shape (1, T, F)
    - If (None, F):                 treat rows as batch -> shape (B, F)
    - If (None, H, W, C):           try (1, T, F, 1) assuming rows=H or T and cols=W or F
    """
    if isinstance(input_shape, list):  # sometimes Keras models have multiple inputs
        input_shape = input_shape[0]
    target = input_shape[1:]  # drop batch
    if len(target) == 1:
        # dense: (None, F)
        if X.ndim == 2:
            return X
        return X.reshape((X.shape[0], -1))
    if len(target) == 2:
        # LSTM/temporal: (None, T, F)
        T, F = target
        # we accept (T,F) as one sequence
        if X.ndim == 2:
            return X[None, :, :]  # (1, T, F)
        return X
    if len(target) == 3:
        # CNN image-like: (None, H, W, C)
        H, W, C = target
        # interpret X rows as H (or T), cols as W (or F)
        H = H or X.shape[0]
        W = W or X.shape[1]
        C = C or 1
        return X.reshape((1, H, W, C))
    # fallback
    return X

def decide_threat(prob: float, label: str, low: float, high: float):
    """
    Map top-probability + label to threat level + action.
    - If top label is a benign class ('Normal'), bias to 'normal' unless confidence is very low.
    - If top label is an attack class, use thresholds.
    """
    benign = label.lower() == "normal"
    if benign:
        if prob >= high:
            return "normal", "Continue normal operation."
        elif prob >= low:
            return "normal", "Continue normal operation; keep monitoring."
        else:
            return "suspicious", "Limit speed; cross-check with redundant sensors."
    else:
        if prob >= high:
            return "attack", "Switch to safe mode: pull over, alert control center, restrict autonomous functions."
        elif prob >= low:
            return "suspicious", "Limit speed; increase following distance; reweight away from affected sensors."
        else:
            return "suspicious", "Limit speed; increase diagnostics; confirm with secondary checks."

# ================== FASTAPI ==================
app = FastAPI(
    title="Unified AV Hacking Detection Agent",
    version="2.0",
    description="Single /predict endpoint for egomotion LSTM and LiDAR/Radar CNN, with combined decisions and event logging."
)

# ------------ Schemas ------------
class EgoReading(BaseModel):
    Throttle_Position: float
    Brake_Pressure: float
    Steering_Angle: float
    Vehicle_Speed: float
    Gear_Position: float
    Rotation_X: float
    Rotation_Y: float
    Rotation_Z: float
    Position_X: float
    Rotation_Q1: float
    Rotation_Q2: float
    Rotation_Q3: float
    Position_Y: float
    Velocity_X: float
    Velocity_Y: float
    Velocity_Z: float
    Position_Z: float
    Bias1: float
    Bias2: float
    Bias3: float
    Bias4: float
    Latitude: float
    Longitude: float
    Speed: float
    Heading: float
    Acceleration_X: float
    Acceleration_Y: float
    Acceleration_Z: float
    Gyroscope_X: float
    Gyroscope_Y: float
    Gyroscope_Z: float
    Yaw_Rate: float

class LRPoint(BaseModel):
    X: float; Y: float; Z: float
    Intensity: float
    Azimuth: float
    Elevation: float
    Range: float
    Speed: float
    RCS: float
    Power: float
    Noise: float

class PredictUnified(BaseModel):
    model: Literal["egomotion","lidar_radar","both"]
    # For egomotion, send either a time sequence (T x 32) or a batch of single-steps (B x 32).
    egomotion_data: Optional[List[List[float]]] = Field(None, description="2D list. If provided and model includes egomotion, rows are timesteps or batch items, cols are features in training order.")
    # For lidar/radar, send a 2D matrix (T x 11) or (H x W) with 11 columns in training order.
    lidar_radar_data: Optional[List[List[float]]] = Field(None, description="2D list. If provided and model includes lidar_radar, rows are points/time, cols are 11 features in training order.")

class PredictOut(BaseModel):
    model_used: str
    per_model: Dict[str, Any]
    combined: Optional[Dict[str, Any]] = None  # only when model="both"

# ------------ Load bundles on startup ------------
EGO = None
LR  = None

@app.on_event("startup")
def load_everything():
    global EGO, LR
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if CSV_MIRROR:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    EGO = Bundle(EGOMOTION_MODEL_PATH, EGOMOTION_SCALER_PATH, EGOMOTION_ENCODER_PATH)
    LR  = Bundle(LIDAR_RADAR_MODEL_PATH, LIDAR_RADAR_SCALER_PATH, LIDAR_RADAR_ENCODER_PATH)
    log.info("Models loaded. Ego input: %s | LR input: %s", EGO.input_shape, LR.input_shape)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "egomotion_loaded": EGO is not None,
        "lidar_radar_loaded": LR is not None,
        "egomotion_labels": EGO_LABELS,
        "lidar_radar_labels": LR_LABELS,
        "egomotion_input_shape": EGO.input_shape if EGO else None,
        "lidar_radar_input_shape": LR.input_shape if LR else None,
        "thresholds": {
            "egomotion": {"low": EGO_LOW, "high": EGO_HIGH},
            "lidar_radar": {"low": LR_LOW, "high": LR_HIGH}
        }
    }
def preprocess_lidar_radar(data: List[List[float]]) -> np.ndarray:
    """Apply the same feature engineering used during training."""
    import pandas as pd

    columns = ['X','Y','Z','Intensity','Azimuth','Elevation','Range',
               'Speed','RCS','Power','Noise']
    df = pd.DataFrame(data, columns=columns)

    # === Feature Engineering ===
    df['Velocity_Magnitude'] = np.sqrt(df['Speed']**2 + df['Azimuth']**2)
    df['Velocity_Range_Ratio'] = df['Speed'] / (df['Range'] + 1e-5)
    df['Log_Intensity'] = np.log1p(df['Intensity'])
    df['Power_Noise_Ratio'] = df['Power'] / (df['Noise'] + 1e-5)

    # === Match training feature order ===
    ordered_features = [
        'X','Y','Z','Intensity','Azimuth','Elevation','Range','Speed',
        'RCS','Power','Noise',
        'Velocity_Magnitude','Velocity_Range_Ratio','Log_Intensity','Power_Noise_Ratio'
    ]
    return df[ordered_features].values

# ------------ Core prediction helpers ------------
def predict_model(bundle: Bundle, X2d: List[List[float]], labels: List[str],
                  desc_map: Dict[str,str], low: float, high: float):
    # --- Special preprocessing for lidar/radar ---
    if labels == LR_LABELS:  
        X = preprocess_lidar_radar(X2d)
    else:
        X = ensure_2d(X2d)

    # Apply scaler if available
    if bundle.scaler is not None:
        X = bundle.scaler.transform(X)

    # === FIX for Egomotion LSTM ===
    if labels == EGO_LABELS:
        # Model expects shape (batch, timesteps, features) = (1, 32, 1)
        X_in = X.reshape((1, X.shape[1], 1))
    elif labels == LR_LABELS:
        # CNN expects (batch, timesteps, features=1)
        X_in = X.reshape((X.shape[0], X.shape[1], 1))
    else:
        # fallback to generic reshaper
        X_in = reshape_to_model(X, bundle.input_shape)

    preds = bundle.model.predict(X_in)
    if isinstance(preds, list):
        preds = preds[0]

    if preds.ndim == 2 and preds.shape[0] > 1:
        preds = preds.mean(axis=0, keepdims=True)

    p = softmax_if_needed(preds[0])
    idx = int(np.argmax(p))
    label = labels[idx] if idx < len(labels) else str(idx)
    prob = float(p[idx])
    threat, action = decide_threat(prob, label, low, high)

    return {
        "label": label,
        "prob": prob,
        "threat_level": threat,
        "action": action,
        "probs": p.tolist(),
        "labels": labels,
        "description": desc_map.get(label, "")
    }


def combine_decision(ego_res: Optional[Dict[str,Any]], lr_res: Optional[Dict[str,Any]]):
    if ego_res is None and lr_res is None:
        return None
    # severity order
    rank = {"normal": 0, "suspicious": 1, "attack": 2}

    parts = [r for r in [ego_res, lr_res] if r is not None]
    # default merge: if any attack -> attack; else if any suspicious -> suspicious; else normal
    worst = max(parts, key=lambda r: rank.get(r["threat_level"], 0))
    combined_level = worst["threat_level"]

    # choose combined label: prefer non-Normal with highest prob; else "Normal"
    candidate = max(parts, key=lambda r: (0 if r["label"].lower()=="normal" else 1, r["prob"]))
    combined_label = candidate["label"]
    combined_prob = candidate["prob"]

    # action: use the strongest action (from worst)
    combined_action = worst["action"]

    return {
        "label": combined_label,
        "prob": combined_prob,
        "threat_level": combined_level,
        "action": combined_action,
        "rationale": "Combined by max severity across models."
    }

# ------------ Unified /predict ------------
@app.post("/predict", response_model=PredictOut)
def predict(req: PredictUnified):
    if req.model == "egomotion":
        if req.egomotion_data is None:
            raise HTTPException(status_code=400, detail="Provide 'egomotion_data' for model='egomotion'.")
        ego_res = predict_model(EGO, req.egomotion_data, EGO_LABELS, EGO_DESC, EGO_LOW, EGO_HIGH)

        # LOG events: log suspicious/attack
        if ego_res["threat_level"] != "normal":
            details = {
                "model": "egomotion",
                "egomotion_input_hash": hashlib.sha1(json.dumps(req.egomotion_data).encode()).hexdigest(),
                "result": ego_res
            }
            write_event("egomotion", ego_res["label"], ego_res["prob"], ego_res["threat_level"], ego_res["action"], details)

        return {"model_used": "egomotion", "per_model": {"egomotion": ego_res}}

    elif req.model == "lidar_radar":
        if req.lidar_radar_data is None:
            raise HTTPException(status_code=400, detail="Provide 'lidar_radar_data' for model='lidar_radar'.")
        lr_res = predict_model(LR, req.lidar_radar_data, LR_LABELS, LR_DESC, LR_LOW, LR_HIGH)

        if lr_res["threat_level"] != "normal":
            details = {
                "model": "lidar_radar",
                "lidar_radar_input_hash": hashlib.sha1(json.dumps(req.lidar_radar_data).encode()).hexdigest(),
                "result": lr_res
            }
            write_event("lidar_radar", lr_res["label"], lr_res["prob"], lr_res["threat_level"], lr_res["action"], details)

        return {"model_used": "lidar_radar", "per_model": {"lidar_radar": lr_res}}

    elif req.model == "both":
        if req.egomotion_data is None or req.lidar_radar_data is None:
            raise HTTPException(status_code=400, detail="Provide BOTH 'egomotion_data' and 'lidar_radar_data' for model='both'.")

        ego_res = predict_model(EGO, req.egomotion_data, EGO_LABELS, EGO_DESC, EGO_LOW, EGO_HIGH)
        lr_res  = predict_model(LR,  req.lidar_radar_data, LR_LABELS,  LR_DESC,  LR_LOW,  LR_HIGH)
        combined = combine_decision(ego_res, lr_res)

        # LOG worst (combined) if not normal
        if combined and combined["threat_level"] != "normal":
            details = {
                "model": "both",
                "egomotion_input_hash": hashlib.sha1(json.dumps(req.egomotion_data).encode()).hexdigest(),
                "lidar_radar_input_hash": hashlib.sha1(json.dumps(req.lidar_radar_data).encode()).hexdigest(),
                "ego_result": ego_res,
                "lr_result": lr_res,
                "combined": combined
            }
            write_event("combined", combined["label"], combined["prob"], combined["threat_level"], combined["action"], details)

        return {"model_used": "both", "per_model": {"egomotion": ego_res, "lidar_radar": lr_res}, "combined": combined}

    else:
        raise HTTPException(status_code=400, detail="model must be 'egomotion' | 'lidar_radar' | 'both'")

# ------------ Event browsing/export ------------
@app.get("/events/recent")
def recent_events(limit: int = Query(50, ge=1, le=500)):
    with SessionLocal() as s:
        rows = s.query(Event).order_by(Event.created_at.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "model": r.model,
                "label": r.label,
                "confidence": r.confidence,
                "threat_level": r.threat_level,
                "action": r.action,
                "details": json.loads(r.details)
            }
            for r in rows
        ]

@app.get("/events/export/csv")
def export_csv():
    if not os.path.exists(CSV_PATH):
        return {"status":"no_csv","path":CSV_PATH}
    # simple link-like info; actual download would be via static server in production
    return {"status":"ok","path":CSV_PATH}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",           # "filename:FastAPI_instance"
        host="127.0.0.1",
        port=8000,
        reload=True
    )
