from fastapi import FastAPI
from db.database import Base, engine
from db.models import Event
from core.models import Bundle
from core.config import (
    EGOMOTION_MODEL_PATH, EGOMOTION_SCALER_PATH, EGOMOTION_ENCODER_PATH,
    LIDAR_RADAR_MODEL_PATH, LIDAR_RADAR_SCALER_PATH, LIDAR_RADAR_ENCODER_PATH
)
from core.logging_config import log
from routes import prediction, events, health

Base.metadata.create_all(engine)

app = FastAPI(
    title="Unified AV Hacking Detection Agent",
    version="2.0",
    description="Single /predict endpoint for egomotion LSTM and LiDAR/Radar CNN, with combined decisions and event logging."
)

@app.on_event("startup")
def load_everything():
    prediction.EGO = Bundle(EGOMOTION_MODEL_PATH, EGOMOTION_SCALER_PATH, EGOMOTION_ENCODER_PATH)
    prediction.LR  = Bundle(LIDAR_RADAR_MODEL_PATH, LIDAR_RADAR_SCALER_PATH, LIDAR_RADAR_ENCODER_PATH)
    log.info("Models loaded. Ego input: %s | LR input: %s", prediction.EGO.input_shape, prediction.LR.input_shape)

app.include_router(prediction.router)
app.include_router(events.router)
app.include_router(health.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
