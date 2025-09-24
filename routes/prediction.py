import hashlib, json
from fastapi import APIRouter, HTTPException
from schemas.requests import PredictUnified
from schemas.responses import PredictOut
from services.prediction import predict_model
from core.config import (
    EGO_LABELS, LR_LABELS, EGO_DESC, LR_DESC,
    EGO_LOW, EGO_HIGH, LR_LOW, LR_HIGH
)
from core.threat import combine_decision
from db.crud import write_event

router = APIRouter()

EGO = None
LR = None

@router.post("/predict", response_model=PredictOut)
def predict(req: PredictUnified):
    global EGO, LR
    if req.model == "egomotion":
        if req.egomotion_data is None:
            raise HTTPException(status_code=400, detail="Provide 'egomotion_data'")
        ego_res = predict_model(EGO, req.egomotion_data, EGO_LABELS, EGO_DESC, EGO_LOW, EGO_HIGH)
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
            raise HTTPException(status_code=400, detail="Provide 'lidar_radar_data'")
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
            raise HTTPException(status_code=400, detail="Provide both inputs")
        ego_res = predict_model(EGO, req.egomotion_data, EGO_LABELS, EGO_DESC, EGO_LOW, EGO_HIGH)
        lr_res  = predict_model(LR, req.lidar_radar_data, LR_LABELS, LR_DESC, LR_LOW, LR_HIGH)
        combined = combine_decision(ego_res, lr_res)
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
        raise HTTPException(status_code=400, detail="model must be egomotion | lidar_radar | both")
