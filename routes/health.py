from fastapi import APIRouter
from core.config import EGO_LABELS, LR_LABELS, EGO_LOW, EGO_HIGH, LR_LOW, LR_HIGH

router = APIRouter()

@router.get("/health")
def health(ego=None, lr=None):
    return {
        "status": "ok",
        "egomotion_loaded": ego is not None,
        "lidar_radar_loaded": lr is not None,
        "egomotion_labels": EGO_LABELS,
        "lidar_radar_labels": LR_LABELS,
        "egomotion_input_shape": ego.input_shape if ego else None,
        "lidar_radar_input_shape": lr.input_shape if lr else None,
        "thresholds": {
            "egomotion": {"low": EGO_LOW, "high": EGO_HIGH},
            "lidar_radar": {"low": LR_LOW, "high": LR_HIGH}
        }
    }
