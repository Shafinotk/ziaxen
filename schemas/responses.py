from typing import Dict, Any, Optional
from pydantic import BaseModel

class PredictOut(BaseModel):
    model_used: str
    per_model: Dict[str, Any]
    combined: Optional[Dict[str, Any]] = None
