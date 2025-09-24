import os
from fastapi import APIRouter, Query
from db.crud import get_recent_events
from core.config import CSV_PATH

router = APIRouter()

@router.get("/events/recent")
def recent_events(limit: int = Query(50, ge=1, le=500)):
    return get_recent_events(limit)

@router.get("/events/export/csv")
def export_csv():
    if not os.path.exists(CSV_PATH):
        return {"status":"no_csv","path":CSV_PATH}
    return {"status":"ok","path":CSV_PATH}
