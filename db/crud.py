import os, json
from datetime import datetime
from db.database import SessionLocal
from db.models import Event
from core.config import CSV_MIRROR, CSV_PATH

def write_event(model, label, confidence, threat_level, action, details):
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

def get_recent_events(limit: int):
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
