from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from db.database import Base

class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    model = Column(String(32), index=True)
    label = Column(String(64))
    confidence = Column(Float)
    threat_level = Column(String(16))
    action = Column(String(256))
    details = Column(Text)
