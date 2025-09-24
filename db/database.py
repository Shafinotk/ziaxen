# db/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from core.config import DB_PATH

# -----------------------------
# SQLAlchemy Base & Engine
# -----------------------------
Base = declarative_base()
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)

# -----------------------------
# Session factory
# -----------------------------
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
