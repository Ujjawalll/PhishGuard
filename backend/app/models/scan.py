from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.sql import func
from backend.app.db.session import Base

class Scan(Base):
    __tablename__ = "scans"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    url = Column(String, index=True, nullable=False)
    domain = Column(String, index=True)
    risk_level = Column(String, nullable=False)
    ml_probability = Column(Float)
    rule_score = Column(Float)
    fused_score = Column(Float)
    stage = Column(String)  # fast or deep
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
