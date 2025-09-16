from sqlalchemy import create_engine, Column, String, Integer, DateTime, func
from sqlalchemy.orm import declarative_base, sessionmaker
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DB = f"sqlite:///{ROOT}/oaix.db"
Base = declarative_base()

class WebhookEvent(Base):
    __tablename__ = "webhook_events"

    id = Column(Integer, primary_key=True, autoincrement=True)    
    camera_name = Column(String, nullable=False)
    lot = Column(String, nullable=True)
    expiry = Column(String, nullable=True)
    all_text = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    mime = Column(String, nullable=False)
    image_path = Column(String, nullable=True)  # Store file path instead of base64


class AppLogger(Base):
    __tablename__ = "app_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    log_code = Column(Integer, nullable=False, default=0)
    level = Column(String, nullable=False)
    message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    
# start the database engine
engine = create_engine(_DB, echo=True)
# create the tables
Base.metadata.create_all(engine)
# create the session factory
SessionLocal = sessionmaker(bind=engine)