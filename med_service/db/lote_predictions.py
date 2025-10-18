from sqlalchemy import create_engine, Column, String, Integer, DateTime, Float, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
import os
from datetime import datetime, date
from typing import Optional, List, Dict, Any
import pandas as pd

# Database setup
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DB_PATH = f"sqlite:///{ROOT}/lote_predictions.db"
Base = declarative_base()

class LotePrediction(Base):
    """
    Table to store lote prediction results with match status and metadata
    """
    __tablename__ = "lote_predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Prediction data
    predicted_lot = Column(String, nullable=True)  # OCR extracted lot number
    predicted_expiry = Column(String, nullable=True)  # OCR extracted expiry date
    
    # Match results
    is_match = Column(Boolean, nullable=False)  # True if lot found in database
    match_status = Column(String, nullable=False)  # 'Match', 'No Match', 'No OCR Result', 'Search Error'
    
    # Detection metadata
    detection_confidence = Column(Float, nullable=True)  # YOLO detection confidence
    detection_box = Column(String, nullable=True)  # JSON string of bounding box coordinates
    class_id = Column(Integer, nullable=True)  # YOLO class ID
    
    # Performance metrics
    processing_time_ms = Column(Float, nullable=True)  # Total processing time
    ocr_time_ms = Column(Float, nullable=True)  # OCR processing time
    lote_search_time_ms = Column(Float, nullable=True)  # Database search time
    
    # File paths for artifacts
    full_frame_path = Column(String, nullable=True)
    crop_path = Column(String, nullable=True)
    final_path = Column(String, nullable=True)
    
    # Raw OCR data
    raw_ocr_text = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    prediction_date = Column(String, nullable=False)  # YYYY-MM-DD format for easy querying
    
    def __repr__(self):
        return f"<LotePrediction(id={self.id}, lot='{self.predicted_lot}', match={self.is_match}, date='{self.prediction_date}')>"


class LotePredictionDB:
    """
    Database manager for lote predictions
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    def save_prediction(self, 
                       predicted_lot: Optional[str],
                       predicted_expiry: Optional[str],
                       is_match: bool,
                       match_status: str,
                       detection_confidence: Optional[float] = None,
                       detection_box: Optional[List[int]] = None,
                       class_id: Optional[int] = None,
                       processing_time_ms: Optional[float] = None,
                       ocr_time_ms: Optional[float] = None,
                       lote_search_time_ms: Optional[float] = None,
                       full_frame_path: Optional[str] = None,
                       crop_path: Optional[str] = None,
                       final_path: Optional[str] = None,
                       raw_ocr_text: Optional[str] = None) -> int:
        """
        Save a lote prediction result to the database
        Returns the ID of the created record
        """
        session = self.SessionLocal()
        try:
            prediction = LotePrediction(
                predicted_lot=predicted_lot,
                predicted_expiry=predicted_expiry,
                is_match=is_match,
                match_status=match_status,
                detection_confidence=detection_confidence,
                detection_box=str(detection_box) if detection_box else None,
                class_id=class_id,
                processing_time_ms=processing_time_ms,
                ocr_time_ms=ocr_time_ms,
                lote_search_time_ms=lote_search_time_ms,
                full_frame_path=full_frame_path,
                crop_path=crop_path,
                final_path=final_path,
                raw_ocr_text=raw_ocr_text,
                prediction_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            session.add(prediction)
            session.commit()
            prediction_id = prediction.id
            return prediction_id
        
        except Exception as e:
            session.rollback()
            print(f"Error saving prediction: {e}")
            raise
        finally:
            session.close()
    
    def get_daily_stats(self, target_date: str) -> Dict[str, Any]:
        """
        Get statistics for a specific date (YYYY-MM-DD format)
        """
        session = self.SessionLocal()
        try:
            # Get all predictions for the date
            predictions = session.query(LotePrediction).filter(
                LotePrediction.prediction_date == target_date
            ).all()
            
            total_predictions = len(predictions)
            if total_predictions == 0:
                return {
                    "date": target_date,
                    "total_predictions": 0,
                    "matches": 0,
                    "no_matches": 0,
                    "match_percentage": 0.0,
                    "no_match_percentage": 0.0,
                    "avg_processing_time_ms": 0.0,
                    "avg_confidence": 0.0
                }
            
            matches = sum(1 for p in predictions if p.is_match)
            no_matches = total_predictions - matches
            
            # Calculate averages
            processing_times = [p.processing_time_ms for p in predictions if p.processing_time_ms is not None]
            confidences = [p.detection_confidence for p in predictions if p.detection_confidence is not None]
            
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "date": target_date,
                "total_predictions": total_predictions,
                "matches": matches,
                "no_matches": no_matches,
                "match_percentage": round((matches / total_predictions) * 100, 2),
                "no_match_percentage": round((no_matches / total_predictions) * 100, 2),
                "avg_processing_time_ms": round(avg_processing_time, 2),
                "avg_confidence": round(avg_confidence, 3)
            }
        
        finally:
            session.close()
    
    def get_predictions_by_date(self, target_date: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get all predictions for a specific date with pagination
        """
        session = self.SessionLocal()
        try:
            predictions = session.query(LotePrediction).filter(
                LotePrediction.prediction_date == target_date
            ).order_by(LotePrediction.created_at.desc()).limit(limit).offset(offset).all()
            
            return [
                {
                    "id": p.id,
                    "predicted_lot": p.predicted_lot,
                    "predicted_expiry": p.predicted_expiry,
                    "is_match": p.is_match,
                    "match_status": p.match_status,
                    "detection_confidence": p.detection_confidence,
                    "processing_time_ms": p.processing_time_ms,
                    "created_at": p.created_at.isoformat(),
                    "full_frame_path": p.full_frame_path,
                    "crop_path": p.crop_path,
                    "final_path": p.final_path
                }
                for p in predictions
            ]
        
        finally:
            session.close()
    
    def get_date_range_stats(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get statistics for a date range
        """
        session = self.SessionLocal()
        try:
            predictions = session.query(LotePrediction).filter(
                LotePrediction.prediction_date >= start_date,
                LotePrediction.prediction_date <= end_date
            ).all()
            
            # Group by date
            date_groups = {}
            for p in predictions:
                date = p.prediction_date
                if date not in date_groups:
                    date_groups[date] = []
                date_groups[date].append(p)
            
            # Calculate stats for each date
            stats = []
            for date, preds in date_groups.items():
                total = len(preds)
                matches = sum(1 for p in preds if p.is_match)
                
                stats.append({
                    "date": date,
                    "total_predictions": total,
                    "matches": matches,
                    "no_matches": total - matches,
                    "match_percentage": round((matches / total) * 100, 2) if total > 0 else 0.0
                })
            
            return sorted(stats, key=lambda x: x["date"])
        
        finally:
            session.close()
    
    def export_to_csv(self, target_date: str) -> pd.DataFrame:
        """
        Export predictions for a specific date to a pandas DataFrame (for CSV export)
        """
        session = self.SessionLocal()
        try:
            predictions = session.query(LotePrediction).filter(
                LotePrediction.prediction_date == target_date
            ).order_by(LotePrediction.created_at.desc()).all()
            
            data = []
            for p in predictions:
                data.append({
                    "ID": p.id,
                    "Date": p.prediction_date,
                    "Time": p.created_at.strftime('%H:%M:%S'),
                    "Predicted Lot": p.predicted_lot or "N/A",
                    "Predicted Expiry": p.predicted_expiry or "N/A",
                    "Match Status": p.match_status,
                    "Is Match": "Yes" if p.is_match else "No",
                    "Detection Confidence": p.detection_confidence or 0.0,
                    "Processing Time (ms)": p.processing_time_ms or 0.0,
                    "OCR Time (ms)": p.ocr_time_ms or 0.0,
                    "Search Time (ms)": p.lote_search_time_ms or 0.0,
                    "Full Frame Path": p.full_frame_path or "",
                    "Crop Path": p.crop_path or "",
                    "Final Path": p.final_path or ""
                })
            
            return pd.DataFrame(data)
        
        finally:
            session.close()


# Global instance
lote_prediction_db = LotePredictionDB()
