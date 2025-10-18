# Lote Predictions Reporting System

This document describes the new lote predictions reporting system that tracks and analyzes medicine label detection results.

## Overview

The system automatically logs every lote prediction made by the OCR system, including:
- Predicted lot numbers and expiry dates
- Match status against the database
- Detection confidence and performance metrics
- File paths to saved artifacts

## Database Schema

### LotePrediction Table
- `id`: Primary key
- `predicted_lot`: OCR extracted lot number
- `predicted_expiry`: OCR extracted expiry date
- `is_match`: Boolean indicating if lot was found in database
- `match_status`: Text status ('Match', 'No Match', 'No OCR Result', 'Search Error')
- `detection_confidence`: YOLO detection confidence score
- `detection_box`: JSON string of bounding box coordinates
- `class_id`: YOLO class ID
- `processing_time_ms`: Total processing time in milliseconds
- `ocr_time_ms`: OCR processing time in milliseconds
- `lote_search_time_ms`: Database search time in milliseconds
- `full_frame_path`: Path to full frame image
- `crop_path`: Path to cropped ROI image
- `final_path`: Path to final annotated image
- `raw_ocr_text`: Raw OCR output text
- `created_at`: Timestamp of prediction
- `prediction_date`: Date in YYYY-MM-DD format for easy querying

## API Endpoints

### 1. Daily Statistics
**GET** `/reports/daily-stats?date=YYYY-MM-DD`

Returns daily statistics for a specific date:
```json
{
  "date": "2025-10-18",
  "total_predictions": 100,
  "matches": 85,
  "no_matches": 15,
  "match_percentage": 85.0,
  "no_match_percentage": 15.0,
  "avg_processing_time_ms": 1250.5,
  "avg_confidence": 0.92
}
```

### 2. Predictions List
**GET** `/reports/predictions?date=YYYY-MM-DD&limit=100&offset=0`

Returns paginated list of predictions for a specific date:
```json
{
  "date": "2025-10-18",
  "predictions": [
    {
      "id": 1,
      "predicted_lot": "ABC123",
      "predicted_expiry": "2025-12-31",
      "is_match": true,
      "match_status": "Match",
      "detection_confidence": 0.95,
      "processing_time_ms": 1200.0,
      "created_at": "2025-10-18T10:30:45.123456",
      "full_frame_path": "/path/to/full.jpg",
      "crop_path": "/path/to/crop.jpg",
      "final_path": "/path/to/final.jpg"
    }
  ],
  "limit": 100,
  "offset": 0,
  "count": 1
}
```

### 3. Date Range Statistics
**GET** `/reports/date-range?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`

Returns statistics for a date range:
```json
{
  "start_date": "2025-10-15",
  "end_date": "2025-10-18",
  "daily_stats": [
    {
      "date": "2025-10-15",
      "total_predictions": 95,
      "matches": 80,
      "no_matches": 15,
      "match_percentage": 84.21
    },
    {
      "date": "2025-10-16",
      "total_predictions": 102,
      "matches": 88,
      "no_matches": 14,
      "match_percentage": 86.27
    }
  ]
}
```

### 4. CSV Export
**GET** `/reports/export-csv?date=YYYY-MM-DD`

Downloads a CSV file containing all predictions for the specified date. The CSV includes:
- ID, Date, Time
- Predicted Lot, Predicted Expiry
- Match Status, Is Match
- Detection Confidence
- Processing Time (ms), OCR Time (ms), Search Time (ms)
- File paths

### 5. Summary Dashboard
**GET** `/reports/summary`

Returns summary statistics for today, yesterday, and the last 7 days:
```json
{
  "today": {
    "date": "2025-10-18",
    "total_predictions": 50,
    "matches": 42,
    "match_percentage": 84.0
  },
  "yesterday": {
    "date": "2025-10-17",
    "total_predictions": 95,
    "matches": 80,
    "match_percentage": 84.21
  },
  "last_7_days": [
    {
      "date": "2025-10-18",
      "total_predictions": 50,
      "matches": 42,
      "match_percentage": 84.0
    }
  ]
}
```

## Integration

The system is automatically integrated into the existing `/process` endpoint. Every time a frame is processed:

1. OCR extracts lot number and expiry date
2. System checks if lot exists in the database
3. All results are automatically saved to the predictions database
4. No changes needed to existing workflow

## Frontend Integration

For your Next.js frontend, you can:

1. **Dashboard Page**: Call `/reports/summary` to show today's stats
2. **Daily Reports**: Call `/reports/daily-stats` with date picker
3. **Detailed View**: Call `/reports/predictions` to show individual predictions
4. **Export Feature**: Link to `/reports/export-csv` for CSV downloads
5. **Charts**: Use `/reports/date-range` for trend charts

## Example Frontend Usage

```javascript
// Get today's stats
const response = await fetch('/reports/summary');
const data = await response.json();
console.log(`Today: ${data.today.match_percentage}% success rate`);

// Export CSV for a specific date
const csvUrl = `/reports/export-csv?date=2025-10-18`;
window.open(csvUrl, '_blank'); // Downloads CSV file

// Get predictions for a date with pagination
const predictions = await fetch('/reports/predictions?date=2025-10-18&limit=50&offset=0');
const predData = await predictions.json();
// Display predData.predictions in a table
```

## Database Location

The database is stored as SQLite at: `/mnt/c/code/AI_Vision/multiprocessing/lote_predictions.db`

## Testing

Run the test script to verify functionality:
```bash
cd /mnt/c/code/AI_Vision/multiprocessing/med_service
python test_db.py
```

This will create sample data and test all database operations.
