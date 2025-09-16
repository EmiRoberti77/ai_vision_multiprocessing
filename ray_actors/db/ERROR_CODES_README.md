# Error Codes System

A comprehensive error tracking and logging system for the AI Vision Multiprocessing Framework.

## Overview

This system provides standardized error codes with categories, severity levels, and automated logging to help track and debug issues across the entire application.

## Files

- `error_codes.py` - Main error code definitions and utilities
- `db_logger.py` - Enhanced logger with error code support  
- `error_codes_examples.py` - Usage examples and demonstrations

## Error Code Ranges

| Range | Category | Description |
|-------|----------|-------------|
| 1000-1999 | System/Infrastructure | Core system failures, GPU issues, memory problems |
| 2000-2999 | Video/Stream | RTSP connections, frame processing, video codecs |
| 3000-3999 | YOLO/Detection | Model loading, inference errors, detection timeouts |
| 4000-4999 | OCR Processing | Text extraction, Tesseract issues, parsing errors |
| 5000-5999 | Database | Connection failures, query errors, transaction issues |
| 6000-6999 | Webhook/Network | HTTP requests, JSON serialization, network connectivity |
| 7000-7999 | gRPC/Communication | Server startup, client connections, protobuf issues |
| 8000-8999 | File/Storage | File I/O, disk space, permission errors |
| 9000-9999 | Configuration | Invalid settings, missing parameters, validation |

## Usage Examples

### Basic Error Logging

```python
from app_base import AppBase
from db.error_codes import ErrorCode

app = AppBase()

# Log a stream connection error
app.log_error(
    ErrorCode.STREAM_CONNECTION_FAILED,
    context="channel_1", 
    details="RTSP timeout after 30 seconds"
)

# Log an OCR error
app.log_error(
    ErrorCode.LOT_NUMBER_NOT_FOUND,
    context="channel_2",
    details="OCR confidence: 0.65, no lot pattern matched"
)
```

### In Exception Handlers

```python
try:
    # Stream connection code
    connect_to_stream(rtsp_url)
except ConnectionError as e:
    app.log_error(
        ErrorCode.STREAM_CONNECTION_FAILED,
        context=f"channel_{channel_id}",
        details=str(e)
    )
except TimeoutError as e:
    app.log_error(
        ErrorCode.STREAM_TIMEOUT,
        context=f"channel_{channel_id}",
        details=f"Timeout: {str(e)}"
    )
```

### Video Processor Integration

```python
# In your video processing loop
if not model_loaded:
    self.log_error(
        ErrorCode.MODEL_LOAD_FAILED,
        context=f"GPU_{gpu_id}",
        details=f"Model path: {model_path}"
    )

if ocr_confidence < 0.5:
    self.log_error(
        ErrorCode.TEXT_CONFIDENCE_LOW,
        context=self.name,
        details=f"Confidence: {ocr_confidence}"
    )
```

## Error Information

Each error code includes:

- **Category**: System area (VIDEO, OCR, DATABASE, etc.)
- **Severity**: Impact level (INFO, WARNING, ERROR, CRITICAL)
- **Description**: Human-readable explanation
- **Formatted Message**: Standardized log format

## Severity Levels

- **CRITICAL**: System cannot continue, requires immediate attention
- **ERROR**: Functionality impaired, needs attention  
- **WARNING**: Potential issue, should be monitored
- **INFO**: Informational, normal operation

## Database Storage

Errors are automatically stored in the `app_logs` table with:
- Timestamp
- Severity level
- Formatted error message with code, context, and details
- Structured format for easy searching and analysis

## Utilities

### Get Error Information
```python
from db.error_codes import get_error_info, ErrorCode

info = get_error_info(ErrorCode.STREAM_CONNECTION_FAILED)
print(info['category'])    # ErrorCategory.VIDEO
print(info['severity'])    # ErrorSeverity.ERROR
print(info['description']) # "Failed to connect to video stream"
```

### Format Error Messages
```python
from db.error_codes import format_error_message, ErrorCode

message = format_error_message(
    ErrorCode.STREAM_CONNECTION_FAILED,
    context="channel_1",
    details="RTSP timeout"
)
# Output: [STREAM_CONNECTION_FAILED:2001] Failed to connect to video stream | Context: channel_1 | Details: RTSP timeout
```

## Benefits

1. **Standardized Error Tracking**: Consistent error identification across the system
2. **Categorized Logging**: Easy filtering and analysis by error type
3. **Severity-based Alerting**: Automatic priority assignment based on error severity
4. **Contextual Information**: Rich error details with channel, GPU, or component context
5. **Database Integration**: Persistent storage for error analysis and trending
6. **Easy Debugging**: Structured error format makes troubleshooting easier

## Running Examples

```bash
cd ray_actors
python3 db/error_codes_examples.py
```

This will demonstrate the error code system and log sample errors to your database.
