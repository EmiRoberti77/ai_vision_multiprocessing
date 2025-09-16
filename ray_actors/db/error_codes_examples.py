from error_codes import ErrorCode, ErrorSeverity, ErrorCategory, get_error_info, format_error_message
from app_base import AppBase

def example_usage():
    """Examples of using the error code system"""
    
    # Initialize the app base (which includes the logger)
    app = AppBase()
    
    # Example 1: Stream connection error
    try:
        # Simulate stream connection failure
        raise ConnectionError("RTSP stream timeout")
    except ConnectionError as e:
        app.log_error(
            ErrorCode.STREAM_CONNECTION_FAILED,
            context="channel_1",
            details=str(e)
        )
    
    # Example 2: YOLO model loading error
    try:
        # Simulate model loading failure
        raise FileNotFoundError("Model file not found")
    except FileNotFoundError as e:
        app.log_error(
            ErrorCode.MODEL_LOAD_FAILED,
            context="GPU_0",
            details=f"Model path: /path/to/model.pt - {str(e)}"
        )
    
    # Example 3: OCR processing error
    try:
        # Simulate OCR failure
        raise RuntimeError("Tesseract not initialized")
    except RuntimeError as e:
        app.log_error(
            ErrorCode.OCR_ENGINE_FAILED,
            context="channel_2",
            details=str(e)
        )
    
    # Example 4: Database error
    try:
        # Simulate database error
        raise Exception("Connection pool exhausted")
    except Exception as e:
        app.log_error(
            ErrorCode.DB_CONNECTION_FAILED,
            context="SQLite",
            details=str(e)
        )
    
    # Example 5: Webhook error
    app.log_error(
        ErrorCode.WEBHOOK_SEND_FAILED,
        context="http://192.168.1.188:3000/webhooks/camera",
        details="HTTP 500 - Internal Server Error"
    )
    
    print("All example errors have been logged!")


def demonstrate_error_info():
    """Demonstrate how to get error information"""
    
    error_codes_to_check = [
        ErrorCode.STREAM_CONNECTION_FAILED,
        ErrorCode.MODEL_LOAD_FAILED,
        ErrorCode.OCR_ENGINE_FAILED,
        ErrorCode.DB_CONNECTION_FAILED,
        ErrorCode.WEBHOOK_SEND_FAILED,
        ErrorCode.GRPC_SERVER_START_FAILED,
        ErrorCode.FILE_NOT_FOUND,
        ErrorCode.INVALID_CONFIG_PARAMETER
    ]
    
    print("Error Code Information:")
    print("=" * 80)
    
    for error_code in error_codes_to_check:
        info = get_error_info(error_code)
        print(f"Code: {error_code.name} ({error_code.value})")
        print(f"  Category: {info['category'].value}")
        print(f"  Severity: {info['severity'].value}")
        print(f"  Description: {info['description']}")
        print(f"  Formatted: {format_error_message(error_code, 'example_context', 'example details')}")
        print("-" * 40)


def show_error_categories():
    """Show all available error categories and their ranges"""
    
    print("Error Code Categories and Ranges:")
    print("=" * 50)
    print("1000-1999: System/Infrastructure errors")
    print("2000-2999: Video/Stream processing errors")
    print("3000-3999: YOLO/Detection errors")
    print("4000-4999: OCR processing errors")
    print("5000-5999: Database errors")
    print("6000-6999: Webhook/Network errors")
    print("7000-7999: gRPC/Communication errors")
    print("8000-8999: File/Storage errors")
    print("9000-9999: Configuration/Validation errors")
    print()
    
    # Show available severities
    print("Available Severities:")
    for severity in ErrorSeverity:
        print(f"  {severity.value}")
    print()
    
    # Show available categories
    print("Available Categories:")
    for category in ErrorCategory:
        print(f"  {category.value}")


def video_processor_examples():
    """Examples specific to video processor errors"""
    
    app = AppBase()
    
    # Stream connection issues
    app.log_error(
        ErrorCode.STREAM_CONNECTION_FAILED,
        context="channel_1",
        details="rtsp://127.23.23.15:8554/mystream_4 - Connection timeout"
    )
    
    # Frame processing issues
    app.log_error(
        ErrorCode.FRAME_DECODE_ERROR,
        context="channel_1",
        details="Corrupted H.264 frame at timestamp 00:05:23"
    )
    
    # YOLO detection issues
    app.log_error(
        ErrorCode.DETECTION_TIMEOUT,
        context="channel_1",
        details="Detection took 2.5s, exceeding 1s timeout"
    )
    
    # OCR specific issues
    app.log_error(
        ErrorCode.LOT_NUMBER_NOT_FOUND,
        context="channel_1",
        details="OCR confidence: 0.65, no lot pattern matched"
    )
    
    app.log_error(
        ErrorCode.EXPIRY_DATE_PARSE_ERROR,
        context="channel_1",
        details="Found text '2026-08' but could not parse as valid date"
    )
    
    print("Video processor examples logged!")


if __name__ == "__main__":
    print("Running Error Code Examples...")
    print()
    
    # Show error categories and ranges
    show_error_categories()
    print()
    
    # Demonstrate error information
    demonstrate_error_info()
    print()
    
    # Run usage examples
    example_usage()
    print()
    
    # Video processor specific examples
    video_processor_examples()
    
    print("Examples completed! Check your database for the logged errors.")
