from enum import Enum, IntEnum

class ErrorCode(IntEnum):
    """
    Error codes for AI Vision Multiprocessing System
    
    Error code ranges:
    1000-1999: System/Infrastructure errors
    2000-2999: Video/Stream processing errors  
    3000-3999: YOLO/Detection errors
    4000-4999: OCR processing errors
    5000-5999: Database errors
    6000-6999: Webhook/Network errors
    7000-7999: gRPC/Communication errors
    8000-8999: File/Storage errors
    9000-9999: Configuration/Validation errors
    """
    
    # System/Infrastructure errors (1000-1999)
    SYSTEM_STARTUP_FAILED = 1001
    SYSTEM_SHUTDOWN_ERROR = 1002
    MEMORY_ALLOCATION_ERROR = 1003
    GPU_INITIALIZATION_FAILED = 1004
    CPU_OVERLOAD = 1005
    THREAD_CREATION_FAILED = 1006
    PROCESS_SPAWN_ERROR = 1007
    RESOURCE_EXHAUSTED = 1008
    PERMISSION_DENIED = 1009
    SERVICE_UNAVAILABLE = 1010
    
    # Video/Stream processing errors (2000-2999)
    STREAM_CONNECTION_FAILED = 2001
    STREAM_TIMEOUT = 2002
    STREAM_DISCONNECTED = 2003
    INVALID_STREAM_URL = 2004
    UNSUPPORTED_VIDEO_FORMAT = 2005
    FRAME_DECODE_ERROR = 2006
    FRAME_CORRUPTION = 2007
    VIDEO_CODEC_ERROR = 2008
    CAMERA_NOT_FOUND = 2009
    RTSP_AUTH_FAILED = 2010
    FRAME_RATE_TOO_LOW = 2011
    RESOLUTION_NOT_SUPPORTED = 2012
    STREAM_BUFFER_OVERFLOW = 2013
    
    # YOLO/Detection errors (3000-3999)
    MODEL_LOAD_FAILED = 3001
    MODEL_INFERENCE_ERROR = 3002
    INVALID_MODEL_PATH = 3003
    MODEL_VERSION_MISMATCH = 3004
    DETECTION_TIMEOUT = 3005
    CONFIDENCE_THRESHOLD_ERROR = 3006
    BBOX_VALIDATION_FAILED = 3007
    CLASS_ID_INVALID = 3008
    GPU_MEMORY_INSUFFICIENT = 3009
    MODEL_CORRUPTED = 3010
    
    # OCR processing errors (4000-4999)
    OCR_ENGINE_FAILED = 4001
    TEXT_EXTRACTION_ERROR = 4002
    IMAGE_PREPROCESSING_FAILED = 4003
    TESSERACT_ERROR = 4004
    INVALID_OCR_REGION = 4005
    TEXT_CONFIDENCE_LOW = 4006
    LOT_NUMBER_NOT_FOUND = 4007
    EXPIRY_DATE_PARSE_ERROR = 4008
    OCR_TIMEOUT = 4009
    IMAGE_QUALITY_TOO_LOW = 4010
    ROTATION_DETECTION_FAILED = 4011
    OCR_PROCESS_IMAGE_FAILED = 4012
    OCR_SAVE_IMAGE_FAILED = 4013
    INVALID_OCR_IMAGE_SIZE = 4014
    
    # Database errors (5000-5999)
    DB_CONNECTION_FAILED = 5001
    DB_QUERY_ERROR = 5002
    DB_TIMEOUT = 5003
    DB_CONSTRAINT_VIOLATION = 5004
    DB_TRANSACTION_FAILED = 5005
    DB_LOCK_TIMEOUT = 5006
    DB_SCHEMA_MISMATCH = 5007
    DB_DISK_FULL = 5008
    DB_CORRUPTION = 5009
    DB_MIGRATION_FAILED = 5010
    
    # Webhook/Network errors (6000-6999)
    WEBHOOK_SEND_FAILED = 6001
    WEBHOOK_TIMEOUT = 6002
    WEBHOOK_INVALID_URL = 6003
    WEBHOOK_AUTH_FAILED = 6004
    HTTP_CONNECTION_ERROR = 6005
    HTTP_RESPONSE_ERROR = 6006
    JSON_SERIALIZATION_ERROR = 6007
    NETWORK_UNREACHABLE = 6008
    DNS_RESOLUTION_FAILED = 6009
    SSL_CERTIFICATE_ERROR = 6010
    
    # gRPC/Communication errors (7000-7999)
    GRPC_SERVER_START_FAILED = 7001
    GRPC_CLIENT_CONNECTION_FAILED = 7002
    GRPC_REQUEST_TIMEOUT = 7003
    GRPC_INVALID_REQUEST = 7004
    GRPC_AUTHENTICATION_FAILED = 7005
    GRPC_SERVICE_UNAVAILABLE = 7006
    PROTOBUF_SERIALIZATION_ERROR = 7007
    GRPC_CHANNEL_ERROR = 7008
    COMMAND_QUEUE_FULL = 7009
    RESULT_QUEUE_OVERFLOW = 7010
    
    # File/Storage errors (8000-8999)
    FILE_NOT_FOUND = 8001
    FILE_PERMISSION_ERROR = 8002
    DISK_SPACE_INSUFFICIENT = 8003
    FILE_CORRUPTION = 8004
    IMAGE_SAVE_FAILED = 8005
    LOG_FILE_ERROR = 8006
    CONFIG_FILE_MISSING = 8007
    DIRECTORY_CREATION_FAILED = 8008
    FILE_LOCK_ERROR = 8009
    BACKUP_FAILED = 8010
    
    # Configuration/Validation errors (9000-9999)
    INVALID_CONFIG_PARAMETER = 9001
    MISSING_REQUIRED_CONFIG = 9002
    CONFIG_VALIDATION_FAILED = 9003
    ENVIRONMENT_VARIABLE_MISSING = 9004
    INVALID_WORKER_COUNT = 9005
    INVALID_GPU_ID = 9006
    THRESHOLD_OUT_OF_RANGE = 9007
    PORT_ALREADY_IN_USE = 9008
    INVALID_FRAME_ORIENTATION = 9009
    UNSUPPORTED_PROCESSOR_TYPE = 9010


class ErrorSeverity(Enum):
    """Error severity levels for categorizing errors"""
    CRITICAL = "CRITICAL"    # System cannot continue, requires immediate attention
    ERROR = "ERROR"          # Functionality impaired, needs attention
    WARNING = "WARNING"      # Potential issue, should be monitored
    INFO = "INFO"           # Informational, normal operation


class ErrorCategory(Enum):
    """Error categories for grouping related errors"""
    SYSTEM = "SYSTEM"
    VIDEO = "VIDEO"
    DETECTION = "DETECTION"
    OCR = "OCR"
    DATABASE = "DATABASE"
    NETWORK = "NETWORK"
    COMMUNICATION = "COMMUNICATION"
    STORAGE = "STORAGE"
    CONFIGURATION = "CONFIGURATION"


def get_error_info(error_code: ErrorCode) -> dict:
    """
    Get detailed information about an error code
    
    Args:
        error_code: The ErrorCode enum value
        
    Returns:
        dict: Contains category, severity, and description
    """
    error_map = {
        # System errors
        ErrorCode.SYSTEM_STARTUP_FAILED: {
            "category": ErrorCategory.SYSTEM,
            "severity": ErrorSeverity.CRITICAL,
            "description": "System failed to start properly"
        },
        ErrorCode.GPU_INITIALIZATION_FAILED: {
            "category": ErrorCategory.SYSTEM,
            "severity": ErrorSeverity.CRITICAL,
            "description": "GPU initialization failed"
        },
        ErrorCode.MEMORY_ALLOCATION_ERROR: {
            "category": ErrorCategory.SYSTEM,
            "severity": ErrorSeverity.ERROR,
            "description": "Memory allocation failed"
        },
        
        # Video/Stream errors
        ErrorCode.STREAM_CONNECTION_FAILED: {
            "category": ErrorCategory.VIDEO,
            "severity": ErrorSeverity.ERROR,
            "description": "Failed to connect to video stream"
        },
        ErrorCode.STREAM_TIMEOUT: {
            "category": ErrorCategory.VIDEO,
            "severity": ErrorSeverity.WARNING,
            "description": "Video stream connection timeout"
        },
        
        # YOLO/Detection errors
        ErrorCode.MODEL_LOAD_FAILED: {
            "category": ErrorCategory.DETECTION,
            "severity": ErrorSeverity.CRITICAL,
            "description": "YOLO model failed to load"
        },
        ErrorCode.DETECTION_TIMEOUT: {
            "category": ErrorCategory.DETECTION,
            "severity": ErrorSeverity.WARNING,
            "description": "Object detection timeout"
        },
        
        # OCR errors
        ErrorCode.OCR_ENGINE_FAILED: {
            "category": ErrorCategory.OCR,
            "severity": ErrorSeverity.ERROR,
            "description": "OCR engine initialization failed"
        },
        ErrorCode.LOT_NUMBER_NOT_FOUND: {
            "category": ErrorCategory.OCR,
            "severity": ErrorSeverity.WARNING,
            "description": "Lot number not found in OCR text"
        },
        
        # Database errors
        ErrorCode.DB_CONNECTION_FAILED: {
            "category": ErrorCategory.DATABASE,
            "severity": ErrorSeverity.CRITICAL,
            "description": "Database connection failed"
        },

        # Saving image to disk error
        ErrorCode.IMAGE_SAVE_FAILED: {
            "category": ErrorCategory.OCR,
            "severity": ErrorSeverity.ERROR,
            "description": "Saving image to disk error"
        },
        
        # Webhook errors
        ErrorCode.WEBHOOK_SEND_FAILED: {
            "category": ErrorCategory.NETWORK,
            "severity": ErrorSeverity.ERROR,
            "description": "Failed to send webhook"
        },
        
        # gRPC errors
        ErrorCode.GRPC_SERVER_START_FAILED: {
            "category": ErrorCategory.COMMUNICATION,
            "severity": ErrorSeverity.CRITICAL,
            "description": "gRPC server failed to start"
        },
        
        # File/Storage errors
        ErrorCode.FILE_NOT_FOUND: {
            "category": ErrorCategory.STORAGE,
            "severity": ErrorSeverity.ERROR,
            "description": "Required file not found"
        },
        
        # Configuration errors
        ErrorCode.INVALID_CONFIG_PARAMETER: {
            "category": ErrorCategory.CONFIGURATION,
            "severity": ErrorSeverity.ERROR,
            "description": "Invalid configuration parameter"
        }
    }
    
    return error_map.get(error_code, {
        "category": ErrorCategory.SYSTEM,
        "severity": ErrorSeverity.ERROR,
        "description": f"Unknown error code: {error_code}"
    })


def format_error_message(error_code: ErrorCode, context: str = "", details: str = "") -> str:
    """
    Format a standardized error message
    
    Args:
        error_code: The ErrorCode enum value
        context: Additional context (e.g., channel name, file path)
        details: Specific error details
        
    Returns:
        str: Formatted error message
    """
    error_info = get_error_info(error_code)
    
    message = f"[{error_code.name}:{error_code.value}] {error_info['description']}"
    
    if context:
        message += f" | Context: {context}"
    
    if details:
        message += f" | Details: {details}"
    
    return message
