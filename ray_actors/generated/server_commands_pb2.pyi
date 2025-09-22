from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Command(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN: _ClassVar[Command]
    START: _ClassVar[Command]
    STOP: _ClassVar[Command]
    RESTART: _ClassVar[Command]

class FrameOrientation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_ORIENTATION: _ClassVar[FrameOrientation]
    PORTRAIT: _ClassVar[FrameOrientation]
    LANDSCAPE: _ClassVar[FrameOrientation]

class Rotation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_ROTATION: _ClassVar[Rotation]
    ROTATE_0: _ClassVar[Rotation]
    ROTATE_90: _ClassVar[Rotation]
    ROTATE_180: _ClassVar[Rotation]
    ROTATE_270: _ClassVar[Rotation]

class ProcessorType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ANY: _ClassVar[ProcessorType]
    GPU: _ClassVar[ProcessorType]
    CPU: _ClassVar[ProcessorType]
UNKNOWN: Command
START: Command
STOP: Command
RESTART: Command
UNKNOWN_ORIENTATION: FrameOrientation
PORTRAIT: FrameOrientation
LANDSCAPE: FrameOrientation
UNKNOWN_ROTATION: Rotation
ROTATE_0: Rotation
ROTATE_90: Rotation
ROTATE_180: Rotation
ROTATE_270: Rotation
ANY: ProcessorType
GPU: ProcessorType
CPU: ProcessorType

class ExecuteCommand(_message.Message):
    __slots__ = ("command", "name", "call_back_url", "input_url", "frame_orientation", "rotation", "processor_type", "model_name")
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CALL_BACK_URL_FIELD_NUMBER: _ClassVar[int]
    INPUT_URL_FIELD_NUMBER: _ClassVar[int]
    FRAME_ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    PROCESSOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    command: Command
    name: str
    call_back_url: str
    input_url: str
    frame_orientation: FrameOrientation
    rotation: Rotation
    processor_type: ProcessorType
    model_name: str
    def __init__(self, command: _Optional[_Union[Command, str]] = ..., name: _Optional[str] = ..., call_back_url: _Optional[str] = ..., input_url: _Optional[str] = ..., frame_orientation: _Optional[_Union[FrameOrientation, str]] = ..., rotation: _Optional[_Union[Rotation, str]] = ..., processor_type: _Optional[_Union[ProcessorType, str]] = ..., model_name: _Optional[str] = ...) -> None: ...

class ExecuteCommandRequest(_message.Message):
    __slots__ = ("execute_commands",)
    EXECUTE_COMMANDS_FIELD_NUMBER: _ClassVar[int]
    execute_commands: _containers.RepeatedCompositeFieldContainer[ExecuteCommand]
    def __init__(self, execute_commands: _Optional[_Iterable[_Union[ExecuteCommand, _Mapping]]] = ...) -> None: ...

class ExecuteCommandResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...
