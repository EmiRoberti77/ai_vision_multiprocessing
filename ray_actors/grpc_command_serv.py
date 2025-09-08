import grpc
from concurrent import futures
from generated import server_commands_pb2_grpc as pb2_grpc
from generated import server_commands_pb2 as pb2


class ServerCommand(pb2_grpc.ServerCommandsServicer):
    def validate_cmd(self, request:pb2.ExecuteCommandRequest)-> bool | str:
        valid=True
        message = 'success'
        if not request.command:
            valid=False
            message = 'invalid command'
        if not request.input_url:
            valid=False
            message = 'invalid input_url'
        if not request.call_back_url:
            valid=False
            message = 'invalid call_back_url'
        

        return valid, message

    
    def ExecuteCommand(self, request:pb2.ExecuteCommandRequest, context):
        success, message = self.validate_cmd(request)
        return pb2.ExecuteCommandResponse(success=success, message=message)
        