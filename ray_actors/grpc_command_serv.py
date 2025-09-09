
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
        if not request.frame_orientation:
            valid=False
            message = 'invalid frame orientation'
        if not request.rotation:
            valid=False
            message = 'invalid rotation'
        if not request.processor_type:
            valid=False
            message = 'invalid processor_type'
        

        return valid, message

    
    def ExecuteCommand(self, request:pb2.ExecuteCommandRequest, context):
        success, message = self.validate_cmd(request)
        if not success:
            return pb2.ExecuteCommandResponse(success=success, message=message)
        
        if request.command == pb2.Command.START:
            message = f"starting server"
            print(message)
            
        if request.command == pb2.Command.STOP:
            message = f"stoppiny server"
            print(message)

        return pb2.ExecuteCommandResponse(success=success, message=message)



def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ServerCommandsServicer_to_server(ServerCommand(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gRPC server listening on port:{port}")
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
