import grpc
from concurrent import futures
from generated import server_commands_pb2_grpc as pb2_grpc
from generated import server_commands_pb2 as pb2
from video_processor import DetectionManager as DM
from video_processor import Detection_processor_type, DetectionParams
from db.db_logger import OAIX_db_Logger, LoggerLevel


class ServerCommand(pb2_grpc.ServerCommandsServicer):
    def __init__(self) -> None:
        self.dm = DM()
        self.db_logger = OAIX_db_Logger()
    
    def fill_detection_params(
        self,
        excecute_command:pb2.ExecuteCommand,
    ) -> DetectionParams:

        if excecute_command.processor_type == pb2.ProcessorType.GPU:
            det_process = Detection_processor_type.GPU
        elif excecute_command.processor_type == pb2.ProcessorType.CPU:
            det_process = Detection_processor_type.CPU
        else:
            det_process = Detection_processor_type.ANY

        det_rotation = excecute_command.rotation == pb2.Rotation.ROTATE_90

        detection_params = DetectionParams(
            name=excecute_command.name,
            video_source=excecute_command.input_url,
            webhook_call_back_url=excecute_command.call_back_url,
            rotate_90_clock=det_rotation,
            processor_type=det_process,
        )

        return detection_params

    def validate_cmd(self, request:pb2.ExecuteCommandRequest)-> tuple[bool, str]:
        valid=True
        message = 'success'

        if not len(request.execute_commands):
            return False, 'invalid execute_commands length'

        for execute_command in request.execute_commands:
            if not execute_command.command:
                return False, 'invalid command'
            if not execute_command.name:
                return False, 'invalid name'
            if not execute_command.input_url:
                return False, 'invalid input_url'
            if not execute_command.call_back_url:
                return False, 'invalid call_back_url'
            if not execute_command.frame_orientation:
                return False, 'invalid frame orientation'
            if not execute_command.rotation:
                return False, 'invalid rotation'
            if not execute_command.processor_type:
                return False, 'invalid processor_type'
            
        return valid, message

    
    def ExecuteCommand(self, request:pb2.ExecuteCommandRequest, context):
        success, message = self.validate_cmd(request)
        if not success:
            return pb2.ExecuteCommandResponse(success=success, message=message)
        
        for excecute_cmd in request.execute_commands:
            if excecute_cmd.command == pb2.Command.START: 
                message = f"starting server"
                dp = self.fill_detection_params(excecute_cmd)
                if self.dm.add(dp):
                    print(f"Channel_added_to_DM_{excecute_cmd.name=}:{excecute_cmd.input_url}")
                    if self.dm.start(excecute_cmd.name):
                        msg = f"Channel_start_DM_{excecute_cmd.name=}"
                        print(msg)
                        # self.db_logger(msg=msg)            
                print(excecute_cmd)
                
            if excecute_cmd.command == pb2.Command.STOP:
                message = f"stopping server"
                if self.dm.stop(excecute_cmd.name):
                    print(f"Channel_stopped_to_DM_{excecute_cmd.name=}:{excecute_cmd.input_url}")
                    if self.dm.remove(excecute_cmd.name):
                        print(f"Channel_remove_DM_{excecute_cmd.name=}")      
                print(excecute_cmd)

        return pb2.ExecuteCommandResponse(success=success, message=message)



def serve(port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pb2_grpc.add_ServerCommandsServicer_to_server(ServerCommand(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gRPC server listening on port:{port}")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print(f"exiting service")


if __name__ == '__main__':
    serve()
