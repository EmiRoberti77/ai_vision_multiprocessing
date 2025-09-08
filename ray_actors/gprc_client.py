import grpc
from generated import server_commands_pb2_grpc as pb2_grpc
from generated import server_commands_pb2 as pb2

def main(host:str='localhost', port:int=50051)->None:
    target = f"{host}:{port}"
    with grpc.insecure_channel(target=target) as channel:
        stub = pb2_grpc.ServerCommandsStub(channel=channel)
        req = pb2.ExecuteCommandRequest(
                                        command=1,
                                        call_back_url='http://localhost:8000/callback',
                                        input_url='',
                                        frame_orientation=pb2.FrameOrientation.UNKNOWN_ORIENTATION,
                                        rotation=pb2.UNKNOWN_ORIENTATION,
                                        processor_type=pb2.ProcessorType.GPU
                                    )   

        resp = stub.ExecuteCommand(req)
        print(resp) 


if __name__ == "__main__":
    main()
