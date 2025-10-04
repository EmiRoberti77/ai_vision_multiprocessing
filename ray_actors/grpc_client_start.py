import grpc
import sys
from generated import server_commands_pb2_grpc as pb2_grpc
from generated import server_commands_pb2 as pb2

models = {
    "model_oaix_box":"oaix_medicine_v1.pt",
    "model_yolo_11_m":"yolo11m.pt",
}

_WEBHOOK = "http://192.168.1.188:3000/webhooks/camera"
_INPUT = "rtsp://127.23.23.15:8554/mystream_"
_NAME = "channel_1"
_MODEL_NAME = models.get("model_oaix_box")

print('START_COMMAND')
print(f"{_WEBHOOK=}")
print(f"{_INPUT=}")
print(f"{_NAME=}")
print(f"{_MODEL_NAME=}")


def main(stream_val:str,  rotate_90_clock_wise:bool = False, host: str = "localhost", port: int = 50051) -> None:
    target = f"{host}:{port}"
    with grpc.insecure_channel(target=target) as channel:
        stub = pb2_grpc.ServerCommandsStub(channel=channel)
        # print(pb2.Rotation.ROTATE_0, pb2.Rotation.ROTATE_90)
        channel_1 = pb2.ExecuteCommand(
            command=pb2.Command.START,
            name=_NAME,
            call_back_url=_WEBHOOK,
            input_url=_INPUT+stream_val,
            frame_orientation=pb2.FrameOrientation.LANDSCAPE,
            rotation=pb2.Rotation.ROTATE_90 if rotate_90_clock_wise else pb2.Rotation.ROTATE_0,
            processor_type=pb2.ProcessorType.GPU,
            model_name=_MODEL_NAME
        )

        req = pb2.ExecuteCommandRequest(execute_commands=[channel_1])
        print(f"{channel_1=}")
        resp = stub.ExecuteCommand(req)
        print(resp)


if __name__ == "__main__":

    print(sys.argv)
    stream_val = 4
    rotate_90_clock_wise = False

    if len(sys.argv) > 1:
        stream_val = sys.argv[1]
        rotate_90_clock_wise = sys.argv[2].lower() in ('true', '1', 'yes')

    print(f"{stream_val=}")
    print(f"{rotate_90_clock_wise=}")

    main(stream_val=stream_val, rotate_90_clock_wise=rotate_90_clock_wise)
