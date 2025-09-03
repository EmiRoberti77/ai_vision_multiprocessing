python3 -m grpc_tools.protoc \
  -I proto \
  --python_out=multprocessing/generated \
  --grpc_python_out=multprocessing/generated \
  --pyi_out=multprocessing/generated \
  ./proto/server_commands.proto