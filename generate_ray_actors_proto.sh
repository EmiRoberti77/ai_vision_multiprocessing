python3 -m grpc_tools.protoc \
  -I proto \
  --python_out=ray_actors/generated \
  --grpc_python_out=ray_actors/generated \
  --pyi_out=ray_actors/generated \
  ./proto/server_commands.proto