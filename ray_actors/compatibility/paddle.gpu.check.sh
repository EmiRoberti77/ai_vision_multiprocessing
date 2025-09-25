python - <<'PY'
import paddle
print("paddle:", paddle.__version__)
print("compiled_with_cuda:", paddle.is_compiled_with_cuda())
print("device before set:", paddle.device.get_device())
try:
    paddle.device.set_device("gpu")  # selects gpu:0
    print("device after set:", paddle.device.get_device())
    # tiny GPU op
    x = paddle.randn([1024, 1024], dtype='float32')
    w = paddle.randn([1024, 1024], dtype='float32')
    y = x @ w
    print("matmul ok, mean:", float(y.mean()))
except Exception as e:
    print("GPU test error:", e)
PY
