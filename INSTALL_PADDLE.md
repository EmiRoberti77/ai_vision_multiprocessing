Install PaddleOCR with GPU on NVIDIA RTX 5090 (CUDA 12.x)

1) Create/activate a Python 3.10+ venv 
2) Install OpenCV and base deps:

```bash
pip install -r requirements_paddle_gpu.txt --no-cache-dir
```

3) Install the correct PaddlePaddle GPU wheel for CUDA 12.x:

- for the RTX5090 on WSL use this -- python3 -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/-cu128/

- Find wheels: see `https://www.paddlepaddle.org.cn/install/quick` and pick Linux, Pip, CUDA 12.x
- Example (CUDA 12.0+):

```bash
pip install --upgrade --no-cache-dir paddlepaddle-gpu==3.0.0 --index-url https://www.paddlepaddle.org.cn/whl/cu120
```

4) Install PaddleOCR:

```bash
pip install --no-cache-dir paddleocr==2.7.0.3
```

5) Verify import:

```bash
python -c "import paddle; import paddleocr; import cv2; print('OK', paddle.__version__)"
```

Notes
- If you use CUDA 12.1/12.2/12.3, adjust the index URL path (cu121/cu122/cu123) accordingly.
- On WSL2, ensure NVIDIA drivers are recent (>= 575.x) and `nvidia-smi` works.


