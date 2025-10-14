python3 -m venv venv
source venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
cd med_service
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

