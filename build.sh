pip install faster-whisper
pip install flask gunicorn
apt-get update && apt-get install -y git
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/m-bain/whisperx.git
python load_model.py