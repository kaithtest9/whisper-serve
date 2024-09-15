pip install faster-whisper
pip install flask gunicorn
apt-get update && apt-get install -y git
pip install git+https://github.com/m-bain/whisperx.git
python load_model.py