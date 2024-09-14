from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

@app.route('/audio', methods=['POST'])
def audio():
    return jsonify({
        'transcript': 'Hello, World!'
    })

@app.route('/transcript', methods=['POST'])
def transcript():
    from faster_whisper import WhisperModel
    jfk_path = "jfk.flac"
    model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root="/app")
    segments, info = model.transcribe(jfk_path, word_timestamps=True)
    resp = ''
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        resp += f"[%.2fs -> %.2fs] {segment.text}\n" % (segment.start, segment.end)
    return resp

if __name__ == '__main__':
    app.run(debug=True)