from flask import Flask, jsonify, request
import time

app = Flask(__name__)

hg_token = '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'

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
    start = time.time()
    model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root="/tmp")
    print(f"Model load time: {time.time() - start} seconds")
    start = time.time()
    segments, info = model.transcribe(jfk_path, word_timestamps=True)
    resp = ''
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        resp += f"[%.2fs -> %.2fs] {segment.text}\n" % (segment.start, segment.end)
    print(f"Transcription time: {time.time() - start} seconds")

    return resp

@app.route('/transcribe-by-whisper-x', methods=['POST'])
def transcribe_by_whisper_x():
    import whisperx
    import gc 
    start = time.time()
    model = whisperx.load_model("base", device="cpu", compute_type="int8", download_root="/tmp")
    audio = whisperx.load_audio('jfk.flac')
    print(f"Model load time: {time.time() - start} seconds")
    result = model.transcribe(audio, batch_size=16)
    print(f"Transcription time: {time.time() - start} seconds")
    print(result["segments"]) # before alignment

    return jsonify(result)

@app.route('/transcribe-by-whisper-x-align', methods=['POST'])
def transcribe_by_whisper_x_align():
    import whisperx
    import gc 
    start = time.time()
    model = whisperx.load_model("base", device="cpu", compute_type="int8", download_root="/tmp")
    audio = whisperx.load_audio('jfk.flac')
    print(f"Model load time: {time.time() - start} seconds")
    result = model.transcribe(audio, batch_size=16)
    print(f"Transcription time: {time.time() - start} seconds")
    start = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu", download_root="/tmp")
    result = whisperx.align(result["segments"], model_a, metadata, audio, 'cpu', return_char_alignments=False)
    print(result["segments"]) # after alignment
    print(f"Alignment time: {time.time() - start} seconds")

    return jsonify(result)

@app.route('/transcribe-by-whisper-x-align-char', methods=['POST'])
def transcribe_by_whisper_x_align_char():
    import whisperx
    import gc 
    start = time.time()
    model = whisperx.load_model("base", device="cpu", compute_type="int8", download_root="/tmp")
    audio = whisperx.load_audio('jfk.flac')
    print(f"Model load time: {time.time() - start} seconds")
    result = model.transcribe(audio, batch_size=16)
    print(f"Transcription time: {time.time() - start} seconds")
    start = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu", download_root="/tmp")
    result = whisperx.align(result["segments"], model_a, metadata, audio, 'cpu', return_char_alignments=True)
    print(result["segments"])
    print(f"Alignment time: {time.time() - start} seconds")
    start = time.time()
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hg_token, device='cpu')
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(result["segments"])
    print(f"Diarization time: {time.time() - start} seconds")

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)