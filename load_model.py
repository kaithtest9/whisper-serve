from faster_whisper import WhisperModel
import time

hg_token = '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'

jfk_path = "jfk.flac"
model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root="/app")
segments, info = model.transcribe(jfk_path, word_timestamps=True)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

def trans():
    import whisperx
    import gc 
    start = time.time()
    model = whisperx.load_model("base", device="cpu", compute_type="int8")
    audio = whisperx.load_audio('jfk.flac')
    print(f"Model load time: {time.time() - start} seconds")
    result = model.transcribe(audio, batch_size=16)
    print(f"Transcription time: {time.time() - start} seconds")
    start = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu")
    result = whisperx.align(result["segments"], model_a, metadata, audio, 'cpu', return_char_alignments=True)
    print(result["segments"])
    print(f"Alignment time: {time.time() - start} seconds")
    start = time.time()
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hg_token, device='cpu')
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(result["segments"])
    print(f"Diarization time: {time.time() - start} seconds")