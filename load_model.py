from faster_whisper import WhisperModel

jfk_path = "jfk.flac"
model = WhisperModel("tiny", device="cpu", compute_type="int8", download_root="/app")
segments, info = model.transcribe(jfk_path, word_timestamps=True)
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))