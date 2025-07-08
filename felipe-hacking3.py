import whisper

model = whisper.load_model("large-v3", download_root="./cache")
result = model.transcribe("./corpus/How AI is helping transform coral reef conservation _ Ben Williams _ TEDxLondon.mp3", task="transcribe", language="en")
print('result:', result['text'])
