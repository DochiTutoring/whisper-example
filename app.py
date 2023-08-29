import whisper

# [Reference] https://github.com/openai/whisper
model_name: str = 'base.en'
model = whisper.load_model('base.en')

audio_filename = 'files/sample-0.mp3'
result = model.transcribe(audio_filename)

print(result['text'])
