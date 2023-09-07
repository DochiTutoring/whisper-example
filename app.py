import datetime

from stt import predict_by_whisper

# whisper (단독으로 3분 36초 음성을 42초만에 변환을 할 수 있음)
print(datetime.datetime.now())
whisper_example_result = predict_by_whisper('files/sample-conversation.mp3')
print(datetime.datetime.now())

print(whisper_example_result)
