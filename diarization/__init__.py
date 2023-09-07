import os

from pyannote.audio import Pipeline

access_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
model_ckpt = 'pyannote/speaker-diarization@2.1'

pipeline = Pipeline.from_pretrained(model_ckpt, use_auth_token=access_token)

diarization = pipeline()
