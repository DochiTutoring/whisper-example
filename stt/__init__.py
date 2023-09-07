import torch.cuda
import whisper

model_name = 'base.en'

def predict_by_whisper(audio_filename: str) -> str:
    """
    [Reference] https://github.com/opeanai/whisper

    :param audio_filename: str
            오디오 파일의 이름 (확장자 포함)
    :return: stt predict 결과
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_name)

    return model.transcribe(audio_filename)['text']
