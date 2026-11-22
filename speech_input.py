"""Transcribe local microphone audio with Whisper large-v3."""
import re
from pathlib import Path

TOKEN = re.compile(r"[a-z]+")


class AudioCaptureError(RuntimeError):
    pass


def words(transcript):
    return TOKEN.findall(transcript.casefold())


def parse_answers(transcript, vocabulary):
    tokens = words(transcript)
    # Whisper may add a conjunction before the last answer.
    if len(tokens) == 6 and tokens[-2] == "and":
        tokens.pop(-2)
    if len(tokens) == 5 and all(token in vocabulary or token == "skip" for token in tokens):
        return tuple(tokens)
    return None


class SpeechInput:
    def __init__(self, device, microphone, model_dir, *, prepare=False):
        import onnxruntime
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
        from huggingface_hub.errors import LocalEntryNotFoundError

        onnxruntime.disable_telemetry_events()
        print("Loading Whisper large-v3.", flush=True)
        try:
            model_path = Path(download_model("large-v3", cache_dir=model_dir, local_files_only=True))
        except LocalEntryNotFoundError:
            model_path = None
        if model_path is None or not all((model_path / filename).is_file() for filename in (
            "model.bin", "config.json", "tokenizer.json", "preprocessor_config.json",
        )):
            if not prepare:
                raise RuntimeError("The local Whisper model is missing or incomplete. Run main.py --prepare before testing.")
            print("Downloading the Whisper large-v3 model.", flush=True)
            model_path = Path(download_model("large-v3", cache_dir=model_dir))
        if not (model_path / "tokenizer.json").is_file():
            raise RuntimeError("The local Whisper tokenizer is missing. Run main.py --prepare again.")
        self.model = WhisperModel(
            str(model_path), local_files_only=True, device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        self.microphone = microphone

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=5, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def listen(self, seconds):
        import sounddevice as sd

        # WAV decoding resamples device-native recordings to Whisper's 16 kHz.
        import io
        import wave

        try:
            rate = int(sd.query_devices(self.microphone, "input")["default_samplerate"])
            print(f"Listening for {seconds:g} seconds…", flush=True)
            audio = sd.rec(round(rate * seconds), samplerate=rate, channels=1, dtype="int16",
                           device=self.microphone, blocking=True)
        except sd.PortAudioError as error:
            raise AudioCaptureError(str(error)) from error
        data = io.BytesIO()
        with wave.open(data, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(audio.astype("<i2", copy=False).tobytes())
        data.seek(0)
        return self.transcribe(data)
