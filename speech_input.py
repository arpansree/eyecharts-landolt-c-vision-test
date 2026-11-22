"""Transcribe local microphone audio with Whisper large-v3."""
import re

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
    def __init__(self, device, microphone, model_dir):
        from faster_whisper import WhisperModel
        from huggingface_hub.errors import LocalEntryNotFoundError

        print("Loading Whisper large-v3.", flush=True)
        options = {
            "device": device,
            "compute_type": "float16" if device == "cuda" else "int8",
            "download_root": model_dir,
        }
        try:
            self.model = WhisperModel("large-v3", local_files_only=True, **options)
        except LocalEntryNotFoundError:
            print("Downloading the Whisper large-v3 model.", flush=True)
            self.model = WhisperModel("large-v3", **options)
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
