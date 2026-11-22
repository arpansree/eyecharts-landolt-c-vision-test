"""Transcribe local microphone audio with Whisper large-v3."""
import logging
import math
import re
import sys
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
        import ctranslate2
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
        log_level = ctranslate2.get_log_level()
        quiet_startup = sys.platform == "darwin" and device == "cpu" and log_level < logging.ERROR
        if quiet_startup:
            # The CPU model emits an experimental MPS recommendation on Apple silicon.
            ctranslate2.set_log_level(logging.ERROR)
        try:
            self.model = WhisperModel(
                str(model_path), local_files_only=True, device=device,
                compute_type="float16" if device == "cuda" else "int8",
            )
        finally:
            if quiet_startup:
                ctranslate2.set_log_level(log_level)
        self.microphone = microphone

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=5, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def listen(self, seconds, *, stop_on_pause=True):
        import numpy as np
        import sounddevice as sd
        from faster_whisper.vad import get_speech_timestamps
        from scipy.signal import resample_poly

        try:
            rate = int(sd.query_devices(self.microphone, "input")["default_samplerate"])
            remaining = round(rate * seconds)
            block = max(1, round(rate * 0.4))
            divisor = math.gcd(rate, 16000)
            samples = []
            print(f"Listening for up to {seconds:g} seconds…", flush=True)
            with sd.InputStream(device=self.microphone, samplerate=rate, channels=1,
                                dtype="float32") as stream:
                while remaining:
                    frames = min(block, remaining)
                    audio, overflowed = stream.read(frames)
                    if overflowed:
                        raise AudioCaptureError("The microphone dropped audio.")
                    samples.append(audio[:, 0])
                    remaining -= frames
                    if stop_on_pause or not remaining:
                        audio = resample_poly(np.concatenate(samples), 16000 // divisor, rate // divisor)
                    if stop_on_pause and remaining:
                        speech = get_speech_timestamps(
                            audio, min_silence_duration_ms=1200, min_speech_duration_ms=100, speech_pad_ms=0,
                        )
                        if speech and len(audio) - speech[-1]["end"] >= 19200:
                            break
        except sd.PortAudioError as error:
            raise AudioCaptureError(str(error)) from error
        return self.transcribe(audio)
