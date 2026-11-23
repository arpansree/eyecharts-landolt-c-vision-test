"""Transcribe local microphone audio with Whisper large-v3."""
import logging
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

TOKEN = re.compile(r"[a-z]+")
SPEECH_RATE = 16000
DECODE_PAUSE = 6400


class AudioCaptureError(RuntimeError):
    pass


def words(transcript):
    return TOKEN.findall(transcript.casefold())


def parse_answers(transcript, vocabulary):
    tokens = words(transcript)
    # Whisper might put an "and" before the last answer, which is fine
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
        # a live test uses the cached model. only --prepare can download it
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
            # hide the experimental MPS tip during Mac startup without hiding errors
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
        self._stream = None

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=5, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    @contextmanager
    def microphone_session(self):
        import sounddevice as sd
        from faster_whisper.vad import get_vad_model

        print("Preparing microphone…", flush=True)
        get_vad_model()
        try:
            rate = int(sd.query_devices(self.microphone, "input")["default_samplerate"])
            with closing(sd.InputStream(device=self.microphone, samplerate=rate, channels=1,
                                        dtype="float32")) as stream:
                # warm up the mic now so we don't miss the first "ready". discard the setup audio
                stream.start()
                stream.abort()
                self._stream = stream
                try:
                    yield
                finally:
                    self._stream = None
        except sd.PortAudioError as error:
            raise AudioCaptureError(f"Could not initialize the microphone: {error}") from error

    def listen(self, seconds, *, stop_on_pause=True, accept_early=None, pause_seconds=1.2):
        import sounddevice as sd
        from faster_whisper.vad import get_speech_timestamps

        stream = self._stream
        if stream is None:
            raise AudioCaptureError("The microphone has not been prepared.")
        with ThreadPoolExecutor(max_workers=1) as decoder:
            try:
                audio, speech_end, pending, pending_end = self._record(
                    stream, seconds, stop_on_pause, accept_early, decoder, get_speech_timestamps,
                    pause_seconds,
                )
            except sd.PortAudioError as error:
                raise AudioCaptureError(str(error)) from error

            if pending is not None:
                # only use the early decode if the mic heard no more speech and all five answers are there
                same_utterance = speech_end == pending_end
                if same_utterance or not pending.cancel():
                    # finish the first decode before reusing the model with the full answer
                    try:
                        transcript = pending.result()
                    except Exception:
                        transcript = None
                    if same_utterance and transcript is not None and accept_early(transcript):
                        return transcript
            return self.transcribe(audio)

    def _record(self, stream, seconds, stop_on_pause, accept_early, decoder, detect_speech, pause_seconds):
        rate = int(stream.samplerate)
        remaining = round(rate * seconds)
        block = max(1, round(rate * 0.4))
        end_pause = round(pause_seconds * SPEECH_RATE)
        divisor = math.gcd(rate, SPEECH_RATE)
        samples = []
        speech_end = pending = pending_end = None
        print(f"Listening for up to {seconds:g} seconds…", flush=True)
        stream.start()
        try:
            while remaining:
                frames = min(block, remaining)
                audio, overflowed = stream.read(frames)
                if overflowed:
                    raise AudioCaptureError("The microphone dropped audio.")
                samples.append(audio[:, 0])
                remaining -= frames
                if stop_on_pause or not remaining:
                    audio = resample_poly(np.concatenate(samples), SPEECH_RATE // divisor, rate // divisor)
                if not stop_on_pause:
                    continue
                speech = detect_speech(
                    audio,
                    min_silence_duration_ms=320,
                    min_speech_duration_ms=100 if accept_early is None else 0,
                    speech_pad_ms=0,
                )
                speech_end = speech[-1]["end"] if speech else None
                if speech_end is None:
                    continue
                pause = len(audio) - speech_end
                if pause >= end_pause:
                    break
                if accept_early is not None and pending is None and remaining and pause >= DECODE_PAUSE:
                    # start at the first pause, but keep recording in case they're still answering
                    pending = decoder.submit(self.transcribe, audio)
                    pending_end = speech_end
        finally:
            stream.abort()
        return audio, speech_end, pending, pending_end
