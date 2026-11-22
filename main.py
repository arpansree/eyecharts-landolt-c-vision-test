"""Run the EyeCharts Landolt C prototype or check a saved screenshot."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--image", type=Path, help="Read a saved screenshot without starting a session.")
    mode.add_argument("--audio-file", type=Path, help="Check Whisper with an audio file.")
    mode.add_argument("--prepare", action="store_true", help="Download and load Whisper before a session.")
    parser.add_argument("--monitor", type=int, default=1, help="MSS display number, starting at 1.")
    parser.add_argument("--microphone", type=int, help="PortAudio input device number.")
    parser.add_argument("--listen-seconds", type=float, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model-dir", help="Optional location for the Whisper model cache.")
    parser.add_argument("--tesseract", default="tesseract", help="Tesseract executable or path.")
    args = parser.parse_args()
    if not 2 <= args.listen_seconds <= 30:
        parser.error("--listen-seconds must be between 2 and 30.")
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nSession canceled.", file=sys.stderr)
        return 130
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"EyeCharts prototype stopped. {error}", file=sys.stderr)
        return 1
    return 0


def run(args):
    if args.prepare or args.audio_file:
        from speech_input import SpeechInput

        if args.audio_file and not args.audio_file.is_file():
            raise FileNotFoundError(args.audio_file)
        speech = SpeechInput(args.device, args.microphone, args.model_dir)
        if args.audio_file:
            print(speech.transcribe(str(args.audio_file)))
        return

    from chart_reader import Chart, ChartError, read_acuity, read_directions

    if not shutil.which(args.tesseract):
        raise ChartError("Install Tesseract or pass --tesseract with its executable path.")
    if args.image:
        import numpy as np
        from PIL import Image

        with Image.open(args.image) as screenshot:
            image = np.asarray(screenshot.convert("L"))
        chart = Chart(read_acuity(image, args.tesseract), read_directions(image))
        print(json.dumps(asdict(chart)))
        return

    from desktop import EyeChartsDesktop
    from speech_input import SpeechInput
    from workflow import VisionTest

    desktop = EyeChartsDesktop(args.monitor)
    with desktop.session():
        speech = SpeechInput(args.device, args.microphone, args.model_dir)
        test = VisionTest(desktop, speech, args.tesseract, args.listen_seconds)
        input("Press Enter when the first BP reading ends. ")
        bp1_end = time.monotonic()
        input("Seat the participant 6 ft away, align their eye height, and set kiosk lighting. Press Enter when ready. ")
        test.run()
        desktop.say("The vision test is complete. Please proceed to your second blood pressure reading.")
        input("Operator: return to this terminal and press Enter when the second BP reading starts. ")
        test.session.bp_interval_seconds = time.monotonic() - bp1_end
        print(f"BP 1 end to BP 2 start: {test.session.bp_interval_seconds:.1f} seconds.")


if __name__ == "__main__":
    sys.exit(main())
