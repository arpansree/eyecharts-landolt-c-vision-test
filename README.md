# EyeCharts vision-test prototype

EyeCharts displays the chart. This program captures the entire selected display, reads the top-center Snellen acuity and the center row of five Landolt Cs, and stores the gap directions from left to right in session memory. It records spoken English answers and transcribes them on the computer with Whisper **large-v3** via faster-whisper. Downloading the model the first time requires internet access and about 3 GB of storage. Screenshots, audio, transcripts, answer keys, and scores are kept in memory during the session. The terminal displays prompts, scores, results, and the BP interval.

## Set up

Use Python 3.12 or 3.13 and install EyeCharts and Tesseract with the English language data. The CPU path uses INT8 inference. The model runtime provides wheels for Intel and Apple silicon macOS and x64 Windows. [CTranslate2 installation](https://opennmt.net/CTranslate2/installation.html) · [Tesseract installation](https://tesseract-ocr.github.io/tessdoc/Installation.html)

**macOS**

```sh
brew install tesseract
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py --prepare
.venv/bin/python main.py
```

Allow the launching terminal or Python process in macOS Privacy & Security for Screen & System Audio Recording, Accessibility, and Microphone as prompted. Relaunch after changing permissions.

**Windows x64**

Install the Windows Tesseract build linked from its installation guide and put its folder on PATH. You can also pass the path directly with `--tesseract "C:\Program Files\Tesseract-OCR\tesseract.exe"`. Enable microphone access for desktop apps.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --prepare
.\.venv\Scripts\python.exe main.py
```

Use `--monitor N` if fullscreen EyeCharts is on a display other than display 1. Use `--microphone N` to choose a PortAudio input. After each spoken prompt the default listening window is eight seconds. Set `--listen-seconds` to adjust it. Windows systems with the compatible NVIDIA libraries can select `--device cuda`. [faster-whisper GPU setup](https://github.com/SYSTRAN/faster-whisper#gpu)

## Run a session

1. Set EyeCharts to fullscreen Landolt C with a single row of five symbols, a feet-based Snellen header, and a starting level of 20/200. Keep the chart visible and in the foreground after the operator presses Enter and switches to it.
2. At the end of BP 1, press Enter to start the timer. Seat the participant six feet away, align their eye height, and set the room lighting. Confirm the setup at the second prompt.
3. The participant completes the first scored round with both eyes open. After each spoken “Speak now” cue, they say the five gap directions from left to right and say “skip” for any they cannot see. The same cue starts recording when the program asks them to say “ready.”
4. The program resets EyeCharts to a fresh 20/200 row, asks the participant to cover their left eye, and waits for “ready.” It scores the right eye, resets to a fresh 20/200 row, and repeats for the left eye with the right eye covered.
5. All three rounds run through all 11 levels from 20/200 to 20/20. Three or more correct answers pass a row. A skip is a miss. Unclear answers get up to three repeats before the row is marked unscored. The program checks the next displayed acuity after each advance.
6. The program reports the smallest passed row and any smaller unscored rows separately for each round. It resets EyeCharts to 20/200 and guides the participant to BP 2. Press Enter in the terminal when BP 2 starts to record the elapsed time.

Camera-based eye-height alignment, chair guidance, and BP device integration are pending hardware interfaces. The operator handles those steps in this build. Licensed EyeCharts Letters mode and the human-scorer comparison remain future work. The workflow keeps chart capture and speech parsing in separate modules to support that adaptation.

If a run has trouble starting, run `python main.py --check` from the virtual environment. It plays a spoken prompt and waits up to 75 seconds for a readable starting chart. It leaves the chart on the same row and never opens the microphone. To check a saved screenshot, use `--image "/path/to/screenshot.png"`. To check local Whisper inference with a known recording, use `--audio-file "/path/to/voice.wav"`.
