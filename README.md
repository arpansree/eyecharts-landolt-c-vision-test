# EyeCharts vision-test prototype

EyeCharts displays the chart. This program captures the entire selected display, reads the top-center Snellen acuity and the center row of five Landolt Cs, and stores the gap directions from left to right in session memory. It records spoken English answers and transcribes them on the computer with Whisper **large-v3** via faster-whisper. Downloading the model the first time requires internet access and about 3 GB of storage. Screenshots, audio, transcripts, answer keys, and scores are kept in memory during the session. The terminal displays prompts, scores, results, and the BP interval.

## Set up

Run `--prepare` before testing. It is the only mode that can download Whisper model files. Live testing and audio-file checks load the model from the local cache.

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

1. Configure the kiosk once with fullscreen EyeCharts in Landolt C mode, a single row of five symbols, and a feet-based Snellen header. On macOS the prototype opens EyeCharts automatically. On Windows the kiosk launcher must bring EyeCharts to the foreground.
2. Launch `main.py` once the participant is seated and ready for the vision portion. The application announces that it is time for the vision test and returns EyeCharts to 20/200 if it is on another supported level. The participant follows spoken prompts for the rest of the session.
3. The participant says “ready” or “start” once to begin the both-eyes round. After each “Speak now” cue, they say the five gap directions from left to right and say “skip” for any they cannot see.
4. The program resets EyeCharts to a fresh 20/200 row, asks the participant to cover their left eye, and waits for “ready” or “start.” It scores the right eye, resets to a fresh 20/200 row, and repeats for the left eye with the right eye covered.
5. All three rounds run through all 11 levels from 20/200 to 20/20. Three or more correct answers pass a row. A skip is a miss. Unclear answers get up to three repeats before the row is marked unscored. The program checks the next displayed acuity after each advance.
6. The program announces each round’s acuity summary. After the last round it resets EyeCharts to 20/200 and exits so the hub can direct the participant to the next selected screening.

Camera-based eye-height alignment, chair guidance, and automatic BP timing await kiosk hardware integration. The vision test currently runs independently of those devices. Licensed EyeCharts Letters mode and the human-scorer comparison remain future work.

## Local data and reporting

Pass an opaque kiosk session ID to save a UTF-8 text file with the three acuity results. For example, `python main.py --session-id demo-1` saves `vision-demo-1.txt` after a completed vision test. Word can open the text file. Per-row scores, answers, screenshots, and audio stay in process memory. Runs without a session ID keep results in memory.

- **Windows** — `%LOCALAPPDATA%\EyeCharts Vision Test\Reports`
- **macOS** — `~/Library/Application Support/EyeCharts Vision Test/Reports`

The hub should supply a unique session ID, read the result file after the vision process succeeds, and delete it when the participant finishes the kiosk session. The hub owns the combined results view and the option to print. That integration is pending.

Use `python main.py --start-at 20/50` to score just the last five rows in each round. EyeCharts still returns to a fresh 20/200 row before each eye and verifies the intermediate levels on the way to 20/50.

If a run has trouble starting, run `python main.py --check` from the virtual environment. It plays a spoken prompt and waits up to 75 seconds for a readable starting chart. It leaves the chart on the same row and never opens the microphone. To check a saved screenshot, use `--image "/path/to/screenshot.png"`. To check local Whisper inference with a known recording, use `--audio-file "/path/to/voice.wav"`.
