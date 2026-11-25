# EyeCharts Landolt C Vision Test

The Prev-Hub vision prototype reads a Landolt C chart from EyeCharts and guides the participant by voice. It runs three rounds, both eyes, right eye, and left eye, from 20/200 to 20/20. Each round scores eight rows: 20/200, 20/100, 20/80, 20/50, 20/40, 20/30, 20/25, and 20/20. Whisper large-v3 runs locally.

## Setup

On GitHub, select **Code → Download ZIP** and extract it. Open the folder that contains `main.py` and `requirements.txt`.

- **Windows** — Type `powershell` in File Explorer’s address bar and press Enter.
- **macOS** — Open Terminal, type `cd` followed by a space, drag the extracted folder into Terminal, and press Return.

Install EyeCharts, Python 3.12 or 3.13, and Tesseract with English language data. The first Whisper download needs internet access and about 3 GB of storage. [Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)

**Windows x64**

Add Tesseract to PATH and allow microphone access for desktop apps.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --prepare
```

**macOS**

```sh
brew install tesseract
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py --prepare
```

On macOS, allow Terminal or Python access to Screen Recording, Accessibility, and the microphone when prompted.

## Run the test

1. **Before running the program, open EyeCharts and set the chart to 20/200.** Use fullscreen Landolt C with five symbols per row and the feet-based acuity label.
2. Run `.\.venv\Scripts\python.exe main.py` on Windows or `.venv/bin/python main.py` on macOS.
3. Leave EyeCharts in front. On Windows, switch back to EyeCharts after launching the program. Say “ready” or “start” once per round. Full instructions play once, at the start of the both-eyes round. Give one answer for each of the five symbols, from left to right. Say “skip” for any symbol you can’t identify.

EyeCharts resets to 20/200 between rounds. To protect user privacy, the app keeps scores out of the spoken prompts. The kiosk is where users will review and optionally print their results.

The app can start transcribing during a pause while the mic is still listening. It uses that result only if there are five valid answers and it hears no more speech. First attempts use a shorter Whisper input window for brief answers. Retries use the standard window and a wider search. Add `--timing` for a terminal timing breakdown.

## Save results

Every completed test automatically saves a local `.txt` file with the three acuity results. The terminal shows the filename and path.

- **Windows** — `%LOCALAPPDATA%\EyeCharts Vision Test\Reports`
- **macOS** — `~/Library/Application Support/EyeCharts Vision Test/Reports`

The kiosk can pass `--session-id demo-1` to save the results as `vision-demo-1.txt`. Use a unique ID without personal details for each visit. The app keeps audio, screenshots, and individual answers in memory. The hub will handle review, optional printing, and deletion of the results file when the visit ends.

## Troubleshooting

Add `--check` to check the starting chart and spoken output without opening the microphone. Use `--monitor N` if EyeCharts is on a different display. If Tesseract is outside PATH, pass its executable path with `--tesseract`. Run `main.py --help` with the same Python executable to see all options.
