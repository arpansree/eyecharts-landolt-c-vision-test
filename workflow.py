"""Both-eyes and per-eye scoring workflow. Session data stays in memory."""
import time
from dataclasses import dataclass, field

from chart_reader import Chart, ChartError, DIRECTIONS, read_acuity, read_directions
from speech_input import AudioCaptureError, parse_answers, words

LANDOLT_LEVELS = tuple(f"20/{value}" for value in (200, 160, 125, 100, 80, 60, 50, 40, 30, 25, 20))
SCORED_LEVELS = ("20/200", "20/100", "20/80", "20/50", "20/40", "20/30", "20/25", "20/20")


@dataclass
class Row:
    chart: Chart
    transcripts: list[str] = field(default_factory=list)
    answers: tuple[str, ...] | None = None
    matches: tuple[bool, ...] | None = None

    @property
    def score(self):
        return None if self.matches is None else sum(self.matches)


@dataclass
class Session:
    rows: dict[str, list[Row]] = field(default_factory=dict)
    results: dict[str, str] = field(default_factory=dict)


def summarize(rows, subject="this eye"):
    for row in reversed(rows):
        if row.score is not None and row.score >= 3:
            return row.chart.acuity
    if any(row.score is None for row in rows):
        return f"We couldn’t capture a result for {subject} due to a system error."
    return "Below the chart’s tested range."


class VisionTest:
    def __init__(self, desktop, speech, tesseract, listen_seconds):
        self.desktop = desktop
        self.speech = speech
        self.tesseract = tesseract
        self.listen_seconds = listen_seconds
        self.session = Session()

    def capture(self, expected, *, read_row=True, startup=False):
        deadline = time.monotonic() + 75 if startup else None
        previous_issue = None
        attempts_remaining = 4
        while True:
            try:
                image = self.desktop.screenshot()
                level = read_acuity(image, self.tesseract)
                if expected is not None and level != expected:
                    raise ChartError(f"Expected acuity {expected}, but the screen showed {level}.")
                if not read_row:
                    return Chart(level, ())
                try:
                    return Chart(level, read_directions(image))
                except ChartError as error:
                    raise ChartError(f"Could not read all five gaps at {level}: {error}") from error
            except ChartError as error:
                issue = error
                if deadline is not None:
                    if str(issue) != previous_issue:
                        print(f"Waiting for the starting chart: {issue}", flush=True)
                        previous_issue = str(issue)
                    if time.monotonic() >= deadline:
                        break
                else:
                    attempts_remaining -= 1
                    if attempts_remaining == 0:
                        break
                time.sleep(0.8 if startup else 0.3)
        raise ChartError(f"Could not verify the EyeCharts display. {issue}") from issue

    def starting_chart(self, *, reposition=True):
        self.desktop.open_chart()
        first = LANDOLT_LEVELS[0]
        if reposition:
            chart = self.capture(None, read_row=False, startup=True)
            if chart.acuity not in LANDOLT_LEVELS:
                raise ChartError(f"EyeCharts showed {chart.acuity}, outside the supported startup range.")
            for level in reversed(LANDOLT_LEVELS[:LANDOLT_LEVELS.index(chart.acuity)]):
                self.desktop.step(smaller=False)
                self.capture(level, read_row=False)
        return self.capture(first, startup=not reposition)

    def answer(self, row, *, first_row):
        for attempt in range(4):
            if attempt:
                message = "Please repeat all five directions from left to right. Speak now."
            elif first_row:
                message = (
                    "Look at the five symbols. Starting on the left, say which way each gap opens: "
                    "up, down, left, or right. If you can’t tell, say ‘skip.’ Speak now."
                )
            else:
                message = "Next row. Speak now."
            self.desktop.say(message)
            transcript = self.listen(self.listen_seconds, stop_on_pause=attempt == 0)
            row.transcripts.append(transcript)
            if self.capture(row.chart.acuity) != row.chart:
                raise ChartError("The EyeCharts row changed while the participant was answering.")
            row.answers = parse_answers(transcript, DIRECTIONS)
            if row.answers is not None:
                row.matches = tuple(actual == expected for actual, expected in zip(
                    row.answers, row.chart.directions, strict=True
                ))
                return
        self.desktop.say("We couldn’t score that response.")

    def listen(self, seconds, *, stop_on_pause=True):
        try:
            return self.speech.listen(seconds, stop_on_pause=stop_on_pause)
        except AudioCaptureError as error:
            print(f"Microphone capture failed: {error}", flush=True)
            return ""

    def wait_for_start(self, instruction, seconds):
        accepted = (["ready"], ["start"], ["i", "am", "ready"], ["i", "m", "ready"])
        for attempt in range(4):
            message = instruction if attempt == 0 else "Please say ready or start."
            self.desktop.say(f"{message} Speak now.")
            if words(self.listen(seconds)) in accepted:
                return True
        return False

    def ready(self, cover):
        instruction = f"Cover your {cover} eye with the occluder. When you are ready, say the words, ready, or start."
        if not self.wait_for_start(instruction, max(8, self.listen_seconds)):
            raise ChartError("Could not confirm readiness. The next eye was not tested.")

    def reset(self, previous_start):
        for level in reversed(LANDOLT_LEVELS[:-1]):
            self.desktop.step(smaller=False)
            chart = self.capture(level, read_row=level == LANDOLT_LEVELS[0])
        # EyeCharts can randomly repeat a sequence. Give it another chance.
        for _ in range(4):
            if chart != previous_start:
                return chart
            self.desktop.step(smaller=True)
            self.capture(LANDOLT_LEVELS[1], read_row=False)
            self.desktop.step(smaller=False)
            chart = self.capture(LANDOLT_LEVELS[0])
        if chart == previous_start:
            raise ChartError("EyeCharts kept showing the previous 20/200 sequence after reset.")
        return chart

    def run_round(self, eye, start, start_at):
        rows = self.session.rows[eye] = []
        chart = self.capture(LANDOLT_LEVELS[0])
        if chart != start:
            raise ChartError("The starting row changed after reset.")
        first = LANDOLT_LEVELS.index(start_at)
        for index, level in enumerate(LANDOLT_LEVELS):
            scored = index >= first and level in SCORED_LEVELS
            if index:
                self.desktop.step(smaller=True)
                chart = self.capture(level, read_row=scored)
            if not scored:
                continue
            row = Row(chart)
            rows.append(row)
            self.answer(row, first_row=len(rows) == 1)
        subject = "both eyes" if eye == "both" else "this eye"
        self.session.results[eye] = summarize(rows, subject)

    def run(self, *, start_at="20/200"):
        self.desktop.say("It’s time for the vision test.")
        both = self.starting_chart()
        self.desktop.say("Keep both eyes open.")
        if not self.wait_for_start("To begin the both eyes round, say ready or start.", max(8, self.listen_seconds)):
            raise ChartError("Could not confirm readiness. The both-eyes round did not start.")
        self.run_round("both", both, start_at)
        right = self.reset(both)
        self.desktop.say("The chart has reset to twenty over two hundred for the right eye.")
        self.ready("left")
        self.run_round("right", right, start_at)
        left = self.reset(right)
        self.desktop.say("The chart has reset to twenty over two hundred for the left eye.")
        self.ready("right")
        self.run_round("left", left, start_at)
        self.reset(left)
