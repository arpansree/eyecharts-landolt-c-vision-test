"""Both-eyes and per-eye scoring workflow. Session data stays in memory."""
import time
from dataclasses import dataclass, field

from chart_reader import Chart, ChartError, DIRECTIONS, read_acuity, read_directions
from speech_input import AudioCaptureError, parse_answers, words

LANDOLT_LEVELS = tuple(f"20/{value}" for value in (200, 160, 125, 100, 80, 60, 50, 40, 30, 25, 20))


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
    bp_interval_seconds: float | None = None


def summarize(rows, subject="this eye"):
    for index in reversed(range(len(rows))):
        row = rows[index]
        if row.score is not None and row.score >= 3:
            smaller_unscored = [item.chart.acuity for item in rows[index + 1:] if item.score is None]
            message = row.chart.acuity + "."
            if smaller_unscored:
                levels = ", ".join(smaller_unscored)
                if len(smaller_unscored) == 1:
                    message += f" We couldn’t score the {levels} row."
                else:
                    message += f" We couldn’t score these rows: {levels}."
            return message
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
                if level != expected:
                    raise ChartError(f"Expected acuity {expected}, but the screen showed {level}.")
                return Chart(level, read_directions(image) if read_row else ())
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

    def starting_chart(self):
        self.desktop.say("The test will start when the chart is ready. Wait for the words speak now before answering.")
        self.desktop.open_chart()
        print("Show fullscreen EyeCharts with five Landolt Cs and 20/200 at the top.", flush=True)
        return self.capture(LANDOLT_LEVELS[0], startup=True)

    def answer(self, row):
        for attempt in range(4):
            message = (
                "Read the five gaps from left to right. Speak now."
                if attempt == 0 else "Please repeat all five directions, from left to right. Speak now."
            )
            self.desktop.say(message)
            transcript = self.listen(self.listen_seconds)
            row.transcripts.append(transcript)
            if self.capture(row.chart.acuity) != row.chart:
                raise ChartError("The EyeCharts row changed while the participant was answering.")
            row.answers = parse_answers(transcript, DIRECTIONS)
            if row.answers is not None:
                row.matches = tuple(actual == expected for actual, expected in zip(
                    row.answers, row.chart.directions, strict=True
                ))
                self.desktop.say(f"{row.score} out of 5.")
                return
        self.desktop.say("We couldn’t score that response.")

    def listen(self, seconds):
        try:
            return self.speech.listen(seconds)
        except AudioCaptureError as error:
            print(f"Microphone capture failed: {error}", flush=True)
            return ""

    def ready(self, cover):
        for attempt in range(4):
            message = (
                f"Cover your {cover} eye with the occluder. When you are set, say ready. Speak now."
                if attempt == 0 else "Please say ready. Speak now."
            )
            self.desktop.say(message)
            if words(self.listen(max(8, self.listen_seconds))) in (["ready"], ["i", "am", "ready"], ["i", "m", "ready"]):
                return
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

    def run_round(self, eye, start):
        rows = self.session.rows[eye] = []
        chart = start
        for index, level in enumerate(LANDOLT_LEVELS):
            if index:
                if self.capture(chart.acuity) != chart:
                    raise ChartError("The EyeCharts row changed before advance.")
                self.desktop.step(smaller=True)
            chart = self.capture(level)
            if index == 0 and chart != start:
                raise ChartError("The starting row changed after reset.")
            row = Row(chart)
            rows.append(row)
            self.answer(row)
        label, subject = ("Both eyes", "both eyes") if eye == "both" else (f"{eye.capitalize()} eye", "this eye")
        self.session.results[eye] = summarize(rows, subject)
        self.desktop.say(f"{label}. {self.session.results[eye]}")

    def run(self):
        both = self.starting_chart()
        self.desktop.say(
            "The vision test is starting. Keep both eyes open. "
            "For each symbol, say where the opening is: up, down, left, or right. "
            "Say skip if you can’t tell. Wait for speak now before each row."
        )
        self.run_round("both", both)
        right = self.reset(both)
        self.ready("left")
        self.run_round("right", right)
        left = self.reset(right)
        self.ready("right")
        self.run_round("left", left)
        self.reset(left)
