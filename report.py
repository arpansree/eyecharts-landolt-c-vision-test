"""Write a local handoff for a completed vision session."""
import os
import platform
import re
import tempfile
from pathlib import Path
from uuid import uuid4

ROUND_LABELS = {"both": "Both eyes", "right": "Right eye", "left": "Left eye"}
SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def validate_session_id(value):
    if not SESSION_ID.fullmatch(value):
        raise ValueError("Use 1–64 letters, digits, underscores, or hyphens, starting with a letter or digit.")
    return value


def report_directory():
    home = Path.home()
    if platform.system() == "Windows":
        root = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    else:
        root = home / "Library" / "Application Support"
    return root / "EyeCharts Vision Test" / "Reports"


def save_report(results, session_id=None):
    """Atomically write the three results using a supplied or generated session ID."""
    identifier = uuid4().hex if session_id is None else validate_session_id(session_id)
    path = report_directory() / f"vision-{identifier}.txt"
    text = "".join(f"{label}: {results[key].removesuffix('.')}\n" for key, label in ROUND_LABELS.items())
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=".vision-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(text)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path
