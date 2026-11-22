"""Read the Snellen header and Landolt C gap directions from one display."""
import io
import re
import subprocess
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


class ChartError(RuntimeError):
    pass


DIRECTIONS = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}
SNELLEN = re.compile(r"\s*(20|6)\s*/\s*([0-9]+(?:[.,][0-9]+)?)\s*")


@dataclass(frozen=True)
class Chart:
    acuity: str
    directions: tuple[str, ...]


def read_acuity(gray, tesseract):
    height, width = gray.shape
    header = gray[:round(height * 0.18), width // 4:width * 3 // 4]
    # Per-row background removal discards the edge of the black header bar.
    background = np.median(header, axis=1).astype(np.int16)
    ink = (np.abs(header.astype(np.int16) - background[:, None]) >= 65).astype(np.uint8)
    bounds = Image.fromarray(ink).getbbox()
    if bounds is None:
        raise ChartError("No acuity text found in the top center of the display.")
    text = Image.fromarray((1 - ink) * 255).crop(bounds)
    if text.height < 40:
        text = text.resize((round(text.width * 40 / text.height), 40), Image.Resampling.NEAREST)
    encoded = io.BytesIO()
    ImageOps.expand(text, border=16, fill=255).save(encoded, format="PNG")
    try:
        result = subprocess.run(
            [tesseract, "stdin", "stdout", "--psm", "7",
             "-c", "tessedit_char_whitelist=0123456789/., "],
            input=encoded.getvalue(), capture_output=True, check=True, timeout=10,
        )
    except FileNotFoundError as error:
        raise ChartError("Tesseract was not found. Install it or pass --tesseract with its path.") from error
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ChartError(f"Tesseract could not read the acuity header: {error}") from error
    match = SNELLEN.fullmatch(result.stdout.decode("utf-8"))
    if match is None or float(match[2].replace(",", ".")) <= 0:
        raise ChartError("Could not read a Snellen acuity. Show 20/ or 6/ in the EyeCharts header.")
    return f"{match[1]}/{match[2].replace(',', '.')}"


def classify_glyph(component, contrast):
    height, width = component.shape
    size = max(width, height)
    padding = 2
    component = np.pad(component, padding)
    y, x = np.indices(component.shape, dtype=float)
    x = (x - padding - (width - 1) / 2) / size
    y = (y - padding - (height - 1) / 2) / size
    radius_squared = x * x + y * y
    ring = (radius_squared >= 0.29 ** 2) & (radius_squared <= 0.51 ** 2)
    scores = []
    for direction, (dx, dy) in DIRECTIONS.items():
        gap = ((x * dx + y * dy) > 0) & (np.abs(x * dy - y * dx) < 0.10)
        template = ring & ~gap
        overlap = np.count_nonzero(component & template)
        union = np.count_nonzero(component | template)
        scores.append((overlap / union, direction))
    scores.sort(reverse=True)
    if scores[0][0] >= 0.61 and scores[0][0] - scores[1][0] >= 0.06:
        return scores[0][1]
    if scores[0][0] < 0.60:
        return None

    # A narrow gap may be visible in antialiasing even if the thresholded
    # component fills it. Compare lightness at the four ends of the ring.
    contrast = np.pad(contrast.astype(float), padding)
    contrast /= contrast.max()
    ink = []
    for direction, (dx, dy) in DIRECTIONS.items():
        across = np.abs(x * dy - y * dx)
        along = x * dx + y * dy
        band = (along >= 0.25) & (along <= 0.51) & (across <= 0.16)
        weights = np.maximum(0, 1 - across[band] / 0.17)
        ink.append((np.average(contrast[band], weights=weights), direction))
    ink.sort()
    if ink[1][0] - ink[0][0] >= 0.15 and ink[2][0] >= 0.48:
        return ink[0][1]
    return None


def read_directions(gray):
    height, width = gray.shape
    center = gray[height // 4:height * 3 // 4, :]
    samples = center[::max(1, height // 60), ::max(1, width // 60)]
    background = int(np.median(samples))
    contrast = np.abs(center.astype(np.int16) - background)
    mask = contrast >= 55
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    glyphs = []
    for label, (vertical, horizontal) in enumerate(ndimage.find_objects(labels), start=1):
        x, y = horizontal.start, vertical.start
        w, h = horizontal.stop - x, vertical.stop - y
        size = max(w, h)
        if (min(w, h) < 5 or min(w, h) / size < 0.76
                or x == 0 or y == 0 or x + w == width or y + h == len(center)):
            continue
        direction = classify_glyph(labels[vertical, horizontal] == label, contrast[vertical, horizontal])
        if direction:
            glyphs.append((x + w / 2, y + h / 2, size, direction))
    glyphs.sort()
    if len(glyphs) != 5:
        raise ChartError(f"Expected one row of five Landolt Cs. Found {len(glyphs)} candidates.")
    centers = np.array([glyph[:3] for glyph in glyphs])
    sizes = centers[:, 2]
    size = np.median(sizes)
    gaps = np.diff(centers[:, 0])
    if (np.ptp(centers[:, 1]) > max(2, size * 0.35)
            or np.max(np.abs(sizes - size)) > max(2, size * 0.22)
            or gaps.min() < size or gaps.max() > gaps.min() * 1.5):
        raise ChartError("The five candidates do not form a single evenly spaced row.")
    return tuple(glyph[3] for glyph in glyphs)
