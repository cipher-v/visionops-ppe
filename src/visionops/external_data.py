"""Utilities for auditing and mapping the RF100 external dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image, ImageOps

EXTERNAL_CLASS_NAMES = ("helmet", "no-helmet", "no-vest", "person", "vest")
INTERNAL_CLASS_NAMES = (
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
    "forklift",
    "no_vest",
)
EXTERNAL_TO_INTERNAL = {0: 0, 1: 6, 2: 11, 3: 5, 4: 2}
DHASH_DISTANCE_THRESHOLD = 4


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the uppercase SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def dhash64(path: Path) -> int:
    """Return a deterministic 64-bit difference hash after EXIF orientation."""
    with Image.open(path) as image:
        grayscale = ImageOps.exif_transpose(image).convert("L").resize((9, 8))
        pixels = list(grayscale.get_flattened_data())
    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | (
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return value


def hamming_distance(first: int, second: int) -> int:
    """Return the bitwise Hamming distance between two hashes."""
    return (first ^ second).bit_count()


def map_annotation_line(line: str) -> str:
    """Map one valid five-field external YOLO annotation to internal class IDs."""
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"Expected five YOLO fields, found {len(fields)}")
    external_id = int(fields[0])
    if external_id not in EXTERNAL_TO_INTERNAL:
        raise ValueError(f"Unknown external class ID: {external_id}")
    return " ".join((str(EXTERNAL_TO_INTERNAL[external_id]), *fields[1:]))
