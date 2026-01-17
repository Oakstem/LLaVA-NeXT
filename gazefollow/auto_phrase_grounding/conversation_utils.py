#!/usr/bin/env python3
"""Shared helpers for building gaze conversations."""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Pillow may be unavailable in some envs.
    Image = None


OUTSIDE_FRAME_TARGET_DESCRIPTION = "something or someone outside the frame"
LOOKING_PATTERN = re.compile(r"\blooking\b.*", flags=re.IGNORECASE | re.DOTALL)
LOOKING_AT_PHRASE_PATTERN = re.compile(r"\blooking\s+at\b", flags=re.IGNORECASE)
DUPLICATE_WORD_PATTERN = re.compile(r"\b(\w+)\s+\1\b", flags=re.IGNORECASE)
ARTICLE_PATTERN = re.compile(r"^(a|an)\s+", flags=re.IGNORECASE)
QUESTION_SUBJECT_PATTERN = re.compile(
    r"describe\s+where\s+(?:the\s+)?(?P<subject>.+?)\s+is\s+looking(?:\s+at)?",
    flags=re.IGNORECASE | re.DOTALL,
)


def sanitize_person_description(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if "," in text:
        text = text.split(",", 1)[0]
    text = LOOKING_PATTERN.sub("", text).strip(" ,.;:")
    return text or None


def clean_question_phrase(description: str) -> str:
    if not description:
        return description
    text = ARTICLE_PATTERN.sub("", description).strip()
    while True:
        deduped = DUPLICATE_WORD_PATTERN.sub(r"\1 ", text)
        if deduped == text:
            break
        text = deduped.strip()
    return text


def has_looking_at_phrase(value: Optional[str]) -> bool:
    if not value or not isinstance(value, str):
        return False
    return bool(LOOKING_AT_PHRASE_PATTERN.search(value))


def extract_question_subject(prompt: Optional[str]) -> Optional[str]:
    if not prompt:
        return None
    text = prompt.replace("<image>", " ").strip()
    match = QUESTION_SUBJECT_PATTERN.search(text)
    if not match:
        return None
    subject = match.group("subject").strip()
    subject = subject.rstrip(" ?.!:,;")
    if subject.lower().startswith("the "):
        subject = subject[4:].strip()
    subject = clean_question_phrase(subject)
    return subject or None


def build_conversation_entry(question_subject: str, target_desc: str) -> Dict[str, str]:
    if not question_subject or not target_desc:
        raise ValueError("question_subject and target_desc must be provided")
    cleaned_subject = clean_question_phrase(question_subject)
    if cleaned_subject.lower().startswith("the "):
        cleaned_subject = cleaned_subject[4:].strip()
    cleaned_desc = target_desc.strip()
    if not cleaned_desc:
        raise ValueError("target_desc must not be empty")
    return {
        "from": "gpt",
        "value": f"The {cleaned_subject} is looking at {cleaned_desc}",
    }


def is_outside_frame(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        try:
            return int(value) == 0
        except (TypeError, ValueError):
            return False
    text = str(value).strip().lower()
    if not text:
        return False
    if text in {"0", "out", "outside", "outside_frame"}:
        return True
    try:
        return int(float(text)) == 0
    except ValueError:
        return False


def load_image_dims(images_root: Path, relative_path: str) -> tuple[int, int]:
    if Image is None:
        raise ModuleNotFoundError(
            "Pillow is required to load image dimensions. Install it or provide precomputed sizes."
        )
    image_path = images_root / relative_path
    with Image.open(image_path) as img:
        return int(img.width), int(img.height)


def get_target_point(record: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    target = record.get("target") or {}
    point = target.get("point")
    if isinstance(point, Sequence) and len(point) == 2:
        try:
            return float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
    bbox = target.get("bbox")
    if isinstance(bbox, Sequence) and len(bbox) == 4:
        try:
            left, top, right, bottom = (float(coord) for coord in bbox)
        except (TypeError, ValueError):
            return None
        return (left + right) / 2.0, (top + bottom) / 2.0
    return None


def candidate_image_paths(record: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    image_path = record.get("image_path")
    if image_path:
        paths.append(Path(image_path))
    rel = record.get("relative_path") or record.get("image_id")
    if rel:
        rel_path = Path(rel)
        if rel_path.is_absolute():
            paths.append(rel_path)
        else:
            images_root = record.get("_images_root")
            if images_root:
                paths.append(Path(images_root) / rel_path)
    return paths


def _probe_png_size(blob: bytes) -> Optional[Tuple[int, int]]:
    if len(blob) >= 24 and blob.startswith(b"\211PNG\r\n\032\n") and blob[12:16] == b"IHDR":
        width, height = struct.unpack(">II", blob[16:24])
        return int(width), int(height)
    return None


def _probe_jpeg_size(stream) -> Optional[Tuple[int, int]]:
    try:
        stream.seek(0)
        while True:
            byte = stream.read(1)
            if not byte:
                return None
            if byte != b"\xFF":
                continue
            marker = stream.read(1)
            if not marker or marker == b"\xD8":
                continue
            while marker == b"\xFF":
                marker = stream.read(1)
            if marker in {b"\xC0", b"\xC1", b"\xC2", b"\xC3", b"\xC5", b"\xC6", b"\xC7"}:
                stream.read(3)
                height, width = struct.unpack(">HH", stream.read(4))
                return int(width), int(height)
            segment_length = struct.unpack(">H", stream.read(2))[0]
            stream.seek(segment_length - 2, 1)
    except OSError:
        return None
    return None


def read_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    if not image_path:
        return None
    if Image is not None:
        try:
            with Image.open(image_path) as img:
                return int(img.width), int(img.height)
        except (OSError, FileNotFoundError):
            pass
    try:
        with image_path.open("rb") as fh:
            header = fh.read(24)
            png_size = _probe_png_size(header)
            if png_size:
                return png_size
            if header.startswith(b"\xFF\xD8"):
                return _probe_jpeg_size(fh)
    except (OSError, struct.error):
        return None
    return None


def infer_image_dimensions(record: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    width = record.get("gaze_gt_width")
    height = record.get("gaze_gt_height")
    if width is not None and height is not None:
        try:
            return int(width), int(height)
        except (TypeError, ValueError):
            pass
    for path in candidate_image_paths(record):
        size = read_image_size(path)
        if size:
            return size
    return None


def strip_image_extension(image_rel: str) -> str:
    image_rel = image_rel.strip()
    if "." in image_rel:
        return image_rel.rsplit(".", 1)[0]
    return image_rel


def strip_helper_keys(records: List[Dict[str, Any]], helper_keys: Sequence[str]) -> List[Dict[str, Any]]:
    if not helper_keys:
        return records
    helper_set = set(helper_keys)
    cleaned: List[Dict[str, Any]] = []
    for record in records:
        if helper_set.isdisjoint(record.keys()):
            cleaned.append(record)
        else:
            cleaned.append({k: v for k, v in record.items() if k not in helper_set})
    return cleaned


def build_outside_frame_conversation(
    record: Dict[str, Any],
    *,
    outside_answer: str = OUTSIDE_FRAME_TARGET_DESCRIPTION,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    identifier = record.get("annotation_id") or record.get("image_id") or record.get("id")
    if identifier is None:
        return None, "missing identifier"
    image_rel = record.get("relative_path") or record.get("image_id")
    if not image_rel:
        return None, "missing image path"
    source = record.get("source") or {}
    source_desc = sanitize_person_description(source.get("description"))
    if not source_desc:
        return None, "missing source description"
    target_point = get_target_point(record)
    if target_point is None:
        return None, "missing target point"
    dims = infer_image_dimensions(record)
    if dims is None:
        return None, "missing image dimensions"

    question_subject = clean_question_phrase(source_desc)
    conversation = [
        {
            "from": "human",
            "value": f"<image>\nDescribe where the {question_subject} is looking at",
        },
        build_conversation_entry(question_subject, outside_answer),
    ]

    return (
        {
            "id": str(identifier),
            "image": strip_image_extension(str(image_rel)),
            "num_people": 1,
            "conversations": conversation,
            "gaze_gt_x": target_point[0],
            "gaze_gt_y": target_point[1],
            "gaze_gt_width": dims[0],
            "gaze_gt_height": dims[1],
            "in_or_out": "0",
        },
        None,
    )
