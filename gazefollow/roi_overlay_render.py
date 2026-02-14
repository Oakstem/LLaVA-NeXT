from typing import Any, Dict, List, Tuple

import torch
from PIL import ImageDraw


def norm_box_to_pixels(box: torch.Tensor, image_w: int, image_h: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(image_w - 1, round(float(box[0].item()) * image_w))))
    y1 = int(max(0, min(image_h - 1, round(float(box[1].item()) * image_h))))
    x2 = int(max(0, min(image_w - 1, round(float(box[2].item()) * image_w))))
    y2 = int(max(0, min(image_h - 1, round(float(box[3].item()) * image_h))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def draw_labeled_norm_box(
    draw: ImageDraw.ImageDraw,
    box: torch.Tensor,
    image_w: int,
    image_h: int,
    color,
    width: int,
    label: str,
) -> None:
    x1, y1, x2, y2 = norm_box_to_pixels(box, image_w, image_h)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    text = label.strip()
    if not text:
        return
    text_x = x1 + 2
    text_y = max(0, y1 - 14)
    if hasattr(draw, "textbbox"):
        text_left, text_top, text_right, text_bottom = draw.textbbox((text_x, text_y), text)
        draw.rectangle([text_left - 1, text_top - 1, text_right + 1, text_bottom + 1], fill=(0, 0, 0))
    draw.text((text_x, text_y), text, fill=color)


def draw_overlay_text_lines(
    draw: ImageDraw.ImageDraw,
    lines: List[Dict[str, Any]],
    text_x: int = 8,
    text_y: int = 8,
    line_gap: int = 2,
) -> None:
    if not lines:
        return

    measured: List[Dict[str, Any]] = []
    max_width = 0
    total_height = 0
    for line in lines:
        text = str(line.get("text", ""))
        if not text:
            continue
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), text)
            width = max(1, int(right - left))
            height = max(1, int(bottom - top))
        else:
            width, height = draw.textsize(text)
            width = max(1, int(width))
            height = max(1, int(height))
        measured.append({"text": text, "color": line.get("color", (255, 255, 255)), "height": height, "width": width})
        max_width = max(max_width, width)
        total_height += height

    if not measured:
        return
    total_height += line_gap * max(0, len(measured) - 1)
    draw.rectangle(
        [text_x - 2, text_y - 2, text_x + max_width + 2, text_y + total_height + 2],
        fill=(0, 0, 0),
    )

    curr_y = text_y
    for line in measured:
        draw.text((text_x, curr_y), line["text"], fill=line["color"])
        curr_y += int(line["height"]) + line_gap
