import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
import platform
# Add project root to sys.path
try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from gazefollow.generation_utils import fix_wsl_paths, load_image

DEFAULT_IMAGE_PATH = r"D:\Projects\data\gazefollow\train\00000093\00093143.jpg"


def _prompt_bbox_string(label: str) -> str:
    return input(
        f"\nEnter {label} bounding box as x_min,y_min,x_max,y_max "
        "(type 'q' to quit): "
    ).strip()


def _parse_bbox_values(text: str) -> Optional[List[float]]:
    if not text:
        return None
    lower = text.lower()
    if lower in {"q", "quit", "exit"}:
        return None

    parts = [token.strip() for token in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Expected four comma-separated values.")
    return [float(value) for value in parts]


def _convert_to_pixel_bounds(
    values: List[float],
    image_size: Tuple[int, int],
    mode: str,
) -> Tuple[int, int, int, int, str]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    detected_mode = mode
    if mode == "auto":
        if all(0.0 <= val <= 1.0 for val in values):
            detected_mode = "normalized"
        else:
            detected_mode = "pixel"

    x_min_val, y_min_val, x_max_val, y_max_val = values
    if detected_mode == "normalized":
        x_min_val *= (width - 1)
        x_max_val *= (width - 1)
        y_min_val *= (height - 1)
        y_max_val *= (height - 1)

    x_min = int(round(x_min_val))
    y_min = int(round(y_min_val))
    x_max = int(round(x_max_val))
    y_max = int(round(y_max_val))

    x_min = max(0, min(width - 1, x_min))
    y_min = max(0, min(height - 1, y_min))
    x_max = max(x_min + 1, min(width, x_max))
    y_max = max(y_min + 1, min(height, y_max))

    if x_max <= x_min or y_max <= y_min:
        raise ValueError(
            f"Invalid bounds after clamping: {(x_min, y_min, x_max, y_max)}"
        )

    return x_min, y_min, x_max, y_max, detected_mode


def _create_mask(image_size: Tuple[int, int], bounds: Tuple[int, int, int, int]) -> np.ndarray:
    width, height = image_size
    x_min, y_min, x_max, y_max = bounds
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


def _normalize_bounds(bounds: Tuple[int, int, int, int], image_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    width, height = image_size
    if width <= 1 or height <= 1:
        return (0.0, 0.0, 0.0, 0.0)
    x_min, y_min, x_max, y_max = bounds
    return (
        x_min / (width - 1),
        y_min / (height - 1),
        x_max / (width - 1),
        y_max / (height - 1),
    )


class _InteractiveBBoxSelector:
    """Use matplotlib RectangleSelector to capture a bounding box."""

    def __init__(self, image_array: np.ndarray, title: str):
        self.image_array = image_array
        self._raw_bbox: Optional[Tuple[float, float, float, float]] = None
        self.confirmed = False

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image_array)
        self.ax.set_title(
            f"{title}\n"
            "Drag to select a region.\n"
            "Press Enter to confirm, 'r' to reset, or Esc to cancel."
        )
        self.ax.set_axis_off()

        self.selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            interactive=True,
            props=dict(alpha=0.5),
        )
        self._cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self._cid_close = self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_select(self, eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self._raw_bbox = (x1, y1, x2, y2)

    def _on_key_press(self, event):
        key = event.key.lower() if event.key else ""
        if key in {"enter", "return"}:
            if self._raw_bbox is not None:
                self.confirmed = True
                plt.close(self.fig)
        elif key in {"escape", "esc", "q"}:
            self._raw_bbox = None
            self.confirmed = False
            plt.close(self.fig)
        elif key == "r":
            self._raw_bbox = None
            self.selector.set_active(True)

    def _on_close(self, _):
        if not self.confirmed:
            self._raw_bbox = None

    def show(self):
        plt.show()

    def get_bounds(self, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        if not self.confirmed or self._raw_bbox is None:
            return None
        x1, y1, x2, y2 = self._raw_bbox
        width, height = image_size

        x_min = int(round(min(x1, x2)))
        x_max = int(round(max(x1, x2)))
        y_min = int(round(min(y1, y2)))
        y_max = int(round(max(y1, y2)))

        x_min = max(0, min(width - 1, x_min))
        y_min = max(0, min(height - 1, y_min))
        x_max = max(x_min + 1, min(width, x_max))
        y_max = max(y_min + 1, min(height, y_max))
        return x_min, y_min, x_max, y_max


def _collect_bbox_with_gui(image, title: str) -> Optional[Tuple[int, int, int, int]]:
    selector = _InteractiveBBoxSelector(np.asarray(image), title=title)
    selector.show()
    return selector.get_bounds(image.size)


def _collect_bbox_via_text(
    image_size: Tuple[int, int],
    input_mode: str,
    label: str,
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float], str]]:
    width, height = image_size
    detected_mode = input_mode
    while True:
        raw_text = _prompt_bbox_string(label)
        if not raw_text:
            print("Empty input detected. Please provide coordinates.")
            continue
        try:
            parsed = _parse_bbox_values(raw_text)
        except ValueError as exc:
            print(f"{exc}")
            continue
        if parsed is None:
            return None
        try:
            bounds = _convert_to_pixel_bounds(parsed, (width, height), input_mode)
        except ValueError as exc:
            print(f"{exc}")
            continue
        x_min, y_min, x_max, y_max, detected_mode = bounds
        norm_bounds = _normalize_bounds((x_min, y_min, x_max, y_max), (width, height))
        print(
            f"Proposed {label} region (detected mode: {detected_mode}):\n"
            f"  Pixel bounds: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}\n"
            f"  Normalized:  x_min={norm_bounds[0]:.4f}, y_min={norm_bounds[1]:.4f}, "
            f"x_max={norm_bounds[2]:.4f}, y_max={norm_bounds[3]:.4f}\n"
            f"  Width={x_max - x_min} px, Height={y_max - y_min} px"
        )
        confirm = input("Use this region? [y/n]: ").strip().lower()
        if confirm in {"y", "yes"}:
            return (x_min, y_min, x_max, y_max), norm_bounds, detected_mode
        print("Let's try again.")


def _build_output_paths(
    image_path: Path,
    output_dir: Path,
    mask_prefix: str,
    *,
    unique_output: bool,
    timestamp: Optional[str] = None,
) -> Tuple[Path, Path, str]:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    if unique_output:
        base_name = f"{image_path.stem}_{mask_prefix}_{stamp}"
    else:
        base_name = mask_prefix
    mask_path = output_dir / f"{base_name}.npy"
    meta_path = output_dir / f"{base_name}.json"
    return mask_path, meta_path, stamp


def _collect_region(
    image,
    image_size: Tuple[int, int],
    input_mode: str,
    label: str,
    use_gui: bool,
) -> Tuple[Tuple[int, int, int, int], Tuple[float, float, float, float], str]:
    bounds = None
    normalized = None
    detected_mode = input_mode

    if use_gui:
        print(
            f"\nGUI MODE ({label}): A window will open. Drag to draw a rectangle, then press Enter to confirm.\n"
            "Press 'r' to reset the selection or Esc to cancel and fall back to text input."
        )
        gui_bounds = _collect_bbox_with_gui(image, title=f"{label} region")
        if gui_bounds is not None:
            bounds = gui_bounds
            normalized = _normalize_bounds(gui_bounds, image_size)
            detected_mode = "pixel"
            print(
                f"Selected {label} bounds from GUI: "
                f"x_min={gui_bounds[0]}, y_min={gui_bounds[1]}, x_max={gui_bounds[2]}, y_max={gui_bounds[3]}"
            )
        else:
            print("GUI selection cancelled. Falling back to textual input.")

    if bounds is None or normalized is None:
        print(
            f"\nTEXT MODE ({label}): Enter bounding box coordinates when prompted. Example formats:\n"
            "  pixel mode     ->  120, 200, 420, 520\n"
            "  normalized     ->  0.25, 0.40, 0.55, 0.80\n"
            "Type 'q' to abort."
        )
        text_result = _collect_bbox_via_text(image_size, input_mode, label)
        if text_result is None:
            raise RuntimeError("No region confirmed.")
        bounds, normalized, detected_mode = text_result

    return bounds, normalized, detected_mode


def _normalize_path_for_platform(path: str) -> str:
    release = platform.uname().release.lower()
    if "microsoft" in release:
        return fix_wsl_paths(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively collect a single region mask from a bounding box."
    )
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH, help="Path to the source image.")
    parser.add_argument(
        "--output-dir",
        default="region_masks",
        help="Directory to store the generated mask and metadata.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["auto", "pixel", "normalized"],
        default="auto",
        help="Coordinate system for input values.",
    )
    parser.add_argument(
        "--mask-prefix",
        default="active_region_mask",
        help="Base filename for the mask when --unique-output is not used.",
    )
    parser.add_argument(
        "--unique-output",
        action="store_true",
        help="Append image stem and timestamp to output names.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing mask when --unique-output is not used.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable GUI selection and use textual coordinate entry instead.",
    )
    args = parser.parse_args()

    normalized_image_path = _normalize_path_for_platform(args.image_path)
    normalized_output_dir = _normalize_path_for_platform(args.output_dir)

    image_path = Path(normalized_image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    output_dir = Path(normalized_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(str(image_path))
    width, height = image.size
    print(f"Loaded image: {image_path}")
    print(f"Image dimensions: width={width}, height={height}")

    image_size = (width, height)
    bounds, norm_bounds, detected_mode = _collect_region(
        image=image,
        image_size=image_size,
        input_mode=args.input_mode,
        label="ACTIVE",
        use_gui=not args.no_gui,
    )

    mask = _create_mask(image_size, bounds)

    mask_path, meta_path, stamp = _build_output_paths(
        image_path,
        output_dir,
        args.mask_prefix,
        unique_output=args.unique_output,
    )

    if not args.unique_output and not args.overwrite:
        for candidate in (mask_path, meta_path):
            if candidate.exists():
                raise FileExistsError(
                    f"Output file already exists: {candidate}. Use --overwrite or --unique-output."
                )

    np.save(mask_path, mask.astype(np.uint8))

    metadata = {
        "image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "input_mode": args.input_mode,
        "timestamp": datetime.now().isoformat(),
        "run_id": stamp,
        "role": "active",
        "mask_path": str(mask_path),
        "detected_mode": detected_mode,
        "pixel_bounds": {
            "x_min": bounds[0],
            "y_min": bounds[1],
            "x_max": bounds[2],
            "y_max": bounds[3],
        },
        "normalized_bounds": {
            "x_min": norm_bounds[0],
            "y_min": norm_bounds[1],
            "x_max": norm_bounds[2],
            "y_max": norm_bounds[3],
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n✅ Single region mask created successfully!")
    print(f"Mask saved to: {mask_path}")
    print(f"Metadata: {meta_path}")
    print("\nUse this path with run_single_region_extraction.py via --mask-path.")


if __name__ == "__main__":
    main()
