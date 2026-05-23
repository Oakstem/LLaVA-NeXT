import argparse
import csv
import platform
import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from PIL import Image

try:
    project_root = Path(__file__).resolve().parent.parent
except NameError:
    project_root = Path.cwd().parent


DEFAULT_TRAIN_DIR = r"D:\Projects\data\gazefollow\train"


def _normalize_path_for_platform(path: str) -> str:
    if not isinstance(path, str):
        path = str(path)
    release = platform.uname().release.lower()
    if "microsoft" not in release or path.startswith("/mnt/"):
        return path
    normalized = path.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        return f"/mnt/{normalized[0].lower()}/{normalized[3:]}"
    return path


def _create_mask(image_size: tuple[int, int], bounds: tuple[int, int, int, int]) -> np.ndarray:
    width, height = image_size
    x_min, y_min, x_max, y_max = bounds
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 1
    return mask


class _InteractiveBBoxSelector:
    def __init__(self, image_array: np.ndarray, title: str):
        self.image_array = image_array
        self._raw_bbox: Optional[tuple[float, float, float, float]] = None
        self.confirmed = False
        self.skipped = False

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image_array)
        self.ax.set_title(
            f"{title}\n"
            "Drag to select a region.\n"
            "Enter: confirm  Space: skip  r: reset  Esc: cancel"
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
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.fig.canvas.mpl_connect("close_event", self._on_close)

    def _on_select(self, eclick, erelease):
        if eclick.xdata is None or eclick.ydata is None:
            return
        if erelease.xdata is None or erelease.ydata is None:
            return
        self._raw_bbox = (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata)

    def _on_key_press(self, event):
        key = event.key.lower() if event.key else ""
        if key in {"enter", "return"}:
            if self._raw_bbox is not None:
                self.confirmed = True
                plt.close(self.fig)
        elif key == " ":
            self.skipped = True
            self._raw_bbox = None
            plt.close(self.fig)
        elif key == "r":
            self._raw_bbox = None
            self.selector.set_active(True)
        elif key in {"escape", "esc", "q"}:
            self._raw_bbox = None
            plt.close(self.fig)

    def _on_close(self, _):
        if not self.confirmed:
            self._raw_bbox = None

    def show(self) -> None:
        plt.show()

    def get_result(self, image_size: tuple[int, int]) -> tuple[str, Optional[tuple[int, int, int, int]]]:
        if self.skipped:
            return "skip", None
        if not self.confirmed or self._raw_bbox is None:
            return "quit", None

        x1, y1, x2, y2 = self._raw_bbox
        width, height = image_size
        x_min = max(0, min(width - 1, int(round(min(x1, x2)))))
        y_min = max(0, min(height - 1, int(round(min(y1, y2)))))
        x_max = max(x_min + 1, min(width, int(round(max(x1, x2)))))
        y_max = max(y_min + 1, min(height, int(round(max(y1, y2)))))
        return "selected", (x_min, y_min, x_max, y_max)


def _iter_images(train_dir: Path) -> list[Path]:
    image_paths = [
        path
        for path in train_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    random.shuffle(image_paths)
    return image_paths


def _load_image(image_path: Path) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly iterate through images and collect one mask per selected image."
    )
    parser.add_argument("--train-dir", default=DEFAULT_TRAIN_DIR, help="Gazefollow train directory.")
    parser.add_argument("--output-dir", default="region_masks/random_samples", help="Mask output directory.")
    parser.add_argument(
        "--csv-path",
        default="region_masks/random_samples/selected_masks.csv",
        help="CSV file to write selected image and mask mappings.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="region_mask",
        help="Suffix used when naming saved mask files: <image_stem>_<mask_suffix>.npy",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on the number of selected masks to collect. 0 means all images until quit/exhaustion.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for image order.")
    args = parser.parse_args()

    random.seed(args.seed)

    train_dir = Path(_normalize_path_for_platform(args.train_dir)).expanduser().resolve()
    output_dir = Path(_normalize_path_for_platform(args.output_dir)).expanduser().resolve()
    csv_path = Path(_normalize_path_for_platform(args.csv_path)).expanduser().resolve()

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = _iter_images(train_dir)
    if not image_paths:
        raise RuntimeError(f"No images found under {train_dir}")

    rows: list[dict[str, str]] = []
    selected_count = 0
    skipped_count = 0

    print(f"Found {len(image_paths)} images under {train_dir}")
    print(f"Masks will be written to {output_dir}")
    print("Controls: drag to select, Enter to save mask, Space to skip image, Esc/q to stop.")

    for index, image_path in enumerate(image_paths, start=1):
        if args.limit > 0 and selected_count >= args.limit:
            break

        image = _load_image(image_path)
        selector = _InteractiveBBoxSelector(
            np.asarray(image),
            title=f"[{index}/{len(image_paths)}] {image_path.name}",
        )
        selector.show()
        action, bounds = selector.get_result(image.size)

        if action == "skip":
            skipped_count += 1
            print(f"[{index}/{len(image_paths)}] Skipped: {image_path}")
            continue

        if action == "quit":
            print(f"[{index}/{len(image_paths)}] Stopped by user.")
            break

        mask = _create_mask(image.size, bounds)
        mask_path = output_dir / f"{image_path.stem}_{args.mask_suffix}.npy"
        np.save(mask_path, mask.astype(np.uint8))

        rows.append(
            {
                "image_rel_path": image_path.relative_to(train_dir).as_posix(),
                "mask_rel_path": mask_path.relative_to(project_root).as_posix(),
            }
        )
        selected_count += 1
        print(f"[{index}/{len(image_paths)}] Saved: {mask_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_rel_path", "mask_rel_path"])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Collection finished.")
    print(f"Selected images: {selected_count}")
    print(f"Skipped images: {skipped_count}")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
