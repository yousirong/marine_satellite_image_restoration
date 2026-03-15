from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image


TARGET_ROOT = Path("/home/juneyonglee/Desktop/AY_ust/My_Book/performance/2020")
GT_NAME = "gt.png"
MASK_NAME = "mask.png"
OUTPUT_NAME = "gt_masked.png"


def is_red_mask(mask_array: np.ndarray) -> np.ndarray:
    return (
        (mask_array[..., 0] > 150)
        & (mask_array[..., 1] < 100)
        & (mask_array[..., 2] < 100)
    )


def iter_date_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        path for path in root_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )


def apply_white_overlay(gt_path: Path, mask_path: Path, output_path: Path) -> int:
    with Image.open(gt_path) as gt_image, Image.open(mask_path) as mask_image:
        original_mode = gt_image.mode
        gt_rgba = gt_image.convert("RGBA")
        mask_rgba = mask_image.convert("RGBA")

        if gt_rgba.size != mask_rgba.size:
            raise ValueError(
                f"image size mismatch: gt={gt_rgba.size}, mask={mask_rgba.size}"
            )

        gt_array = np.array(gt_rgba, copy=True)
        mask_array = np.array(mask_rgba)

    red_pixels = is_red_mask(mask_array)
    gt_array[red_pixels] = np.array([255, 255, 255, 255], dtype=gt_array.dtype)

    result_image = Image.fromarray(gt_array, mode="RGBA")
    if original_mode != "RGBA":
        result_image = result_image.convert(original_mode)
    result_image.save(output_path, compress_level=1)

    return int(red_pixels.sum())


def main() -> int:
    if not TARGET_ROOT.is_dir():
        print(f"target root is not available: {TARGET_ROOT}", file=sys.stderr)
        return 1

    processed = 0
    total_red_pixels = 0
    problems: list[str] = []

    for date_dir in iter_date_dirs(TARGET_ROOT):
        gt_path = date_dir / GT_NAME
        mask_path = date_dir / MASK_NAME
        output_path = date_dir / OUTPUT_NAME

        missing_inputs = [
            str(path.name) for path in (gt_path, mask_path) if not path.is_file()
        ]
        if missing_inputs:
            problems.append(f"{date_dir}: missing {', '.join(missing_inputs)}")
            continue

        try:
            red_pixels = apply_white_overlay(gt_path, mask_path, output_path)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{date_dir}: {exc}")
            continue

        processed += 1
        total_red_pixels += red_pixels
        print(f"{date_dir.name}: wrote {output_path.name}", flush=True)

    print(f"total files written: {processed}")
    print(f"total red pixels converted: {total_red_pixels}")

    if problems:
        print("problems detected:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
