"""Benchmark harness for the preprocess crop pipeline.

Usage (run from repo root):
    python tests/benchmark_preprocess.py
    python tests/benchmark_preprocess.py --sizes 600 1080 2160 4320 --runs 30

Measures p50/p95/p99 latency, throughput, and peak RSS for each
(image type × resolution) combination.  Results are printed as a Markdown
table so they can be copy-pasted into documentation.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
import time
import tracemalloc
from typing import NamedTuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def _white_border(h: int, w: int) -> np.ndarray:
    """White scanner-bed border around a dark photo."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    b = max(20, min(h, w) // 12)
    img[b:h - b, b:w - b] = (55, 55, 55)
    # Add faint noise to the photo centre so it isn't a trivial solid block
    rng = np.random.RandomState(1)
    img[b:h - b, b:w - b] = np.clip(
        img[b:h - b, b:w - b].astype(np.int16) + rng.randint(-15, 15, (h - 2*b, w - 2*b, 3)),
        0, 255,
    ).astype(np.uint8)
    return img


def _color_border(h: int, w: int) -> np.ndarray:
    """Coloured mat border (pink/cream) around a darker photo."""
    img = np.full((h, w, 3), (170, 140, 200), dtype=np.uint8)
    b = max(20, min(h, w) // 10)
    rng = np.random.RandomState(2)
    img[b:h - b, b:w - b] = np.clip(
        np.full((h - 2*b, w - 2*b, 3), 60).astype(np.int16)
        + rng.randint(-20, 20, (h - 2*b, w - 2*b, 3)),
        0, 255,
    ).astype(np.uint8)
    return img


def _perspective_skew(h: int, w: int) -> np.ndarray:
    """Phone photo of a print — quadrilateral with slight skew."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    margin = min(h, w) // 8
    pts = np.array([
        [margin + margin // 3, margin],
        [w - margin, margin + margin // 4],
        [w - margin - margin // 3, h - margin],
        [margin, h - margin - margin // 4],
    ], dtype=np.int32)
    rng = np.random.RandomState(3)
    fill = np.clip(
        np.full((h, w, 3), 50).astype(np.int16)
        + rng.randint(-20, 20, (h, w, 3)),
        0, 255,
    ).astype(np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 255)
    img[mask > 0] = fill[mask > 0]
    return img


def _no_crop(h: int, w: int) -> np.ndarray:
    """Content fills the frame — should not be cropped."""
    rng = np.random.RandomState(4)
    return rng.randint(40, 180, (h, w, 3), dtype=np.uint8)


GENERATORS = {
    "white_border":   _white_border,
    "color_border":   _color_border,
    "perspective":    _perspective_skew,
    "no_crop":        _no_crop,
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

class Stats(NamedTuple):
    p50: float
    p95: float
    p99: float
    mean: float
    throughput_fps: float  # images/sec
    peak_mb: float


def _measure(fn, image: np.ndarray, runs: int = 25) -> Stats:
    gc.collect()
    # Warmup (JIT caches, import side-effects, etc.)
    for _ in range(3):
        fn(image)

    tracemalloc.start()
    times: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(image)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    times.sort()
    n = len(times)
    return Stats(
        p50=statistics.median(times),
        p95=times[min(n - 1, int(n * 0.95))],
        p99=times[min(n - 1, int(n * 0.99))],
        mean=statistics.mean(times),
        throughput_fps=1000.0 / statistics.mean(times),
        peak_mb=peak / 1_048_576,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_benchmark(
    sizes: list[int],
    image_types: list[str],
    runs: int,
    cfg_kwargs: dict | None = None,
) -> dict[str, Stats]:
    from romav2.preprocess import analyze_and_crop_image, PreprocConfig

    cfg_kwargs = cfg_kwargs or {}
    cfg = PreprocConfig(**cfg_kwargs)

    def fn(img):
        return analyze_and_crop_image(img, cfg=cfg)

    results: dict[str, Stats] = {}

    header = f"{'Image':<30} {'p50':>7} {'p95':>7} {'p99':>7} {'mean':>7} {'fps':>6} {'peak MB':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for itype in image_types:
        gen = GENERATORS[itype]
        for size in sizes:
            h = size
            w = int(size * 4 / 3)
            img = gen(h, w)
            key = f"{itype}_{h}p"
            s = _measure(fn, img, runs=runs)
            results[key] = s
            print(
                f"{key:<30} {s.p50:>7.1f} {s.p95:>7.1f} {s.p99:>7.1f} "
                f"{s.mean:>7.1f} {s.throughput_fps:>6.1f} {s.peak_mb:>8.1f}"
            )

    print(sep)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark romav2 preprocess pipeline")
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[600, 1080, 2160],
        help="Image heights to benchmark (width = height * 4/3)",
    )
    parser.add_argument(
        "--types", nargs="+",
        default=list(GENERATORS.keys()),
        choices=list(GENERATORS.keys()),
        help="Image types to benchmark",
    )
    parser.add_argument("--runs", type=int, default=25, help="Timing runs per combination")
    parser.add_argument(
        "--working-res", type=int, default=None,
        help="Override max_working_res config (0 = disabled)",
    )
    args = parser.parse_args()

    cfg_kwargs: dict = {}
    if args.working_res is not None:
        cfg_kwargs["max_working_res"] = args.working_res

    print(f"\nRoMaV2 Preprocess Benchmark  (runs={args.runs}, sizes={args.sizes})")
    print(f"All times in milliseconds.  fps = images/sec.\n")
    run_benchmark(
        sizes=args.sizes,
        image_types=args.types,
        runs=args.runs,
        cfg_kwargs=cfg_kwargs,
    )


if __name__ == "__main__":
    main()
