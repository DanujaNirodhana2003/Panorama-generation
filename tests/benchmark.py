"""
tests/benchmark.py
==================
Full pipeline benchmark suite.

Runs the pipeline across the self-collected dataset and the synthetic 4-frame
sequence, logging per-pair metrics to outputs/metrics/benchmark_m4.csv.

Metrics logged:
  - pair_label       : image pair name
  - good_matches     : matches after Lowe's ratio test
  - inliers          : RANSAC inliers
  - inlier_ratio_pct : inlier %
  - rmse_px          : reprojection RMSE
  - time_s           : total wall time for the pair
  - blend_mode       : blending mode used
  - fallback         : True if affine fallback triggered

Usage
-----
    python tests/benchmark.py
    python tests/benchmark.py --data path/to/images --blend feather
"""

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import evaluate_pair, seam_intensity_metric
from src.features.extractor  import extract_sift
from src.matching.matcher    import flann_match, extract_point_pairs
from src.geometry.homography import estimate_homography
from src.stitching.warper    import warp_images


def benchmark_directory(data_dir: str, blend_mode: str = "overwrite",
                         out_csv: str | None = None) -> list[dict]:
    """
    Run the full pipeline on every consecutive image pair in data_dir.

    Returns a list of metric dicts, one per pair.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    filenames = []
    for ext in exts:
        filenames.extend(glob.glob(os.path.join(data_dir, ext)))
    filenames = sorted(filenames)

    if len(filenames) < 2:
        print(f"  Need at least 2 images in {data_dir}, found {len(filenames)}.")
        return []

    rows = []
    cfg = {
        "ratio_thresh": 0.70, "min_matches": 10,
        "ransac_thresh": 5.0, "stretch_limit": 1.3,
        "blend_mode": blend_mode, "compensate_exposure": False,
    }

    print(f"\n{'─'*60}")
    print(f"  Dataset : {data_dir}  ({len(filenames)} images)")
    print(f"  Blend   : {blend_mode}")
    print(f"{'─'*60}")

    collage = cv2.imread(filenames[0])
    for i in range(1, len(filenames)):
        img_next = cv2.imread(filenames[i])
        label = f"{os.path.basename(filenames[i-1])}→{os.path.basename(filenames[i])}"

        t0 = time.perf_counter()
        kp1, des1 = extract_sift(collage)
        kp2, des2 = extract_sift(img_next)
        good, _   = flann_match(des1, des2, ratio_thresh=cfg["ratio_thresh"])

        if len(good) < cfg["min_matches"]:
            print(f"  SKIP  {label} : only {len(good)} matches")
            rows.append({"pair_label": label, "good_matches": len(good),
                         "inliers": 0, "inlier_ratio_pct": 0,
                         "rmse_px": float("nan"), "time_s": 0,
                         "blend_mode": blend_mode, "fallback": False,
                         "seam_score": float("nan")})
            continue

        src_pts, dst_pts = extract_point_pairs(kp1, kp2, good)
        H, mask, fallback = estimate_homography(
            src_pts, dst_pts, img_next.shape,
            stretch_limit=cfg["stretch_limit"]
        )

        n_inliers = int(mask.ravel().sum()) if mask is not None else 0
        ratio_pct = n_inliers / max(len(good), 1) * 100

        rmse = float("nan")
        if H is not None and mask is not None and n_inliers > 0:
            from src.evaluation.metrics import compute_rmse
            inl_src = src_pts[mask.ravel() == 1]
            inl_dst = dst_pts[mask.ravel() == 1]
            rmse = compute_rmse(inl_src, inl_dst, H)

        panorama = None
        seam_score = float("nan")
        if H is not None:
            panorama = warp_images(img_next, collage, H, blend_mode=blend_mode)
            if panorama is not None and panorama.ndim == 3:
                seam_score = seam_intensity_metric(panorama)
                collage = panorama

        t1 = time.perf_counter()

        row = {
            "pair_label":      label,
            "good_matches":    len(good),
            "inliers":         n_inliers,
            "inlier_ratio_pct": f"{ratio_pct:.2f}",
            "rmse_px":         f"{rmse:.4f}" if not np.isnan(rmse) else "nan",
            "time_s":          f"{t1-t0:.4f}",
            "blend_mode":      blend_mode,
            "fallback":        fallback,
            "seam_score":      f"{seam_score:.3f}" if not np.isnan(seam_score) else "nan",
        }
        rows.append(row)
        fb_tag = " [FALLBACK]" if fallback else ""
        print(f"  {label}:  matches={len(good)}  inliers={n_inliers} ({ratio_pct:.1f}%)"
              f"  RMSE={rmse:.3f}px  t={t1-t0:.3f}s  seam={seam_score:.2f}{fb_tag}")

    if out_csv and rows:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  CSV saved → {out_csv}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Panorama pipeline benchmark")
    parser.add_argument("--data",  default=None, help="Image directory (default: data/raw/)")
    parser.add_argument("--blend", choices=["overwrite", "feather", "multiband"],
                        default="overwrite")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data or os.path.join(base_dir, "data", "raw")
    out_csv  = os.path.join(base_dir, "outputs", "metrics",
                            f"benchmark_{args.blend}.csv")

    benchmark_directory(data_dir, blend_mode=args.blend, out_csv=out_csv)


if __name__ == "__main__":
    main()
