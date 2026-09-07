"""
src/pipeline.py
===============
Top-level panorama stitching orchestrator.

Replaces the monolithic src/baseline.py __main__ block.
All pipeline stages are now separate, importable modules:

  features/extractor.py    → SIFT / ORB detection
  matching/matcher.py      → FLANN + Lowe's ratio test
  geometry/homography.py   → RANSAC + M4 affine fallback
  stitching/warper.py      → canvas allocation, warping, blending
  blending/feathering.py   → distance-based feathering (M4)
  blending/multiband.py    → Laplacian pyramid blending (M4)
  bundle_adjustment/       → global pose refinement (M4)
  evaluation/metrics.py    → RMSE, inlier ratio, seam metric

Usage
-----
    python src/pipeline.py                    # two-image, default (overwrite) blend
    python src/pipeline.py --blend feather    # two-image, feathered blend
    python src/pipeline.py --blend multiband  # two-image, pyramid blend
    python src/pipeline.py --multi            # all images in data/raw/ sequentially
    python src/pipeline.py --multi --ba       # + global bundle adjustment
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.extractor import extract_sift
from src.matching.matcher   import flann_match, extract_point_pairs
from src.geometry.homography import estimate_homography
from src.stitching.warper   import warp_images
from src.evaluation.metrics import evaluate_pair


# ---------------------------------------------------------------------------
# Config defaults (override with configs/default.yaml if present)
# ---------------------------------------------------------------------------
MIN_MATCH_COUNT = 10
RATIO_THRESH    = 0.70
RANSAC_THRESH   = 5.0
STRETCH_LIMIT   = 1.3


def _load_config(cfg_path: str) -> dict:
    """Load YAML config if present; otherwise return defaults."""
    cfg = {
        "ratio_thresh":   RATIO_THRESH,
        "min_matches":    MIN_MATCH_COUNT,
        "ransac_thresh":  RANSAC_THRESH,
        "stretch_limit":  STRETCH_LIMIT,
        "blend_mode":     "overwrite",
        "pyramid_levels": 5,
        "compensate_exposure": False,
    }
    if os.path.isfile(cfg_path):
        try:
            import yaml
            with open(cfg_path) as f:
                loaded = yaml.safe_load(f)
            if loaded:
                cfg.update(loaded)
        except ImportError:
            print("  [Config] pyyaml not installed — using built-in defaults.")
    return cfg


def stitch_pair(img1: np.ndarray, img2: np.ndarray,
                cfg: dict,
                pair_label: str = "pair",
                log_path: str | None = None) -> tuple[np.ndarray | None, dict]:
    """
    Stitch img2 onto img1 and return (stitched_image, metrics_dict).

    Returns (None, {}) if stitching fails.
    """
    kp1, des1 = extract_sift(img1)
    kp2, des2 = extract_sift(img2)

    good, _ = flann_match(des1, des2,
                          ratio_thresh=cfg["ratio_thresh"])

    print(f"  [{pair_label}] Good matches: {len(good)}")
    if len(good) < cfg["min_matches"]:
        print(f"  [{pair_label}] Not enough matches ({len(good)} < {cfg['min_matches']}).")
        return None, {}

    src_pts, dst_pts = extract_point_pairs(kp1, kp2, good)

    H, mask, fallback = estimate_homography(
        src_pts, dst_pts, img2.shape,
        ransac_reproj_thresh=cfg["ransac_thresh"],
        stretch_limit=cfg["stretch_limit"],
        log_path=log_path,
        pair_label=pair_label,
    )

    if H is None:
        print(f"  [{pair_label}] Homography could not be computed.")
        return None, {}

    n_inliers = int(mask.ravel().sum()) if mask is not None else 0
    inlier_ratio = n_inliers / max(len(good), 1) * 100

    stitched = warp_images(img2, img1, H,
                           blend_mode=cfg["blend_mode"],
                           compensate_exposure=cfg["compensate_exposure"])

    metrics = {
        "pair":            pair_label,
        "good_matches":    len(good),
        "inliers":         n_inliers,
        "inlier_ratio_pct": inlier_ratio,
        "fallback":        fallback,
    }
    return stitched, metrics


def run_two_image(data_dir: str, out_dir: str, cfg: dict,
                  log_path: str | None = None):
    """Run the two-image pipeline (image1.jpeg + image2.jpeg)."""
    img1 = cv2.imread(os.path.join(data_dir, "image1.jpeg"))
    img2 = cv2.imread(os.path.join(data_dir, "image2.jpeg"))
    if img1 is None or img2 is None:
        print(f"Error: Could not load images from '{data_dir}'.")
        print("  Ensure data/raw/image1.jpeg and data/raw/image2.jpeg exist.")
        return

    print(f"Loaded  img1: {img1.shape}   img2: {img2.shape}")

    # Feature-match visualization
    kp1, des1 = extract_sift(img1)
    kp2, des2 = extract_sift(img2)
    print(f"Keypoints  img1: {len(kp1)}   img2: {len(kp2)}")

    good, _ = flann_match(des1, des2, ratio_thresh=cfg["ratio_thresh"])
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good[:60], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    os.makedirs(out_dir, exist_ok=True)
    blend_tag = cfg["blend_mode"]
    match_out = os.path.join(out_dir, f"feature_matches_{blend_tag}.jpg")
    cv2.imwrite(match_out, match_img)
    print(f"Saved match visualization → {match_out}")

    panorama, metrics = stitch_pair(img1, img2, cfg,
                                    pair_label="img1→img2",
                                    log_path=log_path)
    if panorama is None:
        print("Stitching failed.")
        return

    pano_out = os.path.join(out_dir, f"panorama_{blend_tag}.jpg")
    cv2.imwrite(pano_out, panorama)
    print(f"Saved panorama → {pano_out}")
    print(f"Metrics: {metrics}")


def run_multi_image(data_dir: str, out_dir: str, cfg: dict,
                    run_ba: bool = False,
                    log_path: str | None = None):
    """Iteratively stitch all images in data_dir, optionally with bundle adjustment."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    filenames = []
    for ext in exts:
        filenames.extend(glob.glob(os.path.join(data_dir, ext)))
    filenames = sorted(filenames)

    if len(filenames) < 2:
        print(f"Need at least 2 images in '{data_dir}', found {len(filenames)}.")
        return

    print(f"Multi-image stitch: {len(filenames)} images")

    pairwise_data = []
    collage = cv2.imread(filenames[0])

    for i in range(1, len(filenames)):
        img_next = cv2.imread(filenames[i])
        label = f"img{i-1}→img{i}"
        print(f"  [{i}/{len(filenames)-1}] {os.path.basename(filenames[i])}")

        # Collect pairwise data for bundle adjustment
        kp1, des1 = extract_sift(collage)
        kp2, des2 = extract_sift(img_next)
        good, _ = flann_match(des1, des2, ratio_thresh=cfg["ratio_thresh"])

        if len(good) >= cfg["min_matches"]:
            src_pts, dst_pts = extract_point_pairs(kp1, kp2, good)
            H, mask, fallback = estimate_homography(
                src_pts, dst_pts, img_next.shape,
                ransac_reproj_thresh=cfg["ransac_thresh"],
                stretch_limit=cfg["stretch_limit"],
                log_path=log_path, pair_label=label,
            )
            pairwise_data.append({
                "H": H, "mask": mask, "src_pts": src_pts, "dst_pts": dst_pts,
                "label": label,
            })
            if H is not None:
                collage = warp_images(img_next, collage, H,
                                      blend_mode=cfg["blend_mode"],
                                      compensate_exposure=cfg["compensate_exposure"])
            else:
                print(f"  Skipping {os.path.basename(filenames[i])} — H failed.")
        else:
            print(f"  Skipping {os.path.basename(filenames[i])} — too few matches.")

    # Optional bundle adjustment pass
    if run_ba and len(pairwise_data) > 1:
        from src.bundle_adjustment.optimizer import run_bundle_adjustment
        print("\n=== Bundle Adjustment ===")
        valid = [pd for pd in pairwise_data if pd["H"] is not None]
        refined_Hs, report = run_bundle_adjustment(valid)

        # Re-stitch with refined homographies
        collage = cv2.imread(filenames[0])
        for i, (rH, pd) in enumerate(zip(refined_Hs, valid)):
            img_next = cv2.imread(filenames[i + 1])
            collage = warp_images(img_next, collage, rH,
                                  blend_mode=cfg["blend_mode"],
                                  compensate_exposure=cfg["compensate_exposure"])

        # Save BA report
        ba_log = os.path.join(os.path.dirname(out_dir), "metrics", "ba_report.txt")
        os.makedirs(os.path.dirname(ba_log), exist_ok=True)
        with open(ba_log, "w") as f:
            f.write("Bundle Adjustment Report\n")
            f.write("========================\n")
            for label, before, after in zip(
                [pd["label"] for pd in valid],
                report["before_rmse"], report["after_rmse"]
            ):
                f.write(f"{label}: {before:.4f} → {after:.4f} px\n")
        print(f"BA report saved → {ba_log}")

    os.makedirs(out_dir, exist_ok=True)
    blend_tag = cfg["blend_mode"]
    out_name  = f"panorama_multi_{blend_tag}.jpg"
    cv2.imwrite(os.path.join(out_dir, out_name), collage)
    print(f"Saved multi-image panorama → {os.path.join(out_dir, out_name)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Panorama stitching pipeline")
    parser.add_argument("--blend", choices=["overwrite", "feather", "multiband"],
                        default=None, help="Blend mode (overrides config)")
    parser.add_argument("--multi", action="store_true",
                        help="Stitch all images in data/raw/ sequentially")
    parser.add_argument("--ba", action="store_true",
                        help="Run global bundle adjustment after multi-image stitch")
    parser.add_argument("--data", default=None,
                        help="Path to image directory (default: data/raw/)")
    parser.add_argument("--out", default=None,
                        help="Path to output directory (default: outputs/m4/)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data or os.path.join(base_dir, "data", "raw")
    out_dir  = args.out  or os.path.join(base_dir, "outputs", "m4")
    cfg_path = os.path.join(base_dir, "configs", "default.yaml")
    log_path = os.path.join(base_dir, "outputs", "metrics", "fallback_log.csv")

    cfg = _load_config(cfg_path)
    if args.blend:
        cfg["blend_mode"] = args.blend

    print("=== Panorama Pipeline ===")
    print(f"  data     : {data_dir}")
    print(f"  output   : {out_dir}")
    print(f"  blend    : {cfg['blend_mode']}")

    if args.multi:
        print("  mode     : multi-image")
        run_multi_image(data_dir, out_dir, cfg, run_ba=args.ba, log_path=log_path)
    else:
        print("  mode     : two-image")
        run_two_image(data_dir, out_dir, cfg, log_path=log_path)


if __name__ == "__main__":
    main()
