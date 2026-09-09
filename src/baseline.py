import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import glob

# ==============================================================================
# Panorama Generation Pipeline
# Core idea adapted from: Kaggle "Panorama Image Stitcher using OpenCV Python"
#
# Upgrade over baseline (ORB + BruteForce):
#   - SIFT  : Scale-Invariant Feature Transform  → richer float descriptors
#   - FLANN : Fast Library for Approximate Nearest Neighbors → faster matching
#   - Lowe's Ratio Test : filters ambiguous matches (ratio < 0.7)
#   - Multi-image stitching : iteratively stitches N images in a directory
# ==============================================================================


# ------------------------------------------------------------------------------
# Helper: Compute homography + warp img2 to img1's perspective
# ------------------------------------------------------------------------------
def warpImages(img1, img2, H):
    """
    Warp img2 onto img1 using homography matrix H.

    Parameters
    ----------
    img1 : ndarray  – the "base" image (already-stitched collage or first image)
    img2 : ndarray  – the new image to be warped and added
    H    : ndarray  – 3×3 homography matrix mapping img1 -> img2

    Returns
    -------
    output_img : ndarray  – stitched panorama
    """
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # Corner points of img1
    pts1 = np.float32([
        [0, 0], [0, rows1], [cols1, rows1], [cols1, 0]
    ]).reshape(-1, 1, 2)

    # Corner points of img2 projected through H
    pts2 = np.float32([
        [0, 0], [0, rows2], [cols2, rows2], [cols2, 0]
    ]).reshape(-1, 1, 2)
    pts2_transformed = cv2.perspectiveTransform(pts2, H)

    # Bounding box of both images in the common frame
    all_pts = np.concatenate((pts1, pts2_transformed), axis=0)
    [x_min, y_min] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    # Translation to shift everything into positive coordinates
    t = [-x_min, -y_min]
    H_t = np.array([[1, 0, t[0]],
                    [0, 1, t[1]],
                    [0, 0,    1]], dtype=np.float64)

    # Warp img2 onto the canvas
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    output_img = cv2.warpPerspective(img2, H_t.dot(H), (canvas_w, canvas_h))

    # Paste img1 onto the canvas (overwrites warped area with clean base image)
    output_img[t[1]:rows1 + t[1], t[0]:cols1 + t[0]] = img1

    return output_img


# ------------------------------------------------------------------------------
# Core: SIFT + FLANN + Lowe's Ratio Test + RANSAC homography
# ------------------------------------------------------------------------------
def warp(img1, img2, min_match_count=10):
    """
    Stitch img2 next to img1.

    Steps
    -----
    1. Detect SIFT keypoints & descriptors in both images.
    2. Match with FLANN (k-NN, k=2).
    3. Filter matches using Lowe's ratio test (ratio < 0.7).
    4. Compute homography with RANSAC.
    5. Warp and blend via warpImages().

    Parameters
    ----------
    img1            : ndarray  – left/base image
    img2            : ndarray  – right/new image to append
    min_match_count : int      – minimum good matches required (default 10)

    Returns
    -------
    stitched : ndarray or None
    """
    sift = cv2.SIFT_create()

    # Step 1 – Detect & compute SIFT features
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Step 2 – FLANN matcher (KD-Tree, 5 trees, 50 checks)
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Step 3 – Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) < min_match_count:
        print(f"  Not enough good matches: {len(good_matches)} / {min_match_count} required.")
        return None

    print(f"  Good matches found: {len(good_matches)}")

    # Step 4 – Extract point correspondences
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Step 5 – RANSAC homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        print("  Homography could not be computed.")
        return None

    # Step 6 – Warp and stitch
    return warpImages(img2, img1, H)


# ------------------------------------------------------------------------------
# Save helper
# ------------------------------------------------------------------------------
def save_image(directory, file_name, image):
    """Save an image to directory/file_name, creating the directory if needed."""
    os.makedirs(directory, exist_ok=True)
    cv2.imwrite(os.path.join(directory, file_name), image)


# ------------------------------------------------------------------------------
# Multi-image stitching: stitch all images in a directory sequentially
# ------------------------------------------------------------------------------
def stitch_images_in_directory(input_directory, output_directory):
    """
    Iteratively stitch every image in input_directory (sorted order) into a
    single panorama and save it to output_directory.

    Parameters
    ----------
    input_directory  : str – folder containing source images
    output_directory : str – folder where the stitched result is saved
    """
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    filenames = []
    for ext in exts:
        filenames.extend(glob.glob(os.path.join(input_directory, ext)))
    filenames = sorted(filenames)

    if len(filenames) < 2:
        print(f"  Need at least 2 images in '{input_directory}', found {len(filenames)}.")
        return

    print(f"  Stitching {len(filenames)} images from: {input_directory}")

    collage = cv2.imread(filenames[0])
    for i in range(1, len(filenames)):
        img_next = cv2.imread(filenames[i])
        print(f"  [{i}/{len(filenames)-1}] Adding: {os.path.basename(filenames[i])}")
        result = warp(collage, img_next)
        if result is None:
            print(f"  Skipping image {os.path.basename(filenames[i])} – stitching failed.")
        else:
            collage = result

    out_name = 'panorama_' + os.path.basename(filenames[-1])
    save_image(output_directory, out_name, collage)
    print(f"  Saved stitched panorama -> {os.path.join(output_directory, out_name)}")
    return collage


# ------------------------------------------------------------------------------
# Two-image pipeline (original baseline behaviour, now upgraded)
# ------------------------------------------------------------------------------
def run_two_image_pipeline(data_dir, results_dir):
    """
    Original two-image baseline flow, upgraded to use SIFT + FLANN + Lowe's test.
    Reads image1.jpg (left) and image2.jpg (right) from data_dir.
    """
    img1_path = os.path.join(data_dir, 'image1.jpeg')
    img2_path = os.path.join(data_dir, 'image2.jpeg')

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print(f"Error: Could not load images from '{data_dir}'.")
        print("  Ensure image1.jpeg and image2.jpeg exist in the data/ folder.")
        return

    print(f"Loaded images  –  img1: {img1.shape}, img2: {img2.shape}")

    # ── Feature Matching Visualization ──────────────────────────────────────
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    print(f"Keypoints  –  img1: {len(kp1)}, img2: {len(kp2)}")

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )
    matches   = flann.knnMatch(des1, des2, k=2)
    good      = [m for m, n in matches if m.distance < 0.7 * n.distance]
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good[:60], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    match_path = os.path.join(results_dir, 'feature_matches_sift.jpg')
    cv2.imwrite(match_path, match_img)
    print(f"Saved feature match visualization -> {match_path}")

    # ── Stitch ───────────────────────────────────────────────────────────────
    print("Stitching images...")
    panorama = warp(img1, img2)
    if panorama is None:
        print("Stitching failed.")
        return

    pano_path = os.path.join(results_dir, 'panorama_sift.jpg')
    cv2.imwrite(pano_path, panorama)
    print(f"Saved final panorama -> {pano_path}")
    print("Done.")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------
def main():
    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir    = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # ── Mode selection ────────────────────────────────────────────────────────
    # Set MULTI_IMAGE = True to stitch every image in data_dir sequentially.
    # Set MULTI_IMAGE = False to run the classic two-image (image1 + image2) flow.
    MULTI_IMAGE = False

    if MULTI_IMAGE:
        print("=== Multi-Image Panorama Pipeline (SIFT + FLANN) ===")
        stitch_images_in_directory(data_dir, results_dir)
    else:
        print("=== Two-Image Panorama Pipeline (SIFT + FLANN) ===")
        run_two_image_pipeline(data_dir, results_dir)


if __name__ == "__main__":
    main()
