import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# Baseline Panorama Generation Pipeline (ORB + Brute-Force Matcher)
# ==============================================================================

def main():
    print("Starting ORB Baseline Pipeline...")
    
    # Paths to data and results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    
    img1_path = os.path.join(data_dir, 'image1.jpg') # Left image
    img2_path = os.path.join(data_dir, 'image2.jpg') # Right image
    
    # Step 1: Load the overlapping images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not load images from {data_dir}. Please ensure image1.jpg and image2.jpg exist.")
        return

    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Step 2: Initialize the ORB Feature Detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Step 3: Detect and Compute Features
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    print(f"Found {len(keypoints1)} keypoints in image1 and {len(keypoints2)} keypoints in image2.")

    # Step 4: Initialize the Feature Matcher
    # NORM_HAMMING is used for ORB since descriptors are binary
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Step 5: Match the Features
    matches = bf.match(descriptors1, descriptors2)
    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"Found {len(matches)} matches.")

    # Step 6: Visualize the Best Matches
    # Draw top 50 matches for the visualization slide
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    match_save_path = os.path.join(results_dir, 'feature_matches.jpg')
    cv2.imwrite(match_save_path, match_img)
    print(f"Saved feature matching visualization to {match_save_path}")

    # Step 7: Extract Matching Coordinates for Homography
    if len(matches) < 4:
        print("Not enough matches to compute homography.")
        return

    # We use the top 100 matches to calculate Homography for better accuracy
    top_matches = matches[:100]
    
    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)

    # Step 8: Calculate the Homography Matrix
    # Using RANSAC to filter out remaining bad matches
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Homography could not be calculated.")
        return

    # Step 9: Warp Image2 to Align with Image1
    # Get dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Calculate the size of the new panorama canvas
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    # Translation matrix to shift the image so it's not cropped
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    print("Warping images...")
    # Warp image2
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    # Step 10: Blend the Images Together
    # Place image1 onto the canvas
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1

    # Step 11: Save the Final Panorama
    pano_save_path = os.path.join(results_dir, 'baseline_panorama.jpg')
    cv2.imwrite(pano_save_path, result)
    print(f"Saved final panorama to {pano_save_path}")
    print("Baseline execution complete!")

if __name__ == "__main__":
    main()
