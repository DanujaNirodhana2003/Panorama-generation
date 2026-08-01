import cv2
import numpy as np
import os

# ==============================================================================
# Advanced Panorama Generation Pipeline (SIFT + FLANN + Strict RANSAC)
# ==============================================================================

def main():
    print("Starting SIFT Panorama Pipeline...")
    
    # Paths to data and results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    
    img1_path = os.path.join(data_dir, 'image1.jpg') # Left image
    img2_path = os.path.join(data_dir, 'image2.jpg') # Right image
    
    # Step 1: Load and Resize the Images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not load images from {data_dir}. Please ensure image1.jpg and image2.jpg exist.")
        return

    def resize_img(img, max_width=1024):
        h, w = img.shape[:2]
        if w > max_width:
            ratio = max_width / w
            return cv2.resize(img, (max_width, int(h * ratio)))
        return img
        
    img1 = resize_img(img1)
    img2 = resize_img(img2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Step 2: Initialize the SIFT Feature Detector
    sift = cv2.SIFT_create()

    # Step 3: Detect and Compute SIFT Features
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    print(f"Found {len(keypoints1)} SIFT keypoints in image1 and {len(keypoints2)} in image2.")

    # Step 4: Initialize the FLANN Feature Matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Step 5: Match the Features using KNN
    raw_matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Step 6: Apply a Strict Lowe's Ratio Test
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.6 * n.distance: # 0.6 is a strict threshold for real photos
            good_matches.append(m)
            
    print(f"Found {len(good_matches)} good matches after Lowe's Ratio Test (0.6 threshold).")

    # Step 7: Extract Coordinates for Homography
    if len(good_matches) < 4:
        print("Not enough matches to compute homography. Try taking photos with more overlap.")
        return

    src_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Step 8: Calculate Homography with Strict RANSAC
    # Using a 3.0 pixel threshold to completely eliminate geometric outliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    if H is None:
        print("Homography could not be calculated. The matches were too noisy.")
        return
        
    # Step 9: Filter Matches and Visualize
    matches_mask = mask.ravel().tolist()
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i] == 1]
    print(f"Found {len(inlier_matches)} perfect inliers after strict RANSAC filtering.")

    # Draw ONLY the verified inliers
    match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, inlier_matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    match_save_path = os.path.join(results_dir, 'sift_matches.jpg')
    cv2.imwrite(match_save_path, match_img)
    print(f"Saved flawless SIFT matches visualization to {match_save_path}")

    # Step 10: Calculate the Panorama Canvas Size
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    print("Warping and blending images...")
    
    # Step 11: Warp Image2 and Blend
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    roi = result[t[1]:h1+t[1], t[0]:w1+t[0]]
    mask_empty = (roi == 0).all(axis=2)
    roi[mask_empty] = img1[mask_empty]

    # Step 12: Save the Final Panorama
    pano_save_path = os.path.join(results_dir, 'sift_panorama.jpg')
    cv2.imwrite(pano_save_path, result)
    print(f"Saved final panorama to {pano_save_path}")
    print("SIFT execution complete!")

if __name__ == "__main__":
    main()
