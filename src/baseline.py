import cv2
import numpy as np
import os

# ==============================================================================
# Baseline Panorama Generation Pipeline (ROI-Based ORB Matching)
# ==============================================================================

def main():
    print("Starting ROI-Based ORB Baseline Pipeline...")
    
    # Paths to data and results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    
    img1_path = os.path.join(data_dir, 'image1.jpg') # Left image
    img2_path = os.path.join(data_dir, 'image2.jpg') # Right image
    
    # Step 1: Load the images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print(f"Error: Could not load images from {data_dir}.")
        return

    # Step 2: Aggressive Resizing (Quality Drop)
    # We resize images to a fixed height of 600px to drop unnecessary detail
    def resize_to_height(img, target_height=600):
        h, w = img.shape[:2]
        ratio = target_height / h
        return cv2.resize(img, (int(w * ratio), target_height))
        
    img1 = resize_to_height(img1)
    img2 = resize_to_height(img2)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Step 3: Extract ROI (Regions of Interest)
    # We only want to look at the right 40% of image1 and the left 40% of image2
    roi_width1 = int(w1 * 0.4)
    roi_width2 = int(w2 * 0.4)
    
    # Crop the images to their overlapping regions
    roi1 = img1[:, w1 - roi_width1:] # Right side of img1
    roi2 = img2[:, :roi_width2]      # Left side of img2

    # Convert ROIs to grayscale for ORB
    gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # Step 4: Detect and Compute ORB Features ONLY in the ROIs
    orb = cv2.ORB_create(nfeatures=3000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_roi1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_roi2, None)
    print(f"Found {len(keypoints1)} keypoints in ROI 1 and {len(keypoints2)} in ROI 2.")

    # Step 5: Match the Features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Lowe's Ratio Test
    good_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
            
    print(f"Found {len(good_matches)} matches after Ratio Test.")

    if len(good_matches) < 4:
        print("Not enough matches to compute homography.")
        return

    # Step 6: Coordinate Shifting
    # Since we found features in cropped images, we must shift their X-coordinates 
    # back to match the original, full-sized images.
    
    src_pts = []
    dst_pts = []
    
    # For visualization, we will also create shifted keypoint objects
    shifted_keypoints1 = []
    shifted_keypoints2 = list(keypoints2) # roi2 starts at X=0, so no shift needed!
    
    for kp in keypoints1:
        # Shift the X coordinate by adding the width of the missing left part
        shifted_pt = (kp.pt[0] + (w1 - roi_width1), kp.pt[1])
        # Create a new keypoint with the shifted coordinate
        shifted_kp = cv2.KeyPoint(x=shifted_pt[0], y=shifted_pt[1], size=kp.size, 
                                  angle=kp.angle, response=kp.response, 
                                  octave=kp.octave, class_id=kp.class_id)
        shifted_keypoints1.append(shifted_kp)

    for m in good_matches:
        src_pts.append(shifted_keypoints2[m.trainIdx].pt)
        dst_pts.append(shifted_keypoints1[m.queryIdx].pt)
        
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    # Step 7: Calculate Homography with RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Homography could not be calculated.")
        return

    # Filter verified matches
    matches_mask = mask.ravel().tolist()
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i] == 1]
    print(f"Found {len(inlier_matches)} verified matches after RANSAC.")

    # Step 8: Visualize Matches on the FULL Images
    match_img = cv2.drawMatches(img1, shifted_keypoints1, img2, shifted_keypoints2, inlier_matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    match_save_path = os.path.join(results_dir, 'feature_matches.jpg')
    cv2.imwrite(match_save_path, match_img)
    print(f"Saved ROI matching visualization to {match_save_path}")

    # Step 9: Warp Image2 to Align with Image1
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    print("Warping and blending images...")
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    # Step 10: NumPy Mask Blending
    roi = result[t[1]:h1+t[1], t[0]:w1+t[0]]
    mask_empty = (roi == 0).all(axis=2)
    roi[mask_empty] = img1[mask_empty]

    # Step 11: Save the Final Panorama
    pano_save_path = os.path.join(results_dir, 'baseline_panorama.jpg')
    cv2.imwrite(pano_save_path, result)
    print(f"Saved final panorama to {pano_save_path}")
    print("ROI ORB execution complete!")

if __name__ == "__main__":
    main()
