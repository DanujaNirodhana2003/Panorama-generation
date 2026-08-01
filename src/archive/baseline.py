import cv2
import numpy as np
import os

# ==============================================================================
# Baseline Panorama Generation Pipeline (Robust ROI & Smart Fallback)
# ==============================================================================

def main():
    print("Starting Robust ROI-Based ORB Baseline Pipeline...")
    
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
    def resize_to_height(img, target_height=600):
        h, w = img.shape[:2]
        ratio = target_height / h
        return cv2.resize(img, (int(w * ratio), target_height))
        
    img1 = resize_to_height(img1)
    img2 = resize_to_height(img2)
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Step 3: Extract ROI (Regions of Interest)
    roi_width1 = int(w1 * 0.4)
    roi_width2 = int(w2 * 0.4)
    
    roi1 = img1[:, w1 - roi_width1:] # Right side of img1
    roi2 = img2[:, :roi_width2]      # Left side of img2

    gray_roi1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
    gray_roi2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

    # Step 4: ORB Detection inside ROIs
    orb = cv2.ORB_create(nfeatures=3000)
    keypoints1, descriptors1 = orb.detectAndCompute(gray_roi1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray_roi2, None)
    print(f"Found {len(keypoints1)} keypoints in ROI 1 and {len(keypoints2)} in ROI 2.")

    # Step 5: Matching and Lowe's Ratio Test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    ratio_matches = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            ratio_matches.append(m)
            
    # Step 6: Spatial Distribution Filter (NMS)
    ratio_matches = sorted(ratio_matches, key=lambda x: x.distance)
    good_matches = []
    accepted_pts = [] 
    
    for m in ratio_matches:
        pt = keypoints2[m.trainIdx].pt
        too_close = False
        for acc_pt in accepted_pts:
            if np.sqrt((pt[0] - acc_pt[0])**2 + (pt[1] - acc_pt[1])**2) < 5.0:
                too_close = True
                break
        if not too_close:
            good_matches.append(m)
            accepted_pts.append(pt)
            
    print(f"Found {len(good_matches)} matches after Ratio Test and Spatial Filtering.")

    if len(good_matches) < 4:
        print("Not enough matches to compute transformation.")
        return

    # Step 7: Coordinate Shifting
    shifted_keypoints1 = []
    shifted_keypoints2 = list(keypoints2) 
    
    for kp in keypoints1:
        shifted_pt = (kp.pt[0] + (w1 - roi_width1), kp.pt[1])
        shifted_kp = cv2.KeyPoint(x=shifted_pt[0], y=shifted_pt[1], size=kp.size, 
                                  angle=kp.angle, response=kp.response, 
                                  octave=kp.octave, class_id=kp.class_id)
        shifted_keypoints1.append(shifted_kp)

    src_pts = []
    dst_pts = []
    for m in good_matches:
        src_pts.append(shifted_keypoints2[m.trainIdx].pt)
        dst_pts.append(shifted_keypoints1[m.queryIdx].pt)
        
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    # Step 8: Smart Fallback Transformation Logic
    # Attempt full Homography first
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    use_affine = False

    if H is not None:
        # Check for extreme distortion
        pts2_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2_warped = cv2.perspectiveTransform(pts2_corners, H)
        
        xmin_w = int(pts2_warped[:, 0, 0].min())
        xmax_w = int(pts2_warped[:, 0, 0].max())
        
        warped_width = xmax_w - xmin_w
        if warped_width > (3 * w2):
            print(f"Warning: Homography caused extreme stretching (warped width: {warped_width}px). Falling back to Affine.")
            use_affine = True
    else:
        use_affine = True

    if use_affine:
        affine_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if affine_matrix is not None:
            H = np.vstack([affine_matrix, [0, 0, 1]])
            print("Successfully calculated Affine Transformation.")
        else:
            print("Error: Could not calculate any transformation.")
            return

    # Visualize inliers
    matches_mask = mask.ravel().tolist()
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if matches_mask[i] == 1]
    
    match_img = cv2.drawMatches(img1, shifted_keypoints1, img2, shifted_keypoints2, inlier_matches, None, 
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    match_save_path = os.path.join(results_dir, 'feature_matches.jpg')
    cv2.imwrite(match_save_path, match_img)
    print(f"Saved ROI matching visualization to {match_save_path}")

    # Step 9: Standard Bounding Box Canvas Creation
    pts1_corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_warped = cv2.perspectiveTransform(pts2_corners, H)
    
    all_corners = np.concatenate((pts1_corners, pts2_warped), axis=0)
    
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # Calculate translation to ensure no negative coordinates
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    
    canvas_w = xmax - xmin
    canvas_h = ymax - ymin
    
    print(f"Warping images to canvas size {canvas_w}x{canvas_h}...")
    result = cv2.warpPerspective(img2, Ht.dot(H), (canvas_w, canvas_h))
    
    # Place img1 into the canvas
    roi = result[t[1]:h1+t[1], t[0]:w1+t[0]]
    # Fill empty (black) regions of warped img2 with img1 pixels
    mask_empty = (roi == 0).all(axis=2)
    roi[mask_empty] = img1[mask_empty]

    # Step 10: Save Final Panorama
    pano_save_path = os.path.join(results_dir, 'baseline_panorama.jpg')
    cv2.imwrite(pano_save_path, result)
    print(f"Saved final panorama to {pano_save_path}")
    print("ROI ORB execution complete!")

if __name__ == "__main__":
    main()
