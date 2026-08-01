import cv2
import numpy as np
import os

def resize_to_height(img, target_height=600):
    h, w = img.shape[:2]
    ratio = target_height / h
    return cv2.resize(img, (int(w * ratio), target_height))

def main():
    print("Generating Historical Images for LaTeX Report...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    history_dir = os.path.join(base_dir, 'results', 'history')
    
    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
        
    img1_path = os.path.join(data_dir, 'image1.jpg')
    img2_path = os.path.join(data_dir, 'image2.jpg')
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return
        
    img1 = resize_to_height(img1)
    img2 = resize_to_height(img2)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # ==========================================
    # STEP 1: Full Image ORB (Crazy Lines)
    # ==========================================
    print("Generating Step 1...")
    orb = cv2.ORB_create(nfeatures=3000)
    kp1_full, des1_full = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2_full, des2_full = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches_full = bf.knnMatch(des1_full, des2_full, k=2)
    
    good_full = []
    for m, n in matches_full:
        if m.distance < 0.75 * n.distance:
            good_full.append(m)
            
    # No RANSAC, just draw raw Lowe's matches to show the false positives across the image
    match_img_step1 = cv2.drawMatches(img1, kp1_full, img2, kp2_full, good_full[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(history_dir, 'step1_full_orb_matches.jpg'), match_img_step1)


    # ==========================================
    # STEP 2: ROI ORB without Spatial Filter (Clustered)
    # ==========================================
    print("Generating Step 2...")
    roi_w1 = int(w1 * 0.4)
    roi_w2 = int(w2 * 0.4)
    roi1 = img1[:, w1 - roi_w1:]
    roi2 = img2[:, :roi_w2]
    
    kp1_roi, des1_roi = orb.detectAndCompute(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), None)
    kp2_roi, des2_roi = orb.detectAndCompute(cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY), None)
    
    matches_roi = bf.knnMatch(des1_roi, des2_roi, k=2)
    good_roi = []
    for m, n in matches_roi:
        if m.distance < 0.75 * n.distance:
            good_roi.append(m)
            
    # Shift coordinates
    shifted_kp1 = []
    for kp in kp1_roi:
        shifted_kp1.append(cv2.KeyPoint(kp.pt[0] + (w1 - roi_w1), kp.pt[1], kp.size))
    shifted_kp2 = list(kp2_roi)
    
    # Draw clustered matches
    match_img_step2 = cv2.drawMatches(img1, shifted_kp1, img2, shifted_kp2, good_roi[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(history_dir, 'step2_roi_clustered_matches.jpg'), match_img_step2)


    # ==========================================
    # STEP 3: Spatial Filtering & Unconstrained Homography (Distortion/Adawela)
    # ==========================================
    print("Generating Step 3...")
    good_roi = sorted(good_roi, key=lambda x: x.distance)
    spaced_matches = []
    accepted_pts = []
    for m in good_roi:
        pt = shifted_kp2[m.trainIdx].pt
        too_close = False
        for acc_pt in accepted_pts:
            if np.sqrt((pt[0]-acc_pt[0])**2 + (pt[1]-acc_pt[1])**2) < 5.0:
                too_close = True
                break
        if not too_close:
            spaced_matches.append(m)
            accepted_pts.append(pt)
            
    src_pts = np.float32([shifted_kp2[m.trainIdx].pt for m in spaced_matches]).reshape(-1,1,2)
    dst_pts = np.float32([shifted_kp1[m.queryIdx].pt for m in spaced_matches]).reshape(-1,1,2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is not None:
        pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        
        # Intentionally NOT clamping to show severe stretching
        # Force a large but valid canvas so it doesn't crash OpenCV
        if xmax - xmin > 5000: xmax = xmin + 5000
        if ymax - ymin > 3000: ymax = ymin + 3000
            
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        
        distorted_result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
        roi = distorted_result[t[1]:h1+t[1], t[0]:w1+t[0]]
        mask_empty = (roi == 0).all(axis=2)
        roi[mask_empty] = img1[mask_empty]
        cv2.imwrite(os.path.join(history_dir, 'step3_unconstrained_homography.jpg'), distorted_result)

    # ==========================================
    # STEP 4: Constrained Homography (Final Perfect)
    # ==========================================
    print("Generating Step 4...")
    if H is not None:
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        
        xmin = min(xmin, 0)
        ymin = min(ymin, 0)
        xmax = max(xmax, w1)
        ymax = max(ymax, h1)
        
        if xmin < -w2: xmin = -w2
        if ymin < -h2: ymin = -h2
        if xmax > w1 + w2: xmax = w1 + w2
        if ymax > h1 + h2: ymax = h1 + h2
            
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        
        final_result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
        roi = final_result[t[1]:h1+t[1], t[0]:w1+t[0]]
        mask_empty = (roi == 0).all(axis=2)
        roi[mask_empty] = img1[mask_empty]
        cv2.imwrite(os.path.join(history_dir, 'step4_constrained_homography.jpg'), final_result)
        
    print("History generation complete!")

if __name__ == "__main__":
    main()
