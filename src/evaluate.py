import cv2
import numpy as np
import os
import time

def calculate_rmse(src_pts, dst_pts, H):
    """
    Calculates the Root Mean Square Error (Reprojection Error) of inliers.
    """
    # Warp src_pts using H
    warped_src_pts = cv2.perspectiveTransform(src_pts, H)
    
    # Calculate Euclidean distance between warped points and actual dst_pts
    errors = np.linalg.norm(warped_src_pts - dst_pts, axis=2)
    
    # RMSE
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse

def main():
    print("Starting M3 Evaluation Script...")
    start_time = time.time()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    img1_path = os.path.join(data_dir, 'image1.jpg')
    img2_path = os.path.join(data_dir, 'image2.jpg')
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    # Resize
    h, w = img1.shape[:2]
    ratio = 600 / h
    img1 = cv2.resize(img1, (int(w * ratio), 600))
    h, w = img2.shape[:2]
    ratio = 600 / h
    img2 = cv2.resize(img2, (int(w * ratio), 600))
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    roi1 = img1[:, w1 - int(w1*0.4):]
    roi2 = img2[:, :int(w2*0.4)]
    
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY), None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            
    # Shift coordinates
    src_pts = []
    dst_pts = []
    for m in good:
        src_pt = kp2[m.trainIdx].pt
        dst_pt = (kp1[m.queryIdx].pt[0] + (w1 - int(w1*0.4)), kp1[m.queryIdx].pt[1])
        src_pts.append(src_pt)
        dst_pts.append(dst_pt)
        
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    end_time = time.time()
    
    if H is not None:
        inliers_mask = mask.ravel() == 1
        inlier_src = src_pts[inliers_mask]
        inlier_dst = dst_pts[inliers_mask]
        
        inlier_ratio = (np.sum(mask) / len(good)) * 100
        rmse = calculate_rmse(inlier_src, inlier_dst, H)
        
        print(f"\n--- PRELIMINARY EVALUATION METRICS ---")
        print(f"Total Features (Img1): {len(kp1)}")
        print(f"Total Features (Img2): {len(kp2)}")
        print(f"Total Matches (Lowe's): {len(good)}")
        print(f"RANSAC Inliers: {np.sum(mask)} ({inlier_ratio:.2f}%)")
        print(f"Reprojection Error (RMSE): {rmse:.4f} pixels")
        print(f"Execution Time: {(end_time - start_time):.4f} seconds")
        print("--------------------------------------\n")
    else:
        print("Failed to compute homography.")

if __name__ == "__main__":
    main()
