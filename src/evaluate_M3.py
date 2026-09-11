import cv2
import numpy as np
import os
import time

def calculate_rmse(src_pts, dst_pts, H):
    warped_src_pts = cv2.perspectiveTransform(src_pts, H)
    errors = np.linalg.norm(warped_src_pts - dst_pts, axis=2)
    return np.sqrt(np.mean(errors ** 2))

def main():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    img1_path = os.path.join(data_dir, 'raw', 'image1.jpeg')
    img2_path = os.path.join(data_dir, 'raw', 'image2.jpeg')
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Images not found. Ensure image1.jpeg and image2.jpeg exist.")
        return

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    end_time = time.time()
    
    if H is not None:
        inliers_mask = mask.ravel() == 1
        inlier_src = src_pts[inliers_mask]
        inlier_dst = dst_pts[inliers_mask]
        rmse = calculate_rmse(inlier_src, inlier_dst, H)
        
        print(f"Total Matches: {len(good)}")
        print(f"Inliers: {np.sum(mask)} ({(np.sum(mask)/len(good))*100:.2f}%)")
        print(f"RMSE: {rmse:.4f} pixels")
        print(f"Time: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()
