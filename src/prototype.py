import cv2
import numpy as np
import os
import glob

# ==============================================================================
# M3 Prototype: Iterative Panorama Generation with Distance-Based Blending
# ==============================================================================

def resize_to_height(img, target_height=600):
    h, w = img.shape[:2]
    ratio = target_height / h
    return cv2.resize(img, (int(w * ratio), target_height))

def blend_images(img1, img2):
    """
    Blends two images of the SAME shape using Distance Transform for smooth feathering.
    img1: The base panorama (left)
    img2: The warped new image (right)
    """
    # Create binary masks for both images (where pixels are non-black)
    mask1 = (cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    mask2 = (cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    
    # Calculate Distance Transform
    # This gives the distance of each valid pixel to the nearest black (invalid) pixel
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 3)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3)
    
    # Create alpha map based on distances
    # Where both overlap, the one further from its edge gets a higher weight
    alpha = dist1 / (dist1 + dist2 + 1e-6)
    
    # Expand alpha to 3 channels for BGR blending
    alpha = cv2.merge([alpha, alpha, alpha])
    
    # Perform blending
    blended = (img1 * alpha + img2 * (1 - alpha))
    return blended.astype(np.uint8)

def get_roi_matches(img1, img2, orb, bf):
    """
    Extracts ROIs, finds ORB features, matches them, and shifts coordinates.
    img1: Panorama (Left)
    img2: New Image (Right)
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # We only look at the rightmost 40% of panorama, and leftmost 40% of the new image
    roi_width1 = min(w1, int(w2 * 0.4)) # In case panorama is very wide, only take w2*0.4
    roi_width2 = int(w2 * 0.4)
    
    roi1 = img1[:, w1 - roi_width1:]
    roi2 = img2[:, :roi_width2]
    
    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY), None)
    
    if des1 is None or des2 is None:
        return [], []
        
    raw_matches = bf.knnMatch(des1, des2, k=2)
    ratio_matches = []
    for m_n in raw_matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.75 * n.distance:
                ratio_matches.append(m)
                
    # Spatial Filtering (NMS)
    ratio_matches = sorted(ratio_matches, key=lambda x: x.distance)
    good_matches = []
    accepted_pts = [] 
    
    for m in ratio_matches:
        pt = kp2[m.trainIdx].pt
        too_close = False
        for acc_pt in accepted_pts:
            if np.sqrt((pt[0] - acc_pt[0])**2 + (pt[1] - acc_pt[1])**2) < 5.0:
                too_close = True
                break
        if not too_close:
            good_matches.append(m)
            accepted_pts.append(pt)
            
    # Shift coordinates back to full image space
    src_pts = []
    dst_pts = []
    for m in good_matches:
        src_pt = kp2[m.trainIdx].pt
        # Panorama keypoints need to be shifted by (w1 - roi_width1)
        dst_pt = (kp1[m.queryIdx].pt[0] + (w1 - roi_width1), kp1[m.queryIdx].pt[1])
        
        src_pts.append(src_pt)
        dst_pts.append(dst_pt)
        
    return np.float32(src_pts).reshape(-1, 1, 2), np.float32(dst_pts).reshape(-1, 1, 2)

def main():
    print("Starting M3 Iterative Panorama Prototype...")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    
    # Read all images in the data directory and sort them alphabetically
    image_paths = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
    if len(image_paths) < 2:
        print("Need at least 2 images to stitch.")
        return
        
    # We will test with a subset of up to 4 images to avoid massive memory issues during prototype
    # Feel free to change this later
    image_paths = image_paths[:4]
    
    print(f"Found {len(image_paths)} images to stitch.")
    
    # Initialize ORB and Matcher
    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Initialize the panorama with the first image
    panorama = cv2.imread(image_paths[0])
    panorama = resize_to_height(panorama)
    
    for i in range(1, len(image_paths)):
        print(f"Stitching image {i+1}/{len(image_paths)}...")
        new_img = cv2.imread(image_paths[i])
        new_img = resize_to_height(new_img)
        
        src_pts, dst_pts = get_roi_matches(panorama, new_img, orb, bf)
        
        if len(src_pts) < 4:
            print("Not enough matches to stitch. Stopping here.")
            break
            
        # Smart Fallback Transformation
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        use_affine = False
        
        h1, w1 = panorama.shape[:2]
        h2, w2 = new_img.shape[:2]
        
        if H is not None:
            pts2_corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            pts2_warped = cv2.perspectiveTransform(pts2_corners, H)
            warped_width = int(pts2_warped[:, 0, 0].max() - pts2_warped[:, 0, 0].min())
            warped_height = int(pts2_warped[:, 0, 1].max() - pts2_warped[:, 0, 1].min())
            if warped_width > (1.3 * w2) or warped_height > (1.3 * h2):
                print("Extreme stretching detected. Falling back to Affine.")
                use_affine = True
        else:
            use_affine = True
            
        if use_affine:
            affine, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if affine is not None:
                H = np.vstack([affine, [0, 0, 1]])
            else:
                print("Failed to find any valid transformation. Stopping here.")
                break
                
        # Create canvas
        pts1_corners = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        all_corners = np.concatenate((pts1_corners, pts2_warped if H is not None else pts1_corners), axis=0)
        
        [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        canvas_w, canvas_h = (xmax - xmin, ymax - ymin)
        print(f"Canvas size for iteration: {canvas_w}x{canvas_h}")
        
        # Warp new image
        warped_img = cv2.warpPerspective(new_img, Ht.dot(H), (canvas_w, canvas_h))
        
        # Place panorama on a black canvas of the same size
        warped_pano = np.zeros_like(warped_img)
        warped_pano[t[1]:h1+t[1], t[0]:w1+t[0]] = panorama
        
        # Blend them together seamlessly
        panorama = blend_images(warped_pano, warped_img)
        
        # Crop black borders to keep it clean for the next iteration
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        x, y, w_box, h_box = cv2.boundingRect(coords)
        panorama = panorama[y:y+h_box, x:x+w_box]
        
        # DEBUG SAVE
        cv2.imwrite(os.path.join(results_dir, f'debug_panorama_iter_{i}.jpg'), panorama)

    pano_save_path = os.path.join(results_dir, 'm3_prototype_panorama.jpg')
    cv2.imwrite(pano_save_path, panorama)
    print(f"Saved M3 iterative panorama to {pano_save_path}")

if __name__ == "__main__":
    main()
