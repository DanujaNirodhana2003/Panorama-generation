# ==============================================================================
# Advanced Panorama Generation Pipeline (SIFT + FLANN + Strict RANSAC)
# ==============================================================================

# Step 1: Load and Resize the Images
# Read "image1.jpg" and "image2.jpg" from the data folder.
# Resize both images to a maximum width of 1024 pixels to ensure processing speed 
# and filter out high-frequency noise from phone cameras.
# Convert both images to Grayscale for feature extraction.

# Step 2: Initialize the SIFT Feature Detector
# Create a SIFT (Scale-Invariant Feature Transform) object. 
# SIFT is much more accurate than ORB because it handles scale and rotation perfectly.

# Step 3: Detect and Compute SIFT Features
# Run SIFT on image1 to get keypoints and descriptors (which are floating-point arrays).
# Run SIFT on image2 to get keypoints and descriptors.

# Step 4: Initialize the FLANN Feature Matcher
# Since SIFT uses floating-point arrays (not binary like ORB), we cannot use Hamming distance.
# We must use FLANN (Fast Library for Approximate Nearest Neighbors) with KD-Tree algorithms.
# FLANN is the fastest and most accurate way to match SIFT features.

# Step 5: Match the Features using KNN
# Use FLANN to find the 2 best matches (k=2) for every descriptor in image1.

# Step 6: Apply a Strict Lowe's Ratio Test
# Loop through the raw matches. If the distance of the best match is less than 0.6 times 
# the distance of the second-best match, keep it. 
# A strict threshold of 0.6 removes almost ALL false positives (the "crazy lines").

# Step 7: Extract Coordinates for Homography
# If we have at least 4 good matches, extract their (x, y) coordinates from both images.

# Step 8: Calculate Homography with Strict RANSAC
# Use cv2.findHomography() to calculate the transformation matrix.
# We use RANSAC with a very tight threshold of 3.0 pixels (instead of 5.0) to strictly 
# reject any remaining outliers that somehow passed Lowe's ratio test.

# Step 9: Filter Matches and Visualize
# Use the mask returned by RANSAC to keep ONLY the 100% verified inlier matches.
# Draw these perfect matches using cv2.drawMatches() and save to "sift_matches.jpg".

# Step 10: Calculate the Panorama Canvas Size
# Calculate where the corners of image2 will land after being warped.
# Find the minimum and maximum X and Y coordinates to figure out how big the final image must be
# so that nothing gets cut off (especially on the left side).

# Step 11: Warp Image2 and Blend
# Warp image2 onto the new, large canvas.
# Use NumPy masking to place image1 onto the canvas ONLY where there are empty (black) pixels.
# This prevents image1 from completely overwriting the correctly warped image2.

# Step 12: Save the Final Panorama
# Save the stitched result as "sift_panorama.jpg" in the results folder.
