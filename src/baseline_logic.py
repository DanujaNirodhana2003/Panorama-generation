# ==============================================================================
# Baseline Panorama Generation Pipeline (ORB + Brute-Force Matcher)
# ==============================================================================

# Step 1: Import necessary libraries
# We will need OpenCV (cv2) for image processing and feature extraction.
# We will need NumPy (np) for matrix operations (like homography).
# We will need Matplotlib (plt) to visualize the matches for the M2 presentation.

# Step 2: Load the overlapping images
# Read "image1.jpg" (left image) and "image2.jpg" (right image) from the data/ folder.
# Convert both images from BGR (OpenCV default) to RGB (for Matplotlib display).
# Convert both images to Grayscale because feature detection works best on single-channel (black & white) images.

# Step 3: Initialize the ORB Feature Detector
# Create an ORB object. We can specify a maximum number of features to find (e.g., 2000) to speed it up.

# Step 4: Detect and Compute Features
# Run the ORB detector on the grayscale version of image1 to get 'keypoints1' and 'descriptors1'.
# Run the ORB detector on the grayscale version of image2 to get 'keypoints2' and 'descriptors2'.

# Step 5: Initialize the Feature Matcher
# Create a Brute-Force (BF) Matcher. 
# Since ORB descriptors are binary strings (0s and 1s), we must use the Hamming distance metric to compare them.

# Step 6: Match the Features
# Use the BF Matcher to find the best matches between 'descriptors1' and 'descriptors2'.
# Sort the matches based on their distance (lower distance means a better, more accurate match).

# Step 7: Visualize the Best Matches (For M2 Presentation)
# Draw the top 50 or 100 best matches connecting image1 and image2 using cv2.drawMatches().
# Save this visualization image to the results/ folder so you can put it on your PowerPoint slide.

# Step 8: Extract Matching Coordinates for Homography
# Check if we have enough good matches (usually we need at least 4 matches to calculate a perspective transformation).
# Extract the (x, y) coordinates of the matching keypoints from image1 and image2 and store them in NumPy arrays.

# Step 9: Calculate the Homography Matrix
# Use cv2.findHomography() with the extracted coordinates. 
# This calculates the 3x3 mathematical matrix that maps the perspective of image2 to image1.
# Note: For the baseline, we might just use all top matches. Later we will add RANSAC here to filter outliers.

# Step 10: Warp Image2 to Align with Image1
# Get the width and height of both images to calculate how big the final canvas needs to be.
# Use cv2.warpPerspective() to distort/bend image2 based on the Homography matrix so it aligns with image1.

# Step 11: Blend the Images Together (Stitching)
# Place image1 onto the left side of the warped image2 on the giant canvas.
# (For the baseline, simple overwriting is enough. We will do smooth blending later).

# Step 12: Save the Final Panorama
# Save the final stitched image to the results/ folder as "baseline_panorama.jpg".
# Display the final image on the screen to confirm it worked.
