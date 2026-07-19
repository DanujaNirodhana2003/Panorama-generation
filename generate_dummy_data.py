import cv2
import numpy as np
import os

# Create a large wide canvas
canvas = np.zeros((400, 800, 3), dtype=np.uint8)
canvas[:] = (200, 200, 200) # Light gray background

# Draw some distinctive shapes so ORB can find features
cv2.rectangle(canvas, (100, 100), (300, 300), (0, 0, 255), -1) # Red square
cv2.circle(canvas, (400, 200), 100, (0, 255, 0), -1)           # Green circle (in the middle, overlapping)
cv2.fillPoly(canvas, [np.array([[600, 100], [500, 300], [700, 300]])], (255, 0, 0)) # Blue triangle

# Add some text
cv2.putText(canvas, "Panorama Setup", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Split into two overlapping images (left and right)
# Left image: x from 0 to 500
image1 = canvas[:, 0:500]
# Right image: x from 300 to 800 (overlap from 300 to 500)
image2 = canvas[:, 300:800]

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'data')
os.makedirs(data_dir, exist_ok=True)

cv2.imwrite(os.path.join(data_dir, 'image1.jpg'), image1)
cv2.imwrite(os.path.join(data_dir, 'image2.jpg'), image2)

print("Generated dummy image1.jpg and image2.jpg in data/ folder.")
