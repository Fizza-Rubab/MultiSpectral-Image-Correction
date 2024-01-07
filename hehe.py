import cv2
import numpy as np

# Load your RGB and multispectral images
rgb_image = cv2.imread(r"C:\Users\User\Downloads\MS_rawImages\DJI_20231015133246_0001_D.JPG")
multispectral_image = cv2.imread(r"C:\Users\User\Downloads\MS_rawImages\DJI_20231015133246_0001_MS_G.TIF")

# Convert both images to grayscale (optional but recommended)
rgb_gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
multispectral_gray = cv2.cvtColor(multispectral_image, cv2.COLOR_BGR2GRAY)

# Detect keypoints and descriptors (using ORB, you can choose a different method)
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(rgb_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(multispectral_gray, None)

# Match keypoints
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract corresponding points
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Find the transformation matrix
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply the transformation to align the RGB image with the multispectral image
aligned_rgb = cv2.warpPerspective(rgb_image, M, (multispectral_image.shape[1], multispectral_image.shape[0]))

# Save the aligned RGB image
print(aligned_rgb.shape, multispectral_gray.shape)
cv2.imwrite('aligned_rgb_image.jpg', aligned_rgb)
