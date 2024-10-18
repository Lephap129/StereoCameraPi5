import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the stereo pair images
frame1 = cv2.imread('frame1_0.png')
frame2 = cv2.imread('frame2_0.png')

# Convert the images to grayscale
grayL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Initialize the stereo block matching object
stereo = cv2.StereoBM_create(numDisparities=16*2, blockSize=15)
stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
stereo.setPreFilterSize(9)
stereo.setPreFilterCap(30)
stereo.setTextureThreshold(10)
stereo.setUniquenessRatio(6)
stereo.setSpeckleWindowSize(50)
stereo.setSpeckleRange(29)
stereo.setDisp12MaxDiff(5)
stereo.setMinDisparity(5)

# Compute the disparity map
disparity = stereo.compute(grayL, grayR)

# Display the disparity map using Matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(disparity, cmap='plasma')
plt.colorbar()
plt.title('Disparity Map')
plt.show()
