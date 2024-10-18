import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Read both images and convert to grayscale
img1 = cv.imread('frame1_0.png', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('frame2_0.png', cv.IMREAD_GRAYSCALE)

# ------------------------------------------------------------
# CALCULATE DISPARITY (DEPTH MAP)
# Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
# and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

# StereoSGBM Parameter explanations:
# https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 5
min_disp = 0
max_disp = 8
# Maximum disparity minus minimum disparity. The value is always greater than zero.
# In the current implementation, this parameter must be divisible by 16.
num_disp = max_disp - min_disp
# Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
# Normally, a value within the 5-15 range is good enough
uniquenessRatio = 5
# Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
# Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
speckleWindowSize = 50
# Maximum disparity variation within each connected component.
# If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
# Normally, 1 or 2 is good enough.
speckleRange = 2
disp12MaxDiff = 1

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
    preFilterCap = 30,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
)

r_matcher = cv.ximgproc.createRightMatcher(stereo)
lmbda = 8000
sigma = 1.5
wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

displ = stereo.compute(img1,img2)
dispr = r_matcher.compute(img2,img1)
displ = np.int16(displ)
dispr = np.int16(dispr)
disparity_SGBM = wls_filter.filter(displ, img1, None, dispr) / 16.0
  
plt.imshow(disparity_SGBM, cmap='plasma')
plt.colorbar()
plt.show()
# Matplotlib setup for real-time display
"""plt.ion()
fig, ax = plt.subplots()
im_display = ax.imshow(disparity_SGBM, cmap='jet', vmin=0)
plt.colorbar(im_display)
plt.title('Real-Time Disparity Map (16-bit)')
cv.imwrite("disparity_SGBM_norm.png", disparity_SGBM)"""

"""disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)
plt.imshow(disparity_SGBM, cmap='plasma')
plt.colorbar()
plt.show()"""

# Normalize the values to a range from 0..255 for a grayscale image
"""disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
cv.imshow("Disparity", disparity_SGBM)
cv.imwrite("disparity_SGBM_norm.png", disparity_SGBM)

cv.waitKey()
cv.destroyAllWindows()"""
# ---------------------------------------------------------------
