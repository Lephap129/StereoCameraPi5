from picamera2 import Picamera2
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


picam2a = Picamera2(0)
picam2b = Picamera2(1)

picam2a.preview_configuration.main.size = (320, 240)
picam2a.preview_configuration.main.format = "RGB888"
picam2a.preview_configuration.controls.FrameRate = 120.0
picam2a.preview_configuration.align()

picam2a.configure("preview")

picam2b.preview_configuration.main.size = (320, 240)
picam2b.preview_configuration.main.format = "RGB888"
picam2b.preview_configuration.controls.FrameRate = 120.0
picam2b.preview_configuration.align()

picam2b.configure("preview")

picam2a.start()
picam2b.start()



# Initialize StereoSGBM
# Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
block_size = 15
min_disp = 0
max_disp = 16*3
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

stereo = cv2.StereoSGBM_create(
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
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)

r_matcher = cv2.ximgproc.createRightMatcher(stereo)
lmbda = 8000
sigma = 1.5
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)



while True:
    frame1 = picam2a.capture_array()
    frame2 = picam2b.capture_array()
    grayL = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    # grayL = cv2.GaussianBlur(grayL, (3, 3), 0)
    # grayR = cv2.GaussianBlur(grayR, (3, 3), 0)
    # # Sharpen
    # bluredL = cv2.GaussianBlur(grayL, (0, 0), 5)
    # bluredR = cv2.GaussianBlur(grayR, (0, 0), 5)
    # grayL = cv2.addWeighted(grayL, 1.5, bluredL, -0.5, 0)
    # grayR = cv2.addWeighted(grayR, 1.5, bluredR, -0.5, 0)
    # # grayL = cv2.medianBlur(grayL, 15)  # Applying median blur to reduce noise
    # # grayR = cv2.medianBlur(grayR, 15)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # grayL = clahe.apply(grayL)
    # grayR = clahe.apply(grayR)
    
    # Compute the disparity map
    displ = stereo.compute(grayL, grayR)
    dispr = r_matcher.compute(grayR, grayL)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    disparity = wls_filter.filter(displ, grayL, None, dispr) / 16.0
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)
    current_time = time.time() 
    cv2.imshow("PiCam2a", grayL)
    #cv2.imshow("PiCam2b", frame2)
    cv2.imshow("PiCam2Disparity", disparity)
    #time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
picam2a.stop()
picam2b.stop()
cv2.destroyAllWindows()

