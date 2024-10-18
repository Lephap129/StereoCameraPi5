from picamera2 import Picamera2
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


frame1 = cv2.imread('frame1_0.png')
frame2 = cv2.imread('frame2_0.png')
grayL = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
# grayL = cv2.GaussianBlur(grayL, (9, 9), 0)
# grayR = cv2.GaussianBlur(grayR, (9, 9), 0)

stereo = cv2.StereoBM.create(numDisparities=16*2, blockSize=15)
stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
stereo.setPreFilterSize(9)
stereo.setPreFilterCap(30)
stereo.setTextureThreshold(10)
stereo.setUniquenessRatio(6)
stereo.setSpeckleWindowSize(50)
stereo.setSpeckleRange(29)
stereo.setDisp12MaxDiff(5)
stereo.setMinDisparity(5) 
disparity = stereo.compute(grayL,grayR)
disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity = np.uint8(disparity)

while True:

    current_time = time.time() 
    cv2.imshow("PiCam2a", grayL)
    cv2.imshow("PiCam2b", grayR)
    cv2.imshow("PiCam2Disparity", disparity)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()

