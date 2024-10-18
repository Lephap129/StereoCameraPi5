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



stereo = cv2.StereoBM.create(numDisparities=16*5, blockSize=5)
stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_XSOBEL)
stereo.setPreFilterSize(9)
stereo.setPreFilterCap(30)
stereo.setTextureThreshold(10)
stereo.setUniquenessRatio(10)
stereo.setSpeckleWindowSize(100)
stereo.setSpeckleRange(50)
stereo.setDisp12MaxDiff(5)
stereo.setMinDisparity(5) 



while True:
    frame1 = picam2a.capture_array()
    frame2 = picam2b.capture_array()
    grayL = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    grayL = cv2.GaussianBlur(grayL, (3, 3), 0)
    grayR = cv2.GaussianBlur(grayR, (3, 3), 0)
    # Sharpen
    bluredL = cv2.GaussianBlur(grayL, (0, 0), 5)
    bluredR = cv2.GaussianBlur(grayR, (0, 0), 5)
    grayL = cv2.addWeighted(grayL, 1.5, bluredL, -0.5, 0)
    grayR = cv2.addWeighted(grayR, 1.5, bluredR, -0.5, 0)
    # grayL = cv2.medianBlur(grayL, 15)  # Applying median blur to reduce noise
    # grayR = cv2.medianBlur(grayR, 15)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayL = clahe.apply(grayL)
    grayR = clahe.apply(grayR)
    disparity = stereo.compute(grayL,grayR)
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

