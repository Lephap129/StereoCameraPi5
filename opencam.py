from picamera2 import Picamera2
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt


picam2a = Picamera2(0)
picam2b = Picamera2(1)

picam2a.preview_configuration.main.size = (640, 480)
picam2a.preview_configuration.main.format = "RGB888"
picam2a.preview_configuration.controls.FrameRate = 120.0
picam2a.preview_configuration.align()

picam2a.configure("preview")

picam2b.preview_configuration.main.size = (640, 480)
picam2b.preview_configuration.main.format = "RGB888"
picam2b.preview_configuration.controls.FrameRate = 120.0
picam2b.preview_configuration.align()

picam2b.configure("preview")

picam2a.start()
picam2b.start()

frame1 = picam2a.capture_array()
frame2 = picam2b.capture_array()

grayL = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities = 48, blockSize = 21)
stereo.setUniquenessRatio(5)
stereo.setMinDisparity(1)
disparity = stereo.compute(grayL,grayR)
while True:
    cv2.imshow("Left Camera", frame1)
    cv2.imshow("Right Camera", frame2)
    plt.imshow(disparity,cmap='hsv')
    plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

