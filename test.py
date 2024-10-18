import cv2 as cv
import numpy as np
from picamera2 import Picamera2

# Initialize two Picamera2 instances for left and right cameras
picamL = Picamera2(0)
picamR = Picamera2(1)

# Configure both cameras
picamL.preview_configuration.main.size = (320, 240)
picamL.preview_configuration.main.format = "RGB888"
picamL.preview_configuration.align()
picamL.configure("preview")

picamR.preview_configuration.main.size = (320, 240)
picamR.preview_configuration.main.format = "RGB888"
picamR.preview_configuration.align()
picamR.configure("preview")

# Start both cameras
picamL.start()
picamR.start()

# Set parameters for StereoBM
stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)

while True:
    # Capture frames from both cameras
    frameL = picamL.capture_array()
    frameR = picamR.capture_array()

    # Convert frames to grayscale
    grayL = cv.cvtColor(frameL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(frameR, cv.COLOR_BGR2GRAY)

    # Compute the disparity map
    disparity = stereo.compute(grayL, grayR)

    # Display frames from both cameras and the disparity map
    cv.imshow('Left Camera', frameL)
    cv.imshow('Right Camera', frameR)
    cv.imshow('Disparity', disparity)

    # Exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
picamL.stop()
picamR.stop()
cv.destroyAllWindows()
