from picamera2 import Picamera2
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt

# Initialize the cameras
picam2a = Picamera2(0)
picam2b = Picamera2(1)

# Configure camera settings for picam2a
picam2a.preview_configuration.main.size = (320, 240)
picam2a.preview_configuration.main.format = "RGB888"
picam2a.preview_configuration.controls.FrameRate = 120.0
picam2a.preview_configuration.align()
picam2a.configure("preview")

# Configure camera settings for picam2b
picam2b.preview_configuration.main.size = (320, 240)
picam2b.preview_configuration.main.format = "RGB888"
picam2b.preview_configuration.controls.FrameRate = 120.0
picam2b.preview_configuration.align()
picam2b.configure("preview")

# Start the cameras
picam2a.start()
picam2b.start()

# Initialize StereoSGBM
block_size = 15
min_disp = 16*4
max_disp = 16*5
uniquenessRatio = 15
speckleWindowSize = 150
speckleRange = 2
disp12MaxDiff = 2
P1=8 * 3 * block_size * block_size
P2=32 * 3 * block_size * block_size
preFilterCap = 30
mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
lmbda = 8000
sigma = 2
num_disp = max_disp - min_disp


stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=P1,
    P2=P2,
    preFilterCap=preFilterCap,
    mode=mode,
)

r_matcher = cv2.ximgproc.createRightMatcher(stereo)
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

disparity = None

# Real-time processing loop
while True:
    # Capture frames from both cameras
    frame1 = picam2a.capture_array()
    frame2 = picam2b.capture_array()
    
    # Convert frames to grayscale
    grayL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # # Apply Gaussian Blur
    # grayL = cv2.GaussianBlur(grayL, (3, 3), 0)
    # grayR = cv2.GaussianBlur(grayR, (3, 3), 0)
    
    # # Sharpen the images
    # bluredL = cv2.GaussianBlur(grayL, (0, 0), 5)
    # bluredR = cv2.GaussianBlur(grayR, (0, 0), 5)
    # grayL = cv2.addWeighted(grayL, 1.5, bluredL, -0.5, 0)
    # grayR = cv2.addWeighted(grayR, 1.5, bluredR, -0.5, 0)
    
    # # Apply CLAHE for contrast enhancement
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # grayL = clahe.apply(grayL)
    # grayR = clahe.apply(grayR)
    
    if disparity is None:
        # Compute the disparity map
        displ = stereo.compute(grayL, grayR)
        dispr = r_matcher.compute(grayR, grayL)
        # displ = np.int16(displ)
        dispr = np.int16(dispr)
        disparity = wls_filter.filter(displ, grayL, None, dispr) / 16.0
        # Matplotlib setup for real-time display
        plt.ion()
        fig, ax = plt.subplots()
        im_display = ax.imshow(displ, cmap='jet', vmin=0)
        plt.colorbar(im_display)
        plt.title('Real-Time Disparity Map (16-bit)')
    else:
        # Compute the disparity map
        displ = stereo.compute(grayL, grayR)
        dispr = r_matcher.compute(grayR, grayL)
        # displ = np.int16(displ)
        dispr = np.int16(dispr)
        disparity = 0.1*disparity + 0.9 * wls_filter.filter(displ, grayL, None, dispr) / 16.0
        # Update the Matplotlib plot
        disparity = wls_filter.filter(displ, grayL, None, dispr) / 16.0
        # Update the Matplotlib plot
        im_display.set_data(disparity)
    
        plt.draw()
        plt.pause(0.1)

    # Show the original grayscale images with OpenCV (optional)
    cv2.imshow("PiCam2a", grayL)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the cameras and close windows
picam2a.stop()
picam2b.stop()
cv2.destroyAllWindows()
plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the last frame displayed
