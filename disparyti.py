import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from picamera2 import Picamera2
# Initialize the cameras
picam2a = Picamera2(0)
picam2b = Picamera2(1)

# Configure camera settings for picam2a
picam2a.preview_configuration.main.size = (640, 480)
picam2a.preview_configuration.main.format = "RGB888"
picam2a.preview_configuration.controls.FrameRate = 120.0
picam2a.preview_configuration.align()
picam2a.configure("preview")

# Configure camera settings for picam2b
picam2b.preview_configuration.main.size = (640, 480)
picam2b.preview_configuration.main.format = "RGB888"
picam2b.preview_configuration.controls.FrameRate = 120.0
picam2b.preview_configuration.align()
picam2b.configure("preview")

# Start the cameras
picam2a.start()
picam2b.start()

class DepthMap: 
    def __init__(self , showImages): 
        #Load image 
        frame1 = picam2a.capture_array()
        frame2 = picam2b.capture_array()
        
        # Convert frames to grayscale
        grayL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        if showImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(grayL)
            plt.subplot(122)
            plt.imshow(grayR)
            plt.show()
    def computeDepthMapSGBM(self):
        window_size = 7
        min_disp = 16
        nDispFactor = 14
        num_disp = 16*nDispFactor-min_disp
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            disp12MaxDiff=1,
            P1=8*3*window_size**2,
            P2=32*3*window_size**2,
            preFilterCap=63,  # Increased to better handle high contrast
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        
        #compute disparity map
        disparity = stereo.compute(grayL,grayR).astype(np.float32) / 16.0
        
        # Display
        plt.imshow(disparity, 'gray')
        plt.colorbar()
        plt.show()
        
def demoViewPics():
    dp = DepthMap(showImages=True)
def demoStrereoSGBM():
    dp = DepthMap(showImages=False)
    dp.computeDepthMapSGBM()
if __name__ == '__name__':
    demoStrereoSGBM()