﻿#      ▄▀▄     ▄▀▄
#     ▄█░░▀▀▀▀▀░░█▄
# ▄▄  █░░░░░░░░░░░█  ▄▄
#█▄▄█ █░░▀░░┬░░▀░░█ █▄▄█

#######################################
##### Authors:                    #####
##### Stephane Vujasinovic        #####
##### Frederic Uhrweiller         ##### 
#####                             #####
##### Creation: 2017              #####
##### Optimization: David Castillo#####
##### Rv: FEB:2018                #####
#######################################


#***********************
#**** Main Programm ****
#***********************

# Package importation
import time
import numpy as np
import cv2
import os
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from multiprocessing import Pool
from picamera2 import Picamera2
import time
import numpy as np
from matplotlib import pyplot as plt
# =========================sub Process===========================

def doWork(st): #j=1 is left, j=2 is right
    grayL = st[0] 
    grayR = st[1]
    j = st[2]
    
    # Used for the filtered image
    if j == 1 :
        disp= stereo.compute(grayL,grayR)
        # disp= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 77
    
    if j == 2 :
        stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time
        disp= stereoR.compute(grayR,grayL)
        # disp= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 77
    return disp

#====================================================

cv2.useOptimized()
wb=Workbook()
ws=wb.active  

# write into the excell worksheet

def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #print x,y,disp[y,x],filteredImg[y,x]
        """
				p p p
				p p p
				p p p
        """
        average=0
        for u in range (-1,2):     # (-1 0 1)
            for v in range (-1,2): # (-1 0 1)
                average += disp[y+u,x+v]
        average=average/9
        #Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        #Distance= np.around(Distance*0.01,decimals=2)
        #print('Distance: '+ str(Distance)+' m')
        print('Average: '+ str(average))
        counterdist = int(input("Enter distance (cm): "))
        ws.append([counterdist, average])

#*************************************************
#***** Parameters for Distortion Calibration *****
#*************************************************

# Termination criteria
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

##===========================================================
filenameL = os.path.join("models/", "{}.npy".format("imgpointsL"))
filenameR = os.path.join("models/", "{}.npy".format("imgpointsR"))
filename_op = os.path.join("models/", "{}.npy".format("objpoints"))
filename_mtR = os.path.join("models/", "{}.npy".format("mtxR"))
filename_dR = os.path.join("models/", "{}.npy".format("distR"))
filename_mtL = os.path.join("models/", "{}.npy".format("mtxL"))
filename_dL = os.path.join("models/", "{}.npy".format("distL"))
filename_chR = os.path.join("models/", "{}.npy".format("ChessImaR"))

# Read
imgpointsR = np.load(filenameR)
imgpointsL = np.load(filenameL)
objpoints = np.load(filename_op)
mtxR = np.load(filename_mtR)
distR = np.load(filename_dR)
mtxL = np.load(filename_mtL)
distL = np.load(filename_dL)
ChessImaR = np.load(filename_chR)

print('Cameras Ready to use')

#********************************************
#***** Calibrate the Cameras for Stereo *****
#********************************************

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC

retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)


#*******************************************
#***** Parameters for the StereoVision *****
#*******************************************

def turnParameter(para, modify):
    global block_size, min_disp, max_disp,uniquenessRatio, speckleWindowSize,speckleRange,disp12MaxDiff,preFilterCap,lmbda,sigma, stereo, wls_filter, r_matcher
    if para == 0:
        if (modify == 'up') and (block_size < 20):
            block_size += 1
        if (modify == 'down') and (block_size > 2):
            block_size -= 1
    if para == 1:
        if (modify == 'up') and (min_disp < 16*5):
            min_disp += 1
        if (modify == 'down') and (min_disp > 0):
            min_disp -= 1
    if para == 2:
        if (modify == 'up') and (max_disp < 16*15):
            max_disp += 16
        if (modify == 'down') and (max_disp > 16):
            max_disp -= 16
    if para == 3:
        if (modify == 'up') and (uniquenessRatio < 25):
            uniquenessRatio += 1
        if (modify == 'down') and (uniquenessRatio > 1):
            uniquenessRatio -= 1
    if para == 4:
        if (modify == 'up') and (speckleWindowSize < 200):
            speckleWindowSize += 10
        if (modify == 'down') and (speckleWindowSize > 50):
            speckleWindowSize -= 10
    if para == 5:
        if (modify == 'up') and (speckleRange < 50):
            speckleRange += 1
        if (modify == 'down') and (speckleRange > 1):
            speckleRange -= 1
    if para == 6:
        if (modify == 'up') and (disp12MaxDiff < 50):
            disp12MaxDiff += 1
        if (modify == 'down') and (disp12MaxDiff > 1):
            disp12MaxDiff -= 1
    if para == 7:
        if (modify == 'up') and (preFilterCap < 100):
            preFilterCap += 10
        if (modify == 'down') and (preFilterCap > 10):
            preFilterCap -= 10
    if para == 8:
        if (modify == 'up') and (lmbda < 100000):
            lmbda += 1000
        if (modify == 'down') and (lmbda > 5000):
            lmbda -= 1000
    if para == 9:
        if (modify == 'up') and (sigma < 2.7):
            sigma += 0.1
        if (modify == 'down') and (sigma > 1.3):
            sigma -= 0.1
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
    # stereo.setBlockSize(block_size)
    # stereo.setMinDisparity(min_disp)
    # stereo.setNumDisparities(num_disp)
    # stereo.setSpeckleRange(speckleRange)
    # stereo.setSpeckleWindowSize(speckleWindowSize)
    # stereo.setUniquenessRatio(uniquenessRatio)
    # stereo.setDisp12MaxDiff(disp12MaxDiff)
    # stereo.setPreFilterCap(preFilterCap)
    # wls_filter.setLambda(lmbda)
    # wls_filter.setSigmaColor(sigma)

listPara = [i for i in range(10)]
paraCtl = {0:"block_size", 
        1:"min_disp", 
        2:"max_disp",
        3:"uniquenessRatio",
        4:"speckleWindowSize",
        5:"speckleRange",
        6:"disp12MaxDiff",
        7:"preFilterCap",
        8:"lmbda",
        9:"sigma"}
choosePara = 0
# Initialize StereoSGBM
block_size = 1
min_disp = 0
max_disp = 16*2
uniquenessRatio = 15
speckleWindowSize = 0
speckleRange = 3
disp12MaxDiff = 1
P1=8 * 3 * block_size * block_size
P2=32 * 3 * block_size * block_size
preFilterCap = 63
mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
lmbda = 80000
sigma = 1.8
num_disp = max_disp - min_disp




disparity = None
fps = 0
prev_frame_time = 0
new_frame_time = 0
count_print = 0

#*************************************
#***** Starting the StereoVision *****
#*************************************

# Call the two cameras
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

fps = 0
prev_frame_time = 0
new_frame_time = 0
while True:
    #mark the start time
    startTime = time.time()
    # Start Reading Camera images
    frameL = picam2a.capture_array()
    frameR = picam2b.capture_array()

    # Rectify the images on rotation and alignement
    # Rectify the image using the calibration parameters founds during the initialisation
    # Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  
    # Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    Right_nice = frameR
    Left_nice = frameL
    
    # Convert from color(BGR) to gray
    grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

    #grayR= Right_nice
    #grayL= Left_nice
    #=======================================================================================
    # Filtering
    kernel= np.ones((3,3),np.uint8)
    # Compute the 2 images for the Depth_image
    # Run the pool in multiprocessing
    st1 = (grayL,grayR,1 )
    st2 = (grayL,grayR,2 )
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

    # Compute stereo image
    disp , dispR = map(doWork, (st1,st2))
    dispL= disp
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)
    # Using the WLS filter
    disparity = wls_filter.filter(dispL,grayL,None,dispR)
    # disparity= cv2.morphologyEx(disparity,cv2.MORPH_CLOSE, kernel)
    disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)
    # disparity= ((disparity.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect
    # Change the Color of the Picture into an Ocean Color_Map
    filt_Color= cv2.applyColorMap(disparity,cv2.COLORMAP_JET) 
    
    cv2.imshow('Filtered Color Depth',filt_Color)
    cv2.imshow('Depth',disparity)  
    cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
    
    new_frame_time = time.time()
    fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    #print results
    count_print+=1
    if count_print > 5:
        print("{}= {}".format(paraCtl[0], block_size))
        print("{}= {}".format(paraCtl[1], min_disp))
        print("{}= {}".format(paraCtl[2], max_disp))
        print("{}= {}".format(paraCtl[3], uniquenessRatio))
        print("{}= {}".format(paraCtl[4], speckleWindowSize))
        print("{}= {}".format(paraCtl[5], speckleRange))
        print("{}= {}".format(paraCtl[6], disp12MaxDiff))
        print("{}= {}".format(paraCtl[7], preFilterCap))
        print("{}= {}".format(paraCtl[8], lmbda))
        print("{}= {}".format(paraCtl[9], sigma))
        print("FPS: {}".format(int(fps)))
        print("Param control: {}".format(paraCtl[choosePara]))
    
    key = cv2.waitKey(10)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif key == ord('w'):
        turnParameter(choosePara,"up")
    elif key == ord('s'):
        turnParameter(choosePara,"down")
    elif key == ord('a'):
        choosePara = choosePara + 1 if choosePara < 9 else 0 
    elif key == ord('d'):
        choosePara = choosePara - 1 if choosePara > 0 else 9


# Save excel
wb.save("readvsdist.xlsx")

# Release the Cameras
picam2a.stop()
picam2b.stop()
cv2.destroyAllWindows()
