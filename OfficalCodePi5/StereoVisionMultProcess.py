﻿from math import *
import threading
import concurrent.futures
import csv
import cv2
import os
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from multiprocessing import Queue
from picamera2 import Picamera2
import time
import numpy as np
from matplotlib import pyplot as plt
import psutil

######################### Record data ########################## 
fields = ['t','h','dx', 'dy', 'X', 'Y','FPS']
filename = "record_data.csv"
with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()


def updateLog(t,h,dx,dy,X,Y,FPS):
    list_append = [{'t':'{:.04f}'.format(t),'h':'{:.02f}'.format(h),'dx': '{:.02f}'.format(dx), 'dy': '{:.02f}'.format(dy), 'X': '{:.02f}'.format(X), 'Y': '{:.02f}'.format(Y), 'FPS': '{}'.format(FPS)}]
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writerows(list_append)
        csvfile.close()

##################################################################################

######################### Camera function #########################

def doWork(st): #j=1 is left, j=2 is right
    grayL = st[0] 
    grayR = st[1]
    j = st[2]
    
    # Used for the filtered image
    if j == 1 :
        disp= stereo.compute(grayL,grayR)
    # Create another stereo for right this time
    if j == 2 :
        stereoR=cv2.ximgproc.createRightMatcher(stereo) 
        disp= stereoR.compute(grayR,grayL)
    return disp

# Preprocessing function
def preprocess_frame(frameL,frameR):
    grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
    return grayL,grayR

# Frame capture thread
def capture_frames():
    global frame_bufferL
    global frame_bufferR
    global fps_cam, fps_cam_buffer
    t0cam = 0
    fps_limit = 50
    while True:
        # Guard CPU performance
        if fps_limit > 20 and psutil.cpu_percent() > 90:
            fps_limit -= 10
        elif fps_limit < 60 and psutil.cpu_percent() < 50:
            fps_limit += 10  
        if psutil.cpu_percent() > 90:
            time.sleep(0.01)
        # Take frame 
        frameL = picam2a.capture_array()
        frameR = picam2b.capture_array()
        with buffer_lock:
            frame_bufferL = frameL
            frame_bufferR = frameR
            t1cam = time.time()
            fps_cam_buffer = 1/(t1cam -t0cam)
            if (fps_cam > fps_limit):
                time.sleep(round(1/fps_limit - 1/fps_cam, 3))
            t0cam = t1cam


# Define the task Stereo to be run by each thread
def TakeStereo(id, t0stereo, frameL,frameR):
    fps_limit = 100
    grayL, grayR = preprocess_frame(frameL,frameR)
    # Filtering
    kernel= np.ones((3,3),np.uint8)
    # Compute the 2 images for the Depth_image
    st1 = (grayL,grayR,1 )
    st2 = (grayL,grayR,2 )
    disp , dispR = map(doWork, (st1,st2))
    dispL= disp
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)
    # Using the WLS filter
    cal_disparity = wls_filter.filter(dispL,grayL,None,dispR)
    cal_disparity= cv2.morphologyEx(cal_disparity,cv2.MORPH_CLOSE, kernel)
    # cal_disparity= ((cal_disparity.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect
    cal_disparity = cv2.normalize(src=cal_disparity, dst=cal_disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    cal_disparity = np.uint8(cal_disparity)
    disparity_buffer = cal_disparity
    # Put result
    queue.put((disparity_buffer,t0stereo))
        
    

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
    stereo.setBlockSize(block_size)
    stereo.setMinDisparity(min_disp)
    stereo.setNumDisparities(num_disp)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setPreFilterCap(preFilterCap)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
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
block_size = 5
min_disp = 0
max_disp = 16*2
uniquenessRatio = 15
speckleWindowSize = 150
speckleRange = 5
disp12MaxDiff = 5
P1=8 * 3 * block_size * block_size
P2=32 * 3 * block_size * block_size
preFilterCap = 63
mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
lmbda = 80000
sigma = 1.8
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


im_display = None
fps = 0
fps_cam = 0
t0stereo = 0
fps_stereo = 0
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

# Shared buffer for frames
frame_bufferL = None
frame_bufferR = None
fps_cam_buffer = None
buffer_lock = threading.Lock()

# Shared variable Process
disparity_buffer = None
fps_stereo_buffer = None
queue = Queue()

# Number of processes in the pool
max_processes = 3
futures = []
next_task_id = 0
limit_task_id = 1000
# Start frame capture thread
update_task = True
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Create a ProcessPoolExecutor
with concurrent.futures.ProcessPoolExecutor(max_workers=max_processes) as executor:
    while True:
        with buffer_lock:
            if frame_bufferL is not None:
                frameL= frame_bufferL
                frameR = frame_bufferR
                fps_cam = 0.99 * fps_cam + 0.01*fps_cam_buffer
                frame_bufferL = None
                frame_bufferR = None
                fps_cam_buffer = None
                if len(futures) <= max_processes*2:
                    if len(futures) <= max_processes:
                        new_future = executor.submit(TakeStereo, next_task_id, t0stereo, frameL, frameR)
                        futures.append(new_future)
                        if next_task_id < limit_task_id:
                            next_task_id += 1
                        else:
                            next_task_id = 0
                    else:
                        print("1.2")
                        if futures[0].done():
                            print("1.2.1")
                            take_stereo_result = queue.get()
                            disparity_buffer = take_stereo_result[0] 
                            t1stereo = time.time()
                            fps_stereo_buffer = 1/ (t1stereo - t0stereo)
                            t0stereo = t1stereo
                            fps_stereo = 0.9* fps_stereo + 0.1*fps_stereo_buffer
                            futures.remove(futures[0])
                            new_future = executor.submit(TakeStereo, next_task_id, t0stereo, frameL, frameR)
                            futures.append(new_future)
                            if next_task_id < limit_task_id:
                                next_task_id += 1
                            else:
                                next_task_id = 0
            
        print("2")
        # Show result
        if disparity_buffer is not None:
            print("2.1")
            disparity = disparity_buffer
            disparity_buffer = None
            filt_Color= cv2.applyColorMap(disparity,cv2.COLORMAP_JET)
            cv2.imshow('Filtered Color Depth',filt_Color)
            # cv2.imshow('Depth',disparity)  
            cv2.imshow('Both Images', frameL) 
            new_frame_time = time.time()
            fps = 0.9 * fps + 0.1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            # Print results
            count_print+=1
            if count_print > 20:
                # print("{}= {}".format(paraCtl[0], block_size))
                # print("{}= {}".format(paraCtl[1], min_disp))
                # print("{}= {}".format(paraCtl[2], max_disp))
                # print("{}= {}".format(paraCtl[3], uniquenessRatio))
                # print("{}= {}".format(paraCtl[4], speckleWindowSize))
                # print("{}= {}".format(paraCtl[5], speckleRange))
                # print("{}= {}".format(paraCtl[6], disp12MaxDiff))
                # print("{}= {}".format(paraCtl[7], preFilterCap))
                # print("{}= {}".format(paraCtl[8], lmbda))
                # print("{}= {}".format(paraCtl[9], sigma))
                print("__________________________________________________")
                # print("threading.activeCount():",threading.activeCount())
                # print("threading.currentThread():",threading.currentThread())
                # print("threading.enumerate():",threading.enumerate())
                print("FPS: {}".format(int(fps)))
                print("FPS_cam: {}".format(int(fps_cam)))
                print("FPS_stereo: {}".format(int(fps_stereo)))
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

#ser.close()
# Release memory
del frameL, frameR
executor.shutdown()
print("Close camera...")
# Release the Cameras
picam2a.stop()
picam2b.stop()
cv2.destroyAllWindows()