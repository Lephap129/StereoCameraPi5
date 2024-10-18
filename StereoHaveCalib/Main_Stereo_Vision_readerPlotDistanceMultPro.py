from math import *
import threading
import concurrent.futures
import csv
import cv2
import os
from openpyxl import Workbook # Used for writing data into an Excel file
from sklearn.preprocessing import normalize
from multiprocessing import Pool
from picamera2 import Picamera2
import time
import numpy as np
from matplotlib import pyplot as plt

#Init Log
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

# Preprocessing function
def preprocess_frame(frameL,frameR):
    grayR= cv2.cvtColor(frameR,cv2.COLOR_BGR2GRAY)
    grayL= cv2.cvtColor(frameL,cv2.COLOR_BGR2GRAY)
    return grayL,grayR

# Frame capture thread
def capture_frames():
    global frame_bufferL
    global frame_bufferR
    global update_task
    global fps_cam, fps_cam_buffer
    t0cam = 0
    fps_limit = 40
    while True:
        if update_task:
            # Start Reading Camera images
            frameL = picam2a.capture_array()
            frameR = picam2b.capture_array()
            with buffer_lock:
                frame_bufferL = frameL
                frame_bufferR = frameR
                t1cam = time.time()
                fps_cam_buffer = t1cam -t0cam
                if (fps_cam > fps_limit):
                    time.sleep(round(1/fps_limit - 1/fps_cam, 3))
                t0cam = t1cam
            #update_task = False

# Define the task Stereo to be run by each thread
def TakeStereo(id, frameL,frameR):
    global idrun, limit_task_id
    global Depth_buffer, prev_frame_time, fps_stereo, fps_stereo_buffer
    t0stereo = 0
    fps_limit = 40
    # print(f"Thread {id} start!!!!!")
    # Convert from color(BGR) to gray
    grayL, grayR = preprocess_frame(frameL,frameR)

    # grayR= frameR
    # grayL= frameL
    #=======================================================================================
    # Filtering
    kernel= np.ones((3,3),np.uint8)
    # Compute the 2 images for the Depth_image
    # Run the pool in multiprocessing
    st1 = (grayL,grayR,1 )
    st2 = (grayL,grayR,2 )
    # Compute stereo image
    disp , dispR = map(doWork, (st1,st2))
    dispL= disp
    dispL= np.int16(dispL)
    dispR= np.int16(dispR)
    # Using the WLS filter
    cal_disparity = wls_filter.filter(dispL,grayL,None,dispR)
    cal_disparity= cv2.morphologyEx(cal_disparity,cv2.MORPH_CLOSE, kernel)
    cal_disparity = cv2.normalize(src=cal_disparity, dst=cal_disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    cal_disparity = np.uint8(cal_disparity)
    cal_disparity= ((cal_disparity.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect
    Depth = 7*10**8*np.power(np.float16(cal_disparity),-2.4)
    with worker:
        # Ensure the task runs in order by checking the count
        while idrun != id:
            worker.release()  # Release the lock to allow other threads
            time.sleep(0.1)  # Brief wait before retrying
            worker.acquire()  # Re-acquire the lock to check the condition
    
        # Now this thread can update the count
        # print(f"Thread {id} updating!!!!!")      
        if idrun < limit_task_id: idrun += 1
        else: idrun = 0
        Depth_buffer = Depth
        t1stereo = time.time()
        fps_stereo_buffer = t1stereo -t0stereo
        # if (fps_stereo > fps_limit):
        #     time.sleep(round(1/fps_limit - 1/fps_stereo, 3))
        t0stereo = t1stereo
        # print(f"Thread {id} finishing!!!!!")
    return id
        
    

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
block_size = 1
min_disp = 4
max_disp = 16*3
uniquenessRatio = 15
speckleWindowSize = 150
speckleRange = 3
disp12MaxDiff = 1
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

# Shared variable
count = 0
idrun = 0
Depth_buffer = None
fps_stereo_buffer = None
worker = threading.Lock()
# Number of threads in the pool
max_threads = 6
futures = []
next_task_id = 0
limit_task_id = 100

# Start frame capture thread
update_task = True
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()



# Create a ThreadPoolExecutor with max threads
with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
    while True:
        new_frame_time = time.time()
        fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        with buffer_lock:
            if frame_bufferL is not None:
                frameL= frame_bufferL
                frameR = frame_bufferR
                fps_cam = 0.99 * fps_cam + 0.01 / fps_cam_buffer
                frame_bufferL = None
                frame_bufferR = None
                fps_cam_buffer = None
                update_task = True
            else:
                continue
        if im_display is None:
            # Convert from color(BGR) to gray
            grayL, grayR = preprocess_frame(frameL,frameR)
            # grayR= frameR
            # grayL= frameL
            #=======================================================================================
            # Filtering
            kernel= np.ones((3,3),np.uint8)
            # Compute the 2 images for the Depth_image
            # Run the pool in multiprocessing
            st1 = (grayL,grayR,1 )
            st2 = (grayL,grayR,2 )
            
            # Compute stereo image
            disp , dispR = map(doWork, (st1,st2))
            dispL= disp
            dispL= np.int16(dispL)
            dispR= np.int16(dispR)
            # Using the WLS filter
            disparity = wls_filter.filter(dispL,grayL,None,dispR)
            disparity= cv2.morphologyEx(disparity,cv2.MORPH_CLOSE, kernel)
            disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            disparity = np.uint8(disparity)
            disparity= ((disparity.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect
            Depth = 7*10**8*np.power(np.float16(disparity),-2.4)
            # Matplotlib setup for real-time display
            plt.ion()
            fig, ax = plt.subplots()
            im_display = ax.imshow(Depth, cmap='jet_r')
            plt.colorbar(im_display)
            plt.title('Real-Time Depth Map (16-bit)')
            continue
        
        if len(futures) <= 20:
            if len(futures) <= max_threads:
                new_future = executor.submit(TakeStereo, next_task_id, frameL, frameR)
                futures.append(new_future)
                if next_task_id < limit_task_id:
                    next_task_id += 1
                else:
                    next_task_id = 0
            else:
                # As soon as a task completes, submit a new one if any are remaining
                for future in futures:
                    if future.done():
                        # Submit the next task and replace the completed future
                        futures.remove(future)
                        new_future = executor.submit(TakeStereo, next_task_id, frameL, frameR)
                        futures.append(new_future)
                        if next_task_id < limit_task_id:
                            next_task_id += 1
                        else:
                            next_task_id = 0
        
        with worker:
            if Depth_buffer is not None:
                Depth = Depth_buffer
                Depth_buffer = None
                fps_stereo = 0.99 * fps_stereo + 0.01 / fps_stereo_buffer
                fps_stereo_buffer = None
                # Update the Matplotlib plot
                im_display.set_data(disparity)
                plt.draw()
                plt.pause(0.1)
        #print results
        count_print+=1
        if count_print > 30:
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
            print("FPS: {}".format(int(fps)))
            print("FPS_cam: {}".format(int(fps_cam)))
            print("FPS_stereo: {}".format(int(fps_stereo)))
            print("Param control: {}".format(paraCtl[choosePara]))
        
        cv2.imshow('Both Images', np.hstack([frameL, frameR]))
        
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
worker.release()
print("Close camera...")
# Release the Cameras
picam2a.stop()
picam2b.stop()
cv2.destroyAllWindows()
plt.close('all')