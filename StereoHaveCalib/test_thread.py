import asyncio
import time
import numpy as np
import cv2
from picamera2 import Picamera2
from matplotlib import pyplot as plt

def doWork(st):
    grayL = st[0]
    grayR = st[1]
    j = st[2]
    if j == 1:
        disp = stereo.compute(grayL, grayR)
    if j == 2:
        stereoR = cv2.ximgproc.createRightMatcher(stereo)
        disp = stereoR.compute(grayR, grayL)
    return disp

async def capture_frame(picam, queue, lock, is_left=True):
    while True:
        frame = picam.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        async with lock:
            queue[0 if is_left else 1] = gray
        await asyncio.sleep(0)  # Yield control to the event loop

async def main():
    global stereo, wls_filter, disparity, fps, prev_frame_time
    
    # Initialize StereoSGBM parameters
    block_size = 1
    min_disp = 0
    max_disp = 16 * 2
    uniquenessRatio = 15
    speckleWindowSize = 150
    speckleRange = 3
    disp12MaxDiff = 1
    P1 = 8 * 3 * block_size * block_size
    P2 = 32 * 3 * block_size * block_size
    preFilterCap = 63
    mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    lmbda = 80000
    sigma = 1.8
    num_disp = max_disp - min_disp

    disparity = None
    fps = 0
    prev_frame_time = 0

    picam2a = Picamera2(0)
    picam2b = Picamera2(1)

    picam2a.preview_configuration.main.size = (320, 240)
    picam2a.preview_configuration.main.format = "RGB888"
    picam2a.preview_configuration.align()
    picam2a.configure("preview")

    picam2b.preview_configuration.main.size = (320, 240)
    picam2b.preview_configuration.main.format = "RGB888"
    picam2b.preview_configuration.align()
    picam2b.configure("preview")

    picam2a.start()
    picam2b.start()

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

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    frame_queue = [None, None]
    lock = asyncio.Lock()

    # Start two tasks for capturing frames
    capture_left_task = asyncio.create_task(capture_frame(picam2a, frame_queue, lock, True))
    capture_right_task = asyncio.create_task(capture_frame(picam2b, frame_queue, lock, False))

    plt.ion()
    fig, ax = plt.subplots()
    im_display = None

    while True:
        async with lock:
            grayL, grayR = frame_queue

        if grayL is None or grayR is None:
            await asyncio.sleep(0)  # Yield control if no frame is ready
            continue

        st1 = (grayL, grayR, 1)
        st2 = (grayL, grayR, 2)

        if disparity is None:
            disp, dispR = map(doWork, (st1, st2))
            dispL, dispR = np.int16(disp), np.int16(dispR)
            disparity = wls_filter.filter(dispL, grayL, None, dispR)
            disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            disparity = ((disparity.astype(np.float32)) - min_disp) / num_disp
            im_display = ax.imshow(disparity, cmap='gray')
            plt.colorbar(im_display)
            plt.title('Real-Time Disparity Map')
        else:
            disp, dispR = map(doWork, (st1, st2))
            dispL, dispR = np.int16(disp), np.int16(dispR)
            disparity = wls_filter.filter(dispL, grayL, None, dispR)
            disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            disparity = ((disparity.astype(np.float32)) - min_disp) / num_disp
            im_display.set_data(disparity)
            plt.draw()
            plt.pause(0.1)

        new_frame_time = time.time()
        fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
        print("FPS: {}".format(int(fps)))
        prev_frame_time = new_frame_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2a.stop()
    picam2b.stop()
    cv2.destroyAllWindows()

    # Cancel tasks if needed
    capture_left_task.cancel()
    capture_right_task.cancel()
    await asyncio.sleep(0)  # Ensure cancellation

# Run the asyncio event loop
asyncio.run(main())
