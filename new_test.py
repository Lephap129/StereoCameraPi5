from picamera2 import Picamera2
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Khởi tạo hai camera
picam2a = Picamera2(0)
picam2b = Picamera2(1)

# Cấu hình camera
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

# Bắt đầu camera
picam2a.start()
picam2b.start()

# Đọc các giá trị hiệu chỉnh hình ảnh từ file XML
cv_file = cv2.FileStorage("stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

# Kiểm tra các bản đồ hiệu chỉnh
if Left_Stereo_Map_x is None or Left_Stereo_Map_y is None or Right_Stereo_Map_x is None or Right_Stereo_Map_y is None:
    print("Error: One or more rectify maps are empty. Please check the file 'stereo_rectify_maps.xml'.")
    exit()

# Tạo các thanh điều chỉnh để tinh chỉnh các tham số của StereoBM
cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp', 600, 600)

cv2.createTrackbar('numDisparities', 'disp', 1, 17, lambda x: None)
cv2.createTrackbar('blockSize', 'disp', 5, 50, lambda x: None)
cv2.createTrackbar('preFilterType', 'disp', 1, 1, lambda x: None)
cv2.createTrackbar('preFilterSize', 'disp', 2, 25, lambda x: None)
cv2.createTrackbar('preFilterCap', 'disp', 5, 62, lambda x: None)
cv2.createTrackbar('textureThreshold', 'disp', 10, 100, lambda x: None)
cv2.createTrackbar('uniquenessRatio', 'disp', 15, 100, lambda x: None)
cv2.createTrackbar('speckleRange', 'disp', 0, 100, lambda x: None)
cv2.createTrackbar('speckleWindowSize', 'disp', 3, 25, lambda x: None)
cv2.createTrackbar('disp12MaxDiff', 'disp', 5, 25, lambda x: None)
cv2.createTrackbar('minDisparity', 'disp', 5, 25, lambda x: None)

# Tạo đối tượng StereoBM
stereo = cv2.StereoBM_create()

while True:
    # Chụp ảnh từ hai camera
    frame1 = picam2a.capture_array()
    frame2 = picam2b.capture_array()

    # Chuyển đổi ảnh sang grayscale
    imgR_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Áp dụng hiệu chỉnh hình ảnh
    Left_nice = cv2.remap(imgL_gray, Left_Stereo_Map_x, Left_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    Right_nice = cv2.remap(imgR_gray, Right_Stereo_Map_x, Right_Stereo_Map_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # Cập nhật các tham số của StereoBM từ các thanh điều chỉnh
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disp') * 16
    blockSize = cv2.getTrackbarPos('blockSize', 'disp') * 2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType', 'disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize', 'disp') * 2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap', 'disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold', 'disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disp')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disp') * 2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disp')
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disp')

    # Thiết lập các tham số cập nhật trước khi tính toán bản đồ disparity
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    # Tính toán bản đồ disparity
    disparity = stereo.compute(Left_nice, Right_nice)

    # Chuyển đổi sang float32 và chuẩn hóa
    disparity = disparity.astype(np.float32)
    disparity = (disparity / 16.0 - minDisparity) / numDisparities

    # Hiển thị bản đồ disparity
    cv2.imshow("disp", disparity)

    # Thoát bằng cách nhấn phím Esc
    if cv2.waitKey(1) == 27:
        break

# Dừng camera và đóng các cửa sổ
picam2a.stop()
picam2b.stop()
cv2.destroyAllWindows()
