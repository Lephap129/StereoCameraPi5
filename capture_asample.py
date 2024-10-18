from picamera2 import Picamera2
import cv2
import time

# Initialize the cameras
picam2a = Picamera2(0)
picam2b = Picamera2(1)

# Configure camera 1
picam2a.preview_configuration.main.size = (640, 480)
picam2a.preview_configuration.main.format = "RGB888"
picam2a.preview_configuration.controls.FrameRate = 120.0
picam2a.preview_configuration.align()
picam2a.configure("preview")

# Configure camera 2
picam2b.preview_configuration.main.size = (640, 480)
picam2b.preview_configuration.main.format = "RGB888"
picam2b.preview_configuration.controls.FrameRate = 120.0
picam2b.preview_configuration.align()
picam2b.configure("preview")

# Start both cameras
picam2a.start()
picam2b.start()

frame_count = 0  # To count the number of captured frames

frame1 = picam2a.capture_array()
frame2 = picam2b.capture_array()

# Save the frames as PNG images
cv2.imwrite(f"frame1_{frame_count}.png", frame1)
cv2.imwrite(f"frame2_{frame_count}.png", frame2)

# Clean up
picam2a.stop()
picam2b.stop()
