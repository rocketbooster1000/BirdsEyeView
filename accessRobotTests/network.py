import cv2
from cscore import CameraServer, VideoSource, CvSink, CvSource, VideoMode
from ntcore import NetworkTableInstance
import numpy as np
import time

ntinst = NetworkTableInstance.getDefault()

# Connect as a client to the robot
ntinst.startClient4("my-dashboard")
# ntinst.setServerTeam(XXXX)  # replace with your team number
# OR use IP directly:
# ntinst.setServer("10.XX.YY.2")
ntinst.setServer("127.0.0.1")


cs = CameraServer
cs.enableLogging()

# Input from physical camera
camera = cs.startAutomaticCapture()
camera.setResolution(640, 480)

time.sleep(1)

# Create sink (read frames) and output source (send processed frames)
sink = cs.getVideo()
output = cs.putVideo("Processed", 640, 480)

img = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
    timestamp, img = sink.grabFrame(img)
    if timestamp == 0:
        continue  # error, skip frame

    # --- do your processing here ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # --------------------------------

    output.putFrame(processed)