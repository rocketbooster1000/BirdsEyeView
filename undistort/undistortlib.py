import json
from pathlib import Path

import cv2
import numpy as np

debug = False

INTRINSICS_PATH = Path(__file__).resolve().parent.parent / "calib" / "intrinsics_shared.json"

DIM=(1280, 720)
K=np.array([[700.411943374252, 0.0, 646.7195419804588], [0.0, 702.8297076955754, 312.9391328006887], [0.0, 0.0, 1.0]])
D=np.array([[-0.08069373318016052], [0.004905727970611169], [0.02408389532843681], [-0.02746516791216298]])
try:
    _intr = json.loads(INTRINSICS_PATH.read_text())
    DIM = tuple(int(v) for v in _intr["DIM"])
    K = np.array(_intr["K"], dtype=np.float64).reshape(3, 3)
    D = np.array(_intr["D"], dtype=np.float64).reshape(4, 1)
except (OSError, ValueError, KeyError) as e:
    print(f"undistortlib: using built-in K/D ({INTRINSICS_PATH} not loaded: {e})")
# DIM=(1280, 720)
# K=np.array([[700.5578619832233, 0.0, 645.9530107829912], [0.0, 702.9014182856116, 313.12721313960304], [0.0, 0.0, 1.0]])
# D=np.array([[-0.08835159712580129], [0.04440595546650344], [-0.03757404870313366], [0.016142337142302704]])
# DIM=(1280, 720)
# K=np.array([[696.3345605280558, 0.0, 643.5351333166279], [0.0, 696.9282310731818, 313.67743031482235], [0.0, 0.0, 1.0]])
# D=np.array([[-0.06735076174841784], [-0.06040914489538806], [0.10159258964252471], [-0.03848930078926364]])

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
# You should replace these 3 lines with the output in calibration step
# DIM=(1280, 720)
# K=np.array([[693.0913630949581, 0.0, 641.1321179565515], [0.0, 692.3287673230535, 315.2052538763502], [0.0, 0.0, 1.0]])
# D=np.array([[-0.08530315688010619], [0.042359415248929784], [-0.18689203246074707], [0.26621848501272444]])
def undistort(img):
    h,w = img.shape[:2]
    # map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # print(undistorted_img.shape)
    # print(img.shape)
    # print("\n")
    return undistorted_img
#     cv2.imshow("undistorted", undistorted_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def undistortImg(img_path):
    img = cv2.imread(img_path)
    # h,w = img.shape[:2]
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop(img, width, height):
    return img[0:height, 0:width]

def resize(img, scale):
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

# undistortImg("undistort/try2/WIN_20260309_16_46_37_Pro.jpg")

if (debug):
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera. Exiting...")
            break
        

        cv2.imshow("undistorted", crop(undistort(frame), 640, 480))
        cv2.imshow("original", frame)
        # cv2.imshow("cropped undistorted", crop)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()