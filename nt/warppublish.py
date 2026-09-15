import cv2
import numpy as np
from cscore import CameraServer, VideoSource, CvSink, CvSource, VideoMode
from ntcore import NetworkTableInstance
import time

def compute_homography(src_points, dst_points):
    """
    src_points: pixel coords in camera image [(x1,y1), (x2,y2), ...]
    dst_points: where those points land in bird's-eye canvas
    """
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    return H

# Example: front camera
# Pick 4 points on the ground plane visible in camera (e.g. lane markers, floor tape)
front_src = [(200, 400), (440, 400), (100, 600), (540, 600)]  # camera pixels
front_dst = [(200, 300), (400, 300), (200, 500), (400, 500)]  # bird's-eye canvas pixels
warp_rotate_dst_90 = [front_dst[2], front_dst[0], front_dst[3], front_dst[1]]
warp_rotate_dst_180 = [warp_rotate_dst_90[2], warp_rotate_dst_90[0], warp_rotate_dst_90[3], warp_rotate_dst_90[1]]
warp_rotate_dst_270 = [warp_rotate_dst_180[2], warp_rotate_dst_180[0], warp_rotate_dst_180[3], warp_rotate_dst_180[1]]

vertical_offset = 150
rotate_src = [(200, 200), (300, 200), (200, 300), (300, 300)]
rotate_dst = [(200, 300 + vertical_offset), (200, 200 + vertical_offset), (300, 300 + vertical_offset), (300, 200 + vertical_offset)]

ROTATE_HOMOGRAPH = compute_homography(rotate_src, rotate_dst)
H_warp_rotate_0 = compute_homography(front_src, front_dst)
H_warp_rotate_90 = compute_homography(front_src, warp_rotate_dst_90)
H_warp_rotate_180 = compute_homography(front_src, warp_rotate_dst_180)
H_warp_rotate_270 = compute_homography(front_src, warp_rotate_dst_270)
# H_warp_rotate_90 = compute_homography(front_src, warp_rotate_dst_90)

H_front = compute_homography(front_src, front_dst)
H_back = compute_homography(front_src, front_dst)  # Replace with actual points for back camera
H_left = compute_homography(front_src, rotate_dst)  # Replace with actual points for left camera
H_right = compute_homography(front_src, front_dst)  # Replace with actual points for right camera



CANVAS_W, CANVAS_H = 600, 600  # bird's-eye output size

masks = []

def warp_to_birdseye(frame, H):
    return cv2.warpPerspective(frame, H, (CANVAS_W, CANVAS_H))

def rotate_frame(frame, H):
    return cv2.warpPerspective(frame, H, (CANVAS_W, CANVAS_H))

def create_weight_mask(shape, index, numMasks):
    """Gradient mask for smooth blending at edges."""
    if (numMasks < 4):
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        # Simple center-weighted gradient
        for y in range(h):
            for x in range(w):
                cx = 1 - abs(x - w/2) / (w/2)
                cy = 1 - abs(y - h/2) / (h/2)
                mask[y, x] = cx * cy
        masks.append(mask)
    return masks[index]

def stitch_views(warped_views, numMasksMade):
    """warped_views: list of 4 warped frames (numpy arrays)"""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    weight_sum = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

    for i, view in enumerate(warped_views):
        mask = create_weight_mask(view.shape, i, numMasksMade)
        for c in range(3):
            canvas[:, :, c] += view[:, :, c].astype(np.float32) * mask
        weight_sum += mask

    # Normalize
    weight_sum = np.maximum(weight_sum, 1e-6)
    for c in range(3):
        canvas[:, :, c] /= weight_sum

    return canvas.astype(np.uint8)

def main():
    ntinst = NetworkTableInstance.getDefault();
    # Connect as a client to the robot
    ntinst.startClient4("my-dashboard")
    # ntinst.setServerTeam(XXXX)  # replace with your team number
    # OR use IP directly:
    # ntinst.setServer("10.XX.YY.2")
    ntinst.setServer("127.0.0.1")
    
    print("Connecting to NetworkTables...")
    
    time.sleep(1)
    
    cs = CameraServer
    cs.enableLogging()
    
    output = cs.putVideo("Processed", 640, 480)
    
    # Open 4 cameras (or video files for testing)
    caps = [
        # cv2.VideoCapture(0, cv2.CAP_DSHOW),  # front
        # cv2.VideoCapture(1, cv2.CAP_DSHOW),  # front
        # cv2.VideoCapture(2, cv2.CAP_DSHOW), # back
        cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # cv2.VideoCapture(1, cv2.CAP_DSHOW)

    ]
    
    numMasksMade = 0

    # Load or compute homographies
    # homographies = [H_front, H_back, H_left]
    homographies = [H_warp_rotate_0, H_warp_rotate_90, H_warp_rotate_180]

    rotates = [ROTATE_HOMOGRAPH]

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera. Exiting...")
                break
            frames.append(frame)
            
        # print(len(frames))

        if len(frames) < 1:
            break

        # Warp all 4 to bird's-eye
        # warped = [warp_to_birdseye(f, H) for f, H in zip(frames, rotates)]
        warped = [warp_to_birdseye(f, H) for f, H in zip(frames, homographies)]
        
        warped_rotate = []
        
        for warp_frame in warped:
            warped_rotate.append(rotate_frame(warp_frame, ROTATE_HOMOGRAPH))
            
        warrotated = []
        
        # for f in frames:
        #     warrotated.append(warp_to_birdseye(f, WARP_ROATTE_HOMOGRAPH))
        
        # print(len(warped))

        # Stitch
        # result = stitch_views(warped, numMasksMade)
        # if (numMasksMade < 4):
        #     numMasksMade += 1
        #     print(f"numMasksMade: {numMasksMade}")
        # result = warped[0]  # For testing, just show one warped view

        # cv2.imshow("Bird's Eye View", result)
        for i in range(len(frames)):
            cv2.imshow("Frame " + str(i), frames[i])
            
        for i in range (len(warped)):
            cv2.imshow("Warped  " + str(i), warped[i])
            
        output.putFrame(warped[0])
            
        # for i in range(len(warped_rotate)):
        #     cv2.imshow("Warp then rotated " + str(i), warped_rotate[i])
            
        # for i in range(len(warrotated)):
        #     cv2.imshow("Warrotated " + str(i), warrotated[i])
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    
    
print("starting")
main()