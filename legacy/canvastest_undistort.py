import cv2
import numpy as np
from undistortlib import resize, undistort
from undistortlib import crop

def create_canvas(width, height):
    return np.zeros((height, width, 3), dtype=np.uint8)

def place_frame(canvas, frame, x, y):
    """Place a frame on the canvas at position (x, y)"""
    h, w = frame.shape[:2]
    canvas[y:y+h, x:x+w] = frame
    
def place_frame_masked(canvas, frame, x, y, black_threshold=10):
    ch, cw = canvas.shape[:2]
    fh, fw = frame.shape[:2]

    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + fw, cw), min(y + fh, ch)
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fy1 + (x2 - x1), fy1 + (y2 - y1)  

    if x2 <= x1 or y2 <= y1:
        return

    src = frame[fy1:fy2, fx1:fx2]
    roi = canvas[y1:y2, x1:x2]

    # Mask: True where frame pixel is NOT black
    mask = np.any(src > black_threshold, axis=2)

    roi[mask] = src[mask]

def compute_homography(src_points, dst_points):
    """
    src_points: pixel coords in camera image [(x1,y1), (x2,y2), ...]
    dst_points: where those points land in bird's-eye canvas
    """
    src = np.float32(src_points)
    dst = np.float32(dst_points)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    return H

A = (200, 300)
B = (400, 300)
C = (200, 500)
D = (400, 500)

xOffset = 300
yOffset = 300
# Example: front camera
# Pick 4 points on the ground plane visible in camera (e.g. lane markers, floor tape)
front_src = [(200, 400), (440, 400), (100, 600), (540, 600)]  # camera pixels
front_dst = [(200 + xOffset, 300 + yOffset), (400 + xOffset, 300 + yOffset), (200 + xOffset, 500 + yOffset), (400 + xOffset, 500 + yOffset)]  # bird's-eye canvas pixels
warp_rotate_dst_90 = [front_dst[2], front_dst[0], front_dst[3], front_dst[1]]
warp_rotate_dst_180 = [(warp_rotate_dst_90[2][0], warp_rotate_dst_90[2][1] - 300), 
                       (warp_rotate_dst_90[0][0], warp_rotate_dst_90[0][1] - 300), 
                       (warp_rotate_dst_90[3][0], warp_rotate_dst_90[3][1] - 300), 
                       (warp_rotate_dst_90[1][0], warp_rotate_dst_90[1][1] - 300)]
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

def warp_to_birdseye(frame, H, dimensions):
    return cv2.warpPerspective(frame, H, dimensions)

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
    # Open 4 cameras (or video files for testing)
    caps = [
        # cv2.VideoCapture(0, cv2.CAP_DSHOW),  # front
        # cv2.VideoCapture(1, cv2.CAP_DSHOW),  # front
        # cv2.VideoCapture(2, cv2.CAP_DSHOW), # back
        # cv2.VideoCapture(0, cv2.CAP_DSHOW),
        cv2.VideoCapture(1, cv2.CAP_DSHOW)

    ]
    
    caps[0].set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    caps[0].set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    dimensions = [
        # (CANVAS_W, CANVAS_H),
        # (CANVAS_W, 1080),
        # (CANVAS_W, CANVAS_H)
        (1280, 720)
    ]
    
    fisheyes = [
        True,
        True,
        True
    ]
    
    numMasksMade = 0

    # Load or compute homographies
    # homographies = [H_front, H_back, H_left]
    homographies = [H_warp_rotate_0, H_warp_rotate_90, H_warp_rotate_180]
    

    rotates = [ROTATE_HOMOGRAPH]

    while True:
        canvas = create_canvas(1920, 1080)
        frames = []
        for i in range(len(caps)):
            cap = caps[i]
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera. Exiting...")
                break
            if (fisheyes[i]):
                # frame = crop(undistort(frame), 640, 480)
                frame = resize(undistort(frame), 0.5)
            frames.append(frame)
            
        # print(len(frames))

        if len(frames) < 1:
            break

        # Warp all 4 to bird's-eye
        # warped = [warp_to_birdseye(f, H) for f, H in zip(frames, rotates)]
        warped = [warp_to_birdseye(f, H, d) for f, H, d in zip(frames, homographies, dimensions)]
        
        warped_rotate = []
            
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
            cv2.imshow("Warped  " + str(i), warped[len(warped) - 1 - i])
            # place_frame(canvas, warped[i], 0, 0)  # Example placement on canvas
            
        # place_frame_masked(canvas, warped[1], 0, 0)
        # place_frame_masked(canvas, warped[2], 300, 200)
        place_frame_masked(canvas, warped[0], 300, 0)
        # place_frame(canvas, warped[1], 0, 0)
        # place_frame(canvas, warped[2], 0, 0)
        # place_frame(canvas, warped[0], 300, 0)
            
        cv2.imshow("stitched", canvas)
            
        
            
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