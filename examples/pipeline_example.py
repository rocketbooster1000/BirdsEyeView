import cv2
from production.streamlined import Pipeline, compute_final_birdseye
from birdseye.sim import SimSource

ctx = Pipeline()                 # uses config/rig.json; Pipeline(rig, feather_px=40) to customise
print(ctx.names)                 # ['north', 'east', 'south', 'west'] - frame order to pass in

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    frames = [frame, frame, frame, frame]
    canvas = compute_final_birdseye(frames, ctx)   # 600x600 BGR image
    cv2.imshow("Birdseye", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break