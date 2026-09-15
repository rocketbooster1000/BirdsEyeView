import cv2 as cv
from undistortlib import crop, undistort

stretch_factor_width = 3
stretch_factor_height = 0.5

# Create a VideoCapture object, 0 for the default camera (built-in webcam)
# cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap = cv.VideoCapture(1, cv.CAP_DSHOW)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
# Check if the camera opened successfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # height, width = frame.shape[:2]

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Optional: Perform operations on the frame here (e.g., convert to grayscale)
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # resize = cv.resize(frame, (int(width * stretch_factor_width), int(height * stretch_factor_height)), interpolation=cv.INTER_AREA)

    # Display the resulting frame
    cv.imshow('Live Camera Feed', frame)
    cv.imshow("undistored", undistort(frame))
    # cv.imshow('Grayscale Feed', gray)
    # cv.imshow('Resized Feed', resize)

    # Press 'q' on the keyboard to exit the stream
    if cv.waitKey(1) == ord('q'):
        break

# When everything is done, release the capture and destroy windows
cap.release()
cv.destroyAllWindows()
print("Camera stream ended and resources released.")
    
