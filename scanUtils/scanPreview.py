import cv2
# pip install opencv-python pygrabber
from pygrabber.dshow_graph import FilterGraph

graph = FilterGraph()
devices = graph.get_input_devices()
for i, name in enumerate(devices):
    print(f"Index {i}: {name}")

# Then open by index
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)