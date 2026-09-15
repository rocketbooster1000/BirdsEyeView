import cv2

# Load the image
image = cv2.imread('modifyscara.png')

# Get original dimensions
height, width = image.shape[:2]

# Define stretch factor (e.g., 2.0 = double the width)
stretch_factor = 2.0

# Calculate new dimensions
new_width = int(width * stretch_factor)
new_height = height  # Keep height the same

# Resize (stretch) the image
stretched = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Save or display
cv2.imwrite('output.jpg', stretched)
cv2.imshow('Stretched Image', stretched)
cv2.waitKey(0)
cv2.destroyAllWindows()