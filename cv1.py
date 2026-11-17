import cv2
import numpy as np
print("OpenCV version:", cv2.__version__)


# Create a blank white image (height, width, 3 colour channels)
image = np.ones((300, 400, 3), dtype=np.uint8) * 255


# Draw a blue rectangle and a green circle
cv2.rectangle(image, (50, 50), (350, 250), (255, 0, 0), 3)
cv2.circle(image, (200, 150), 60, (0, 255, 0), -1)
cv2.imshow("My First OpenCV Window", image)
cv2.waitKey(0)
cv2.destroyAllWindows()