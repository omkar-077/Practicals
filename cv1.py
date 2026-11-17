import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# Create a blank white image
image = np.ones((300, 400, 3), dtype=np.uint8) * 255



cv2.rectangle(image, (50, 50), (350, 250), (255, 0, 0), 3)
cv2.circle(image, (200, 150), 60, (0, 255, 0), -1)
cv2.line(image, (20, 20), (380, 280), (0, 0, 255), 3)
cv2.ellipse(image, (200, 150), (100, 40),
            0, 0, 360, (0, 255, 255), 3)
cv2.putText(image, "Hello OpenCV", (80, 290),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), 2)


# Display image
cv2.imshow("Image Annotation", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

