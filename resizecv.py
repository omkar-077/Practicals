
import cv2
import numpy as np

print("OpenCV version:", cv2.__version__)

# Create a blank image
image = np.ones((400, 500, 3), dtype=np.uint8) * 255

# 1. Load Image
cv2.imshow("Original Image", image)

# 2. Crop Image
crop = image[50:300, 50:350]
cv2.imshow("Cropped Image", crop)

# 3. Resize Image
resize = cv2.resize(image, (300, 300))
cv2.imshow("Resized Image", resize)

# 4. Thresholding
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(
    gray, 127, 255, cv2.THRESH_BINARY
)

cv2.imshow("Threshold Image", thresh)

# 5. Contour Analysis
contours, h = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()

cv2.rectangle(output, (50, 50), (350, 300), (0, 0, 0), -1)
gray2 = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

ret, thresh2 = cv2.threshold(
    gray2, 127, 255, cv2.THRESH_BINARY_INV
)

contours, h = cv2.findContours(
    thresh2, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
cv2.imshow("Contour Analysis", image)

# 6. Blob Detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(gray)

blob = cv2.drawKeypoints(
    gray, keypoints, None, (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

cv2.imshow("Blob Detection", blob)

# Wait and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()