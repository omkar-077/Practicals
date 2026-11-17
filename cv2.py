import cv2

img = cv2.imread("image.jpg")

if img is None:
    print("Image not found!")
    exit()

while True:
    print("\n--- IMAGE PROCESSING ---")
    print("1. Load Image")
    print("2. Crop Image")
    print("3. Resize Image")
    print("4. Thresholding")
    print("5. Contour Analysis")
    print("6. Blob Detection")
    print("7. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        cv2.imshow("Original Image", img)

    elif choice == "2":
        crop = img[100:300, 100:300]
        cv2.imshow("Cropped Image", crop)

    elif choice == "3":
        resize = cv2.resize(img, (300, 300))
        cv2.imshow("Resized Image", resize)

    elif choice == "4":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )
        cv2.imshow("Threshold Image", thresh)

    elif choice == "5":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(
            gray, 127, 255, cv2.THRESH_BINARY
        )

        contours, h = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        output = img.copy()
        cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
        cv2.imshow("Contours", output)

    elif choice == "6":
        gray = cv2.imread("image.jpg", 0)

        detector = cv2.SimpleBlobDetector_create()
        keypoints = detector.detect(gray)

        output = cv2.drawKeypoints(
            gray, keypoints, None, (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        cv2.imshow("Blob Detection", output)

    elif choice == "7":
        cv2.destroyAllWindows()
        break

    else:
        print("Invalid choice!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()