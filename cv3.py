import cv2

# Load Image
img = cv2.imread("image.jpg")

if img is None:
    print("Image not found!")
    exit()

while True:

    print("\n--- IMAGE ANNOTATION ---")
    print("1. Draw Line")
    print("2. Add Text")
    print("3. Draw Circle")
    print("4. Draw Rectangle")
    print("5. Draw Ellipse")
    print("6. Exit")

    choice = input("Enter your choice: ")

    output = img.copy()

    if choice == "1":
        cv2.line(output, (50, 50), (300, 300),
                 (255, 0, 0), 3)
        cv2.imshow("Line", output)

    elif choice == "2":
        cv2.putText(output, "Hello OpenCV",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Text", output)

    elif choice == "3":
        cv2.circle(output, (200, 200), 80,
                   (0, 0, 255), 3)
        cv2.imshow("Circle", output)

    elif choice == "4":
        cv2.rectangle(output, (50, 50), (300, 250),
                      (255, 0, 255), 3)
        cv2.imshow("Rectangle", output)

    elif choice == "5":
        cv2.ellipse(output, (200, 200), (100, 50),
                    0, 0, 360, (0, 255, 255), 3)
        cv2.imshow("Ellipse", output)

    elif choice == "6":
        cv2.destroyAllWindows()
        break

    else:
        print("Invalid choice!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()