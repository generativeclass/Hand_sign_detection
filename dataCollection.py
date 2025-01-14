import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize webcam and hand detector
cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)

# Constants for image processing
offset = 20
imgSize = 300
folder = "Data/Z"   // write the destination folder to store data 
counter = 0

# Create folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera.")
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Adjust cropping coordinates
        y_start = max(0, y - offset)
        y_end = min(img.shape[0], y + h + offset)
        x_start = max(0, x - offset)
        x_end = min(img.shape[1], x + w + offset)

        imgCrop = img[y_start:y_end, x_start:x_end]
        if imgCrop.size == 0:
            print("Cropped image is empty.")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        cv2.imshow("Cropped Image", imgCrop)
        cv2.imshow("White Image", imgWhite)

    cv2.imshow("Original Image", img)

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")
    elif key == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
