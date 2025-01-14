import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(1)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def add_gradient_background(img):
    gradient = np.linspace(0, 1, imgSize)
    gradient = np.tile(gradient, (imgSize, 1))
    gradient = np.expand_dims(gradient, axis=2)
    gradient = np.concatenate((gradient, gradient, gradient), axis=2) * 255
    gradient = gradient.astype(np.uint8)
    return cv2.addWeighted(img, 0.5, gradient, 0.5, 0)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y - offset):min(img.shape[0], y + h + offset),
                      max(0, x - offset):min(img.shape[1], x + w + offset)]

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

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        confidence = prediction[index]

        imgWhite = add_gradient_background(imgWhite)

        overlay = imgOutput.copy()
        cv2.rectangle(overlay, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), -1)
        cv2.putText(overlay, f'{labels[index]}: {confidence:.2f}', (x, y - 26), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(overlay, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 2)
        imgOutput = cv2.addWeighted(overlay, 0.5, imgOutput, 0.5, 0)


    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)

    if key == ord('q'):  # Added key press to quit
        break

cap.release()
cv2.destroyAllWindows()
