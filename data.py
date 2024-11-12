import cv2
import mediapipe as mp
import numpy as np
import csv

# Prompt for label input
label = input("Enter the ASL letter label for this session: ")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam early and set resolution for better performance
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# CSV file to save landmarks
with open('asl_landmarks.csv', mode='w', newline='') as f:
    writer = csv.writer(f)
    # Header for landmark columns
    header = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
    writer.writerow(header)
    
    print("Press 's' to save a frame, and 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image. Exiting.")
            break

        # Flip and convert color for MediaPipe
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks and flatten them into a list
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Show frame with landmarks
                cv2.imshow('ASL Data Collection', frame)
                
                # Press 's' to save data
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    writer.writerow(landmarks + [label])
                    print(f"Frame saved with label '{label}'")
        
        else:
            print("No hand landmarks detected.")
        
        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting program.")
            break

cap.release()
hands.close()
cv2.destroyAllWindows()
