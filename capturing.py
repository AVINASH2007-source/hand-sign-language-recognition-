import cv2
import mediapipe as mp
import numpy as np
import os
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Create a directory for storing the captured landmarks
os.makedirs('hand_signs', exist_ok=True)

# Define hand signs
hand_signs = ['hello', 'thank_you', 'yes', 'no','OK','Peace','Good Job','A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Extend this list as needed

# Create a JSON file to store landmark data
landmark_data = {sign: [] for sign in hand_signs}

# Loop through each hand sign
for sign in hand_signs:
    print(f"Prepare to show the sign for '{sign}'. Press 's' to start capturing, and 'q' to stop.")
    input("Press Enter when ready...")  # Wait for user to get ready

    cap = cv2.VideoCapture(0)
    
    capturing = False
    while True:
        ret, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                landmark_data[sign].append(landmarks)

                # Draw landmarks
                height, width, _ = img.shape
                for lm in hand_landmarks.landmark:
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Capturing", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to start capturing
            capturing = True
            print(f"Capturing {sign}... Press 'q' to stop.")
        elif key == ord('q'):  # Press 'q' to stop capturing
            if capturing:
                print(f"Stopped capturing for {sign}.")
            capturing = False
            break  # Exit capturing loop

    cap.release()
    cv2.destroyAllWindows()

# Save the landmark data to a JSON file
with open('hand_signs/landmarks.json', 'w') as f:
    json.dump(landmark_data, f)

print("Landmark data captured and saved.")
