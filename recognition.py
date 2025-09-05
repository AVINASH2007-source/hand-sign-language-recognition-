import cv2
import mediapipe as mp
import numpy as np
import joblib

# Define the hand signs
hand_signs = ['hello', 'thank_you', 'yes', 'no', 'OK', 'Peace', 'Good Job', 'A', 'B', 'C', 'D','E',
              'F', 'G', 'H', 'I','J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
              'W', 'X', 'Y', 'Z']  # Ensure this matches your training data

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Load the trained model
model = joblib.load('hand_sign_model.pkl')

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark coordinates
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            flat_landmarks = np.array(landmarks).flatten().reshape(1, -1)  # Prepare input for the model

            # Make a prediction
            prediction = model.predict(flat_landmarks)
            predicted_label = hand_signs[prediction[0]]

            # Define finger connections
            finger_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
                (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]

            # Draw landmarks and lines on the frame
            height, width, _ = img.shape
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            for connection in finger_connections:
                start_idx, end_idx = connection
                start_lm = hand_landmarks.landmark[start_idx]
                end_lm = hand_landmarks.landmark[end_idx]
                start_cx, start_cy = int(start_lm.x * width), int(start_lm.y * height)
                end_cx, end_cy = int(end_lm.x * width), int(end_lm.y * height)
                cv2.line(img, (start_cx, start_cy), (end_cx, end_cy), (0, 255, 0), 2)

            cv2.putText(img, predicted_label, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (500, 0, 0), 5)
    cv2.imshow("Hand Sign Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

