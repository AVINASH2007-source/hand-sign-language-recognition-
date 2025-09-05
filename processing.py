import json
import numpy as np

# Define the hand signs (this should match what you defined in Step 1)
hand_signs = ['hello', 'thank_you', 'yes', 'no','OK','Peace','Good Job','A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Ensure this matches your capturing step

# Load the landmark data
with open('hand_signs/landmarks.json', 'r') as f:
    landmark_data = json.load(f)

# Prepare features (X) and labels (y)
X = []
y = []
label_map = {sign: index for index, sign in enumerate(hand_signs)}

for sign, landmarks_list in landmark_data.items():
    for landmarks in landmarks_list:
        # Flatten the landmark list and normalize
        flat_landmarks = np.array(landmarks).flatten()
        X.append(flat_landmarks)
        y.append(label_map[sign])

X = np.array(X)
y = np.array(y)

# Save X and y for training
np.save('hand_signs/X.npy', X)
np.save('hand_signs/y.npy', y)

print("Data preparation complete. Features and labels saved.")
