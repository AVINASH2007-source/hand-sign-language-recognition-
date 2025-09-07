# hand-sign-language-recognition-
This is a trained model for hand sign language recognition. This is created to bridge the communication gap between deaf/mute individuals and those who do not understand sign language.

# 🤟 Sign Language Recognition System

A real-time Sign Language Recognition System that uses computer vision and machine learning to detect and translate hand gestures into readable text. This project aims to bridge the communication gap between hearing-impaired individuals and the wider community.

---

## 📌 Features

- 🖐️ Real-time hand gesture recognition
- 🔤 Translation of gestures to English text
- 💻 User-friendly interface (CLI/GUI)
- 🧠 ML/DL model trained on sign language datasets
- 🌐 Can be integrated into websites, mobile apps, or smart devices

---

## 🧠 Technologies Used

- **Programming Language:** Python
- **Computer Vision:** OpenCV, MediaPipe
- **Machine Learning:** TensorFlow / PyTorch / Scikit-learn (choose based on your project)
- **Dataset:** American Sign Language (ASL) / Indian Sign Language / Custom dataset
- **Other Tools:** NumPy, Matplotlib, Tkinter (for GUI), pyttsx3 (for TTS)


---
FOR IMPLEMENTING THE PROJECT FIRST U HAVE TO IMPORT : hand sign model.pkl which is in the master branch that i have uploaded using LFS since it is a large file
---

This is a non trained model u can train it in your way by using any hand sign languages like American hand sign language, Indian sign language etc..
For more precise recognition u have to train it by a large data sets.
STEPS FOR TRAINING THE MODEL :
1. Exectute capturing.py
2. This will capture your inputs thorugh your camera by using opencv
3. click s to capture and r to stop for the each inputs
4. Then execute training.py .This will train those inputs to the model
5. Then execute processing.py.This will convert those coordinate points into json files and creates a separate arrangement of hand signs
6. After completing all these process u can execute recognition.py . This will recognise your hand signs and gives the desired output

