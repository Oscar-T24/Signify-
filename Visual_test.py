import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

PARAM_NB = 42
LOADFILE = 'weights/RNN1.weights.h5'
INT_TO_WORD = ['NaN','Yes','No','Thank You', 'Hello', 'I love you', 'Goodbye', 'You are welcome', 'Please','Sorry']

class Queue:
    def __init__(self,size):
        self.size = size
        #fuck circular array, I aint doing that sht again
        self.arr = [np.zeros(PARAM_NB) for i in range(size)]
        self.iter = 0
    def add(self,el):
        try :
            self.arr[self.iter] = el
            self.iter +=1
        except IndexError :
            self.arr = self.arr[1:]
            self.arr.append(el)
# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands model with default parameters
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define RNN Model
model = keras.Sequential([
    keras.Input(shape = (5,42)),
    layers.SimpleRNN(64, activation="relu", return_sequences=True),
    layers.SimpleRNN(32, activation="relu"),  # Second RNN layer
    layers.Dense(10, activation="softmax", name = 'outputLayer')  # Output layer for classification
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
if LOADFILE != None:
    model.load_weights(LOADFILE)

# Initialize webcam
cap = cv2.VideoCapture(0)


i = 0
correspondingInt = 0 # int corresponding to sign

lastFiveFrames = Queue(5)
while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image frame to extract hand landmarks
    results = hands.process(frame)

    # Convert image back to BGR for displaying with OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Check if landmarks were detected
    if results.multi_hand_landmarks:
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            
            # Extract the landmark positions
           
        if i % 10 == 0:
            i = 0
            landmark_list = []
            for id, landmark in enumerate(hand_landmarks.landmark):
                height, width, _ = frame.shape
                cx, cy = landmark.x , landmark.y 
                landmark_list += [cx,cy]
            lastFiveFrames.add(np.copy(landmark_list))
            prediction = model.predict(np.expand_dims(lastFiveFrames.arr, axis=0),verbose = 0).tolist()[0]
            if (max(prediction) > 0.97 and prediction.index(max(prediction))!= 0):
                print(INT_TO_WORD[prediction.index(max(prediction))])

            #print(landmark_list)
                
                
        i +=1
            
    # Show the resulting frame
    cv2.imshow("Hand Landmarks", frame)
    # Break the loop when 'q' is pressed
    
    key = cv2.waitKey(1)& 0xFF
    if key == ord('q'):
        break

        
    
            
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
