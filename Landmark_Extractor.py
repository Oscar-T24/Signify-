import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np


LANDMARK_NB = 21
FILE_TO_READ = 'new_hand_landmarks_data.csv'
# Initialize MediaPipe Hands and Drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hands model with default parameters
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
last_frame = 0
try:
    df_existing = pd.read_csv(FILE_TO_READ)
    last_frame = df_existing['Vid'].iloc[-1]  # Get the last Frame ID
except (pd.errors.EmptyDataError, FileNotFoundError,IndexError):
    last_frame = 0  # If no data exists, start from 0
col = ['Vid'] 
for i in range(LANDMARK_NB):
    col += ['x'+str(i) ,'y'+str(i)]
col += ['Word']

# Initialize webcam
cap = cv2.VideoCapture(0)
i = 0
recording = False
ready = False
frameCaptured = 0
correspondingInt = 0 # int corresponding to sign
list_to_export = []
video_list = []
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
            if  recording:
                if i % 10 == 0:
                    print(frameCaptured)
                    i = 0
                    frameCaptured+=1
                    landmark_list = [last_frame]
                    for id, landmark in enumerate(hand_landmarks.landmark):
                        height, width, _ = frame.shape
                        cx, cy = landmark.x , landmark.y 
                        landmark_list += [cx,cy]
                    landmark_list+= [correspondingInt]
                    video_list.append(np.copy(landmark_list))
                    #print(landmark_list)
                    if frameCaptured >=5 :
                        recording = False
                        frameCaptured = 0
                        list_to_export += video_list
                        video_list = []
                        print("RECORDING STOPPED")
                i +=1
            
    # Show the resulting frame
    cv2.imshow("Hand Landmarks", frame)
    # Break the loop when 'q' is pressed
    
    key = cv2.waitKey(1)& 0xFF
    if key == ord('q'):
        df = pd.DataFrame(list_to_export, columns=col)
        df.to_csv(FILE_TO_READ, mode='a', header=not pd.io.common.file_exists(FILE_TO_READ),
                  index=False)
        break
    elif key == ord(' '):
        if not recording:
            recording = True
            last_frame +=1 
            i = 0
            print("RECORDING STARTED")
    elif key > 47 and key < 58:
        correspondingInt = key - 48
        print(correspondingInt)

        
    
            
# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
