import cv2
import mediapipe as mp
import time
import math
import pandas as pd
from pathlib import Path
import sys
import sys
import os
import logging

# Redirect stderr
sys.stderr = open(os.devnull, 'w')

# Set the logging level to ERROR for TensorFlow and Mediapipe
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('cv2').setLevel(logging.ERROR)


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(max_num_hands=self.__maxHands__, min_detection_confidence=self.__detectionCon__,
                                        min_tracking_confidence=self.__trackCon__)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        self.frame_count = 0
        self.two_hand_count = 0
        self.results = None

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, draw=True):
        all_landmarks = []
        if self.results.multi_hand_landmarks:
            for handNo, handLms in enumerate(self.results.multi_hand_landmarks):
                xList = []
                yList = []
                bbox = []
                handData = []

                for id, lm in enumerate(handLms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    handData.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax
                all_landmarks.append((handData, bbox))
        return all_landmarks

    def get_two_hand_percentage(self):
        if self.frame_count == 0:
            return 0
        return (self.two_hand_count / self.frame_count) * 100


def main(video_path, threshold=10, name=""):
    ctime = 0
    ptime = 0
    frame_count = 0
    cap = cv2.VideoCapture(video_path)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    normalized_data = []

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"End of video {video_path}. Closing window...")
            cap.release()
            break

        frame = detector.findFingers(frame)
        all_hands_data = detector.findPosition(frame)

        if len(all_hands_data) > 0:
            for handNo, (landmarks, bbox) in enumerate(all_hands_data):
                # Only write the first hand's data
                if handNo == 0:  # Skip all but the first hand
                    # Normalize and append the data for the first hand
                    for lm_pos in landmarks:
                        xmin, ymin, xmax, ymax = bbox
                        normalized_x = (lm_pos[1] - xmin) / (xmax - xmin)
                        normalized_y = (lm_pos[2] - ymin) / (ymax - ymin)

                        word = name  # Placeholder for the "word" column (you can dynamically change this)

                        normalized_data.append([frame_count, lm_pos[0], normalized_x, normalized_y, word])

        frame_count += 1
        detector.frame_count += 1
        if len(all_hands_data) > 1:
            detector.two_hand_count += 1  # Count frames with two hands

        # Check if the percentage of frames with more than one hand exceeds the threshold
        if detector.get_two_hand_percentage() > threshold:
            print(f"Skipping video due to more than one hand detected frequently.")
            return  # Skip this video if the threshold is exceeded

        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('frame', frame)

    # After video is finished, write the data to CSV only once
    if normalized_data:

        df = pd.DataFrame(normalized_data, columns=['Frame', 'Landmark_ID', 'Normalized_X', 'Normalized_Y', 'Phrase'])

        # Save DataFrame to CSV, append if file exists
        df.to_csv('hand_landmarks_data2.csv', mode='a', header=not pd.io.common.file_exists('hand_landmarks_data2.csv'),
                  index=False)

def export():
    '''Exports the recorded hand landmarks to a Dataframe for analysis (note different formatting than csv)'''
    df = pd.read_csv("hand_landmarks_data2.csv")
    df_pivoted = df.pivot_table(index='Frame', columns='Landmark_ID', values=['Normalized_X', 'Normalized_Y'])

    # Flatten the column multi-index (Normalized_X, Normalized_Y, Landmark_ID)
    df_pivoted.columns = [f'{col[0]}{col[1]}' for col in df_pivoted.columns]

    # Reset the index so 'Frame' becomes a column
    df_final = df_pivoted.reset_index()
    return df_final

if __name__ == "__main__":

    print(export())
    time.sleep(33333)

    import logging

    # Set logging to show only errors
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('mediapipe').setLevel(logging.ERROR)

    # get the video by name and find corresponding phrase (lemma ID) in the csv
    for video in Path("videos").rglob("*.mkv"):
        file_name = Path(video).stem
        file = pd.read_csv("asl_database.csv")
        phrase = file[file['Code'] == file_name]["LemmaID"]
        if phrase.empty:
            print(f"NO MATCHES for {file_name}")
            continue
        print(phrase.iloc[0])
        main(video, name=phrase.iloc[0])
