import cv2
import mediapipe as mp
import time
import math as math
import pygame
import numpy as np
import sys
import pandas as pd
import numpy as np

class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):

                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmsList, bbox

    def findFingerUp(self):
        fingers = []

        if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):

        x1, y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0.255), cv2.FILLED)
        len = math.hypot(x2 - x1, y2 - y1)

        return len, frame, [x1, y1, x2, y2, cx, cy]


class LoadCV:
    def __init__(self):
        """Initialize the camera and hand tracking."""
        self.ptime = 0
        self.cap = cv2.VideoCapture(0)
        self.detector = HandTrackingDynamic()  # Initialize the hand tracking module
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit()

    def get_frame(self):
        """Captures a frame from OpenCV, applies hand tracking, and returns it as a Pygame surface."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = self.detector.findFingers(frame)  # Process frame with hand tracking
        lmsList = self.detector.findPosition(frame)
        #if len(lmsList) != 0:
            #print("PSOITIONS",lmsList[0])

        ctime = time.time()
        fps = 1 / (ctime - self.ptime) if self.ptime != 0 else 0
        self.ptime = ctime

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        # Convert BGR to RGB for Pygame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)  # Rotate to match Pygame's orientation
        frame = np.flip(frame, axis=0)  # Flip vertically

        frame_surface = pygame.surfarray.make_surface(frame)
        return frame_surface

    def record(self,counter:int)->pd.DataFrame:
        '''Returns the hand pose in the form of a list of landmarks [id, x, y] where x,y are normalised coordinates
        for ONE frame
        '''
        ret, frame = self.cap.read()
        if not ret:
            return None

        normalized_data = []

        if len(self.detector.findPosition(frame)) == 0:
            "no hand has been identified"
            return None

        landmarks, bbox = self.detector.findPosition(frame)

        for lm_pos in landmarks:
            xmin, ymin, xmax, ymax = bbox # bounding box coordinates
            normalized_x = (lm_pos[1] - xmin) / (xmax - xmin)
            normalized_y = (lm_pos[2] - ymin) / (ymax - ymin)

            normalized_data.append([counter, lm_pos[0], normalized_x, normalized_y])

        df = pd.DataFrame(normalized_data, columns=['Frame', 'Landmark_ID', 'Normalized_X', 'Normalized_Y'])

        df.to_csv('hand_landmarks_data.csv', mode='a', header=not pd.io.common.file_exists('hand_landmarks_data.csv'),
                  index=False)

        print("RECORDED")
        return df


    def release(self):
        """Release the camera and close OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    LoadCV.main()