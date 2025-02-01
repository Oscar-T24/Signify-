import cv2
import mediapipe as mp
import time
import math


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

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)

        return frame

    def findPosition(self, frame, draw=True):
        all_landmarks = []  # List to store landmarks for all detected hands
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
                print(f"Hand {handNo + 1} Keypoint")
                print(bbox)
                if draw:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

                all_landmarks.append((handData, bbox))  # Store data for this hand

        return all_landmarks

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


def main(video_path):
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(video_path)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print(f"end of video {video_path}. closing window...")
            cap.release()
            break

        frame = detector.findFingers(frame)
        all_hands_data = detector.findPosition(frame)

        # Process multiple hands
        if len(all_hands_data) > 0:
            for handNo, (landmarks, bbox) in enumerate(all_hands_data):
                print(f"Hand {handNo + 1} Landmarks:", landmarks)
                print(f"Hand {handNo + 1} Bounding Box:", bbox)

        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('frame', frame)

        # Close window on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Check if 'q' was pressed to quit
            print("Closing window...")
            cap.release()
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main('B_01_062.mkv')
