import cv2
import mediapipe as mp
import time
import math


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, hand_threshold=0.1):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(max_num_hands=self.__maxHands__, min_detection_confidence=self.__detectionCon__,
                                        min_tracking_confidence=self.__trackCon__)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        # Variables to track hand count occurrences
        self.frame_count = 0
        self.two_hand_count = 0
        self.hand_threshold = hand_threshold  # Threshold for error (percentage of frames with two hands)

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
                if draw:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

                all_landmarks.append((handData, bbox))  # Store data for this hand

        # Count frames and two-hand occurrences
        self.frame_count += 1
        if len(all_landmarks) > 1:
            self.two_hand_count += 1

        return all_landmarks

    def check_hand_threshold(self):
        if self.frame_count == 0:
            return False  # Avoid division by zero

        two_hand_ratio = self.two_hand_count / self.frame_count
        if two_hand_ratio > self.hand_threshold:
            print(f"⚠️ Error: Two hands detected in {two_hand_ratio * 100:.2f}% of frames. Exiting.")
            return True
        return False


def main(video_path):
    ctime = 0
    ptime = 0

    cap = cv2.VideoCapture(video_path)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open video")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video {video_path}. Closing window...")
            break

        frame = detector.findFingers(frame)
        all_hands_data = detector.findPosition(frame)

        # Check if too many frames have two hands
        if detector.check_hand_threshold():
            cap.release()
            cv2.destroyAllWindows()
            return

        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


        cv2.imshow('frame', frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Closing window...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main('B_01_062.mkv')
