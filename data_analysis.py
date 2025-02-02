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

        # Tracking hand count statistics
        self.frame_count = 0
        self.two_hand_count = 0

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
            num_hands = len(self.results.multi_hand_landmarks)
            self.frame_count += 1
            if num_hands > 1:
                self.two_hand_count += 1  # Count frames with two hands

            # Always process only the first detected hand (handNo=0)
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
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmsList, bbox

    def get_two_hand_percentage(self):
        if self.frame_count == 0:
            return 0
        return (self.two_hand_count / self.frame_count) * 100


def main(video_path):
    ptime =0
    ctime=0
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
            break

        frame = detector.findFingers(frame)
        lmsList, bbox = detector.findPosition(frame)

        # Display FPS
        # Calculate FPS
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        # Display FPS on the frame
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        #cv2.imshow('frame', frame)

        # Exit on pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Display percentage of frames with two hands
    two_hand_percentage = detector.get_two_hand_percentage()
    print(f"🖐️ Two hands detected in {two_hand_percentage:.2f}% of frames.")

    cap.release()
    cv2.destroyAllWindows()

    return two_hand_percentage


def analyze(video_path):
    """Processes the input video frame-by-frame and records hand landmarks."""
    video = main('B_01_062.mkv')  # Initialize video capture and hand tracking

    frame_counter = 0  # Keeps track of the frame index

    while True:
        ret, frame = video.cap.read()
        if not ret:
            break  # Stop processing when the video ends

        video.record(counter=frame_counter)  # Record hand landmarks for this frame
        frame_counter += 1  # Increment frame index

    video.release()  # Release resources
    print(f"Finished analyzing {video_path}. Recorded {frame_counter} frames.")



if __name__ == "__main__":
    main('B_01_062.mkv')
