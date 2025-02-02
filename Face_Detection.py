import cv2
from deepface import DeepFace
from gtts import gTTS
import playsound
import os
from txt_to_speech import speech

import os

FACES = {}

for file_path in os.listdir("img/"):
    if file_path.endswith((".jpg", ".png", ".jpeg")):
        name = os.path.splitext(file_path)[0].capitalize()
        FACES[f"img/{file_path}"] = name

# Load pre-trained deep learning model (SSD)
net = cv2.dnn.readNetFromCaffe(
    "weights/deploy.prototxt", "weights/res10_300x300_ssd_iter_140000.caffemodel"
)

cap = cv2.VideoCapture(0)
spam = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x, y, x1, y1 = box.astype("int")

            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

            dfs = DeepFace.find(
            img_path = frame ,
            db_path = "img/",
            enforce_detection = False,
            silent = True
            )
            try :
                text = FACES[dfs[0]['identity'][0]]
                if spam >= 20:
                    spam = 0
                    print(text)
                    speech(text + " is in front of you")
                    
            except :
                pass
            spam +=1
    cv2.imshow("Deep Learning Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
