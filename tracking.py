import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, c = image.shape
        lmList = []

        if results.multi_hand_landmarks:
            for myHand in results.multi_hand_landmarks:
                for lm in myHand.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([cx, cy, lm.z])

            lmList = np.array(lmList)

            _max = np.amax(lmList, axis=0)[-1]
            _min = np.amin(lmList, axis=0)[-1]

            lmList[:, -1] = 255 * ((lmList[:, -1] - _min) / (_max - _min))

            for lm in lmList:
                x, y, z = int(lm[0]), int(lm[1]), lm[2]
                cv2.circle(image, (x, y), 8, thickness=-1, color=[z] * 3)
            lmList = []

        cv2.imshow('handDetector', cv2.flip(image, 1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
