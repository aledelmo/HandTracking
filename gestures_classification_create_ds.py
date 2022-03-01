#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""


import cv2
import csv
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

NULL_LABEL = 0
OK_LABEL = 1
FIST_LABEL = 2

ds = []
idx = 0


def on_mouse_clicked(event, _x, _y, flags, param):
    global idx
    if event == cv2.EVENT_LBUTTONDOWN:
        current = {'ID': idx, 'gesture': flags, 'keypoints': (param - param[0]).tolist()}
        ds.append(current)
        idx += 1


if __name__ == '__main__':
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
            frame_landmarks = np.zeros(shape=(42, 3))

            if results.multi_hand_landmarks:
                if 0 < len(results.multi_hand_landmarks) < 3:
                    for j, hand in enumerate(results.multi_hand_landmarks):
                        for i, landmark in enumerate(hand.landmark):
                            frame_landmarks[21*j+i][0] = int(landmark.x * w)
                            frame_landmarks[21*j+i][1] = int(landmark.y * h)
                            frame_landmarks[21*j+i][2] = landmark.z

                    _max = np.amax(frame_landmarks, axis=0)[-1]
                    _min = np.amin(frame_landmarks, axis=0)[-1]

                    frame_landmarks[:, -1] = 255 * ((frame_landmarks[:, -1] - _min) / (_max - _min))

                    for lm in frame_landmarks:
                        cv2.circle(image, (int(lm[0]), int(lm[1])), 8, thickness=-1, color=[lm[2]] * 3)

            cv2.imshow('handDetector', cv2.flip(image, 1))
            #cv2.setMouseCallback('handDetector', on_mouse_clicked, param=frame_landmarks)

            if cv2.waitKey(1) & 0xFF == ord('n'):
                on_mouse_clicked(cv2.EVENT_LBUTTONDOWN, None, None, NULL_LABEL, frame_landmarks)
                print('null')
            if cv2.waitKey(1) & 0xFF == ord('o'):
                on_mouse_clicked(cv2.EVENT_LBUTTONDOWN, None, None, OK_LABEL, frame_landmarks)
                print('ok')
            if cv2.waitKey(1) & 0xFF == ord('f'):
                on_mouse_clicked(cv2.EVENT_LBUTTONDOWN, None, None, FIST_LABEL, frame_landmarks)
                print('fist')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

    csv_columns = ['ID', 'gesture', 'keypoints']
    with open('resources/ds.csv', "w") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=csv_columns)
        writer.writeheader()
        for data in ds:
            writer.writerow(data)
