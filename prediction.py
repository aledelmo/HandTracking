#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = tf.keras.models.load_model('trained_models/keypoint_classifier.hdf5')

actions = {0: 'palm', 1: 'OK', 2: 'fist'}

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    lum = 1
    tracking = False
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
                            frame_landmarks[21*j+i][2] = landmark.z * w

                    _max = np.amax(frame_landmarks, axis=0)
                    _min = np.amin(frame_landmarks, axis=0)
                    frame_landmarks[:, -1] = 255 * ((frame_landmarks[:, -1] - _min[-1]) / (_max[-1] - _min[-1]))

                    for lm in frame_landmarks:
                        cv2.circle(image, (int(lm[0]), int(lm[1])), 8, thickness=-1, color=[lm[2]] * 3)

                    _max1 = np.amax(frame_landmarks[:21], axis=0)
                    _min1 = np.amin(frame_landmarks[:21], axis=0)
                    cv2.rectangle(image, (int(_min1[0]), int(_min1[1])), (int(_max1[0]), int(_max1[1])),
                                         (255, 0, 0), 3)

                    _max2 = np.amax(frame_landmarks[21:], axis=0)
                    _min2 = np.amin(frame_landmarks[21:], axis=0)
                    cv2.rectangle(image, (int(_min2[0]), int(_min2[1])), (int(_max2[0]), int(_max2[1])),
                                         (255, 0, 0), 3)

                    image = cv2.flip(image, 1)

                    frame_landmarks -= frame_landmarks[0]

                    to_predict = frame_landmarks[:21, :2].flatten()
                    to_predict = np.reshape(to_predict, (1, to_predict.shape[0]))
                    predict_result = model.predict(to_predict)

                    cv2.putText(image, actions[np.argmax(predict_result)], (w - int(_min1[0]), int(_min1[1])),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    to_predict = frame_landmarks[21:, :2].flatten()
                    to_predict = np.reshape(to_predict, (1, to_predict.shape[0]))
                    predict_result2 = model.predict(to_predict)

                    cv2.putText(image, actions[np.argmax(predict_result2)], (w - int(_min2[0]), int(_min2[1])),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                    if len(results.multi_hand_landmarks) == 2 :
                        if np.argmax(predict_result2) == 2 and np.argmax(predict_result) == 0:
                            if tracking:
                                image += lum
                                lum += 5
                            tracking = True
                        else:
                            tracking = False
                            lum = 1
                    else:
                        tracking = False
                        lum = 1

            else:
                image = cv2.flip(image, 1)

            cv2.imshow('handDetector', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
