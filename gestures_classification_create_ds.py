#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""

import os
import cv2
import csv
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
labels = {"n": 0,
          "a": 1,
          "o": 2}


def play_video(video_filepath, ds_filepath):
    video = cv2.VideoCapture(video_filepath)
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    ds = []
    with mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while video.isOpened():
            success, image = video.read()
            try:
                if not success:
                    print("Ignoring empty camera frame.")
                else:
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
                                    frame_landmarks[21 * j + i][0] = int(landmark.x * w)
                                    frame_landmarks[21 * j + i][1] = int(landmark.y * h)
                                    frame_landmarks[21 * j + i][2] = landmark.z

                            _max = np.amax(frame_landmarks, axis=0)[-1]
                            _min = np.amin(frame_landmarks, axis=0)[-1]

                            frame_landmarks[:, -1] = 255 * ((frame_landmarks[:, -1] - _min) / (_max - _min))

                            for lm in frame_landmarks:
                                cv2.circle(image, (int(lm[0]), int(lm[1])), 8, thickness=-1, color=[lm[2]] * 3)

                cv2.imshow("video", image)

                pressed_key = cv2.waitKey(00) & 0xFF
                if pressed_key == ord('q'):
                    break
                elif pressed_key == ord('n') or pressed_key == ord('a') or pressed_key == ord('o'):
                    pressed = str(chr(pressed_key % 256)) if pressed_key % 256 < 128 else '?'
                    ds.append({'ID': len(ds), 'gesture': labels[pressed],
                               'keypoints': (frame_landmarks - frame_landmarks[0]).tolist()})
            except:
                pass

    video.release()
    csv_columns = ['ID', 'gesture', 'keypoints']
    with open(ds_filepath, "w") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=csv_columns)
        writer.writeheader()
        for data in ds:
            writer.writerow(data)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_training = os.path.join("resources", "GOPR1385.MP4")
    ds_path = os.path.join("resources", "ds.csv")
    play_video(video_training, ds_path)
