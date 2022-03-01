#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""

import os
import sys
import time
import cv2
# import gtts
from playsound import playsound
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox,
                               QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget, QStatusBar,
                               QToolBar)
from goprocam import GoProCamera, constants


class Thread(QThread):
    updateFrame = Signal(QImage)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        self.cap = True
        self.actions = {0: 'palm', 1: 'OK', 2: 'fist'}
        self.goproCamera = GoProCamera.GoPro()
        print(self.goproCamera.infoCamera())
        #self.goproCamera.livestream('start')
        #self.goproCamera.video_settings(res="1080p", fps="30")
        #self.goproCamera.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)

        # tts = gtts.gTTS("Enregistrement", lang="fr")
        # tts.save("tts.mp3")

    def set_file(self, fname):
        self.interpreter = tf.lite.Interpreter(model_path='trained_models/final_model.tflite')
        self.interpreter.allocate_tensors()

        self.signatures = self.interpreter.get_signature_list()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        #self.trained_file = tf.keras.models.load_model(os.path.join('trained_models', fname))

    def run(self):
        self.cap = cv2.VideoCapture("udp://127.0.0.1:10000", cv2.CAP_FFMPEG)
        while self.status:
            # lum = 1
            tracking = False
            with mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5) as hands:
                while self.cap.isOpened():
                    success, image = self.cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                    else:

                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(image)

                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        h, w, c = image.shape
                        print(image.shape)
                        frame_landmarks = np.zeros(shape=(42, 3))

                        if results.multi_hand_landmarks:
                            if 0 < len(results.multi_hand_landmarks) < 3:
                                for j, hand in enumerate(results.multi_hand_landmarks):
                                    for i, landmark in enumerate(hand.landmark):
                                        frame_landmarks[21 * j + i][0] = int(landmark.x * w)
                                        frame_landmarks[21 * j + i][1] = int(landmark.y * h)
                                        frame_landmarks[21 * j + i][2] = landmark.z * w

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
                                self.interpreter.set_tensor(self.input_index, np.expand_dims(to_predict, axis=0).astype(np.float32))
                                self.interpreter.invoke()
                                predict_result = self.interpreter.get_tensor(self.output_index)[0]

                                cv2.putText(image, self.actions[np.argmax(predict_result)], (w - int(_min1[0]),
                                                                                             int(_min1[1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                                to_predict = frame_landmarks[21:, :2].flatten()
                                self.interpreter.set_tensor(self.input_index,
                                                            np.expand_dims(to_predict, axis=0).astype(np.float32))
                                self.interpreter.invoke()
                                predict_result2 = self.interpreter.get_tensor(self.output_index)[0]

                                cv2.putText(image, self.actions[np.argmax(predict_result2)], (w - int(_min2[0]),
                                                                                              int(_min2[1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                                if len(results.multi_hand_landmarks) == 2:
                                    if np.argmax(predict_result2) == 2 and np.argmax(predict_result) == 0:
                                        cv2.putText(image, "RECORDING", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                    (0, 0, 255), 3)
                                        if not tracking:
                                            playsound("tts.mp3", block=False)
                                        tracking = True
                                    else:
                                        tracking = False
                                else:
                                    tracking = False


                        else:
                            image = cv2.flip(image, 1)

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        img = QImage(image.data, w, h, c * w, QImage.Format_RGB888)
                        #scaled_img = img.scaled(640, 480, Qt.KeepAspectRatio)
                        self.updateFrame.emit(img)
        sys.exit(-1)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and dimensions
        self.setWindowTitle("Patterns detection")
        self.setGeometry(0, 0, 800, 500)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        #
        tool_bar = QToolBar()
        self.addToolBar(tool_bar)

        # Main menu bar
        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit = QAction("Exit", self, triggered=qApp.quit)
        self.menu_file.addAction(exit)

        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About Qt", self, shortcut=QKeySequence(QKeySequence.HelpContents),
                        triggered=qApp.aboutQt)
        self.menu_about.addAction(about)

        # Create a label for the display camera
        self.label = QLabel(self)
        #self.label.setFixedSize(640, 480)

        # Thread in charge of updating the image
        self.th = Thread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.set_image)

        # Model group
        self.group_model = QGroupBox("Trained model")
        self.group_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_layout = QHBoxLayout()

        self.combobox = QComboBox()
        for file in os.listdir("trained_models"):
            if file.endswith(".hdf5"):
                self.combobox.addItem(file)

        model_layout.addWidget(QLabel("File:"), 10)
        model_layout.addWidget(self.combobox, 90)
        self.group_model.setLayout(model_layout)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop/Close")
        self.button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.group_model, 1)
        right_layout.addLayout(buttons_layout, 1)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(right_layout)

        # Central widget
        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Connections
        self.button1.clicked.connect(self.start)
        self.button2.clicked.connect(self.kill_thread)
        self.button2.setEnabled(False)
        self.combobox.currentTextChanged.connect(self.set_model)

    @Slot()
    def set_model(self, text):
        self.th.set_file(text)

    @Slot()
    def kill_thread(self):
        print("Finishing...")
        self.button2.setEnabled(False)
        self.button1.setEnabled(True)
        self.th.cap.release()
        cv2.destroyAllWindows()
        self.status = False
        self.th.terminate()
        # Give time for the thread to finish
        time.sleep(5)

    @Slot()
    def start(self):
        print("Starting...")
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        self.th.set_file(self.combobox.currentText())
        self.th.start()

    @Slot(QImage)
    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
