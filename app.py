#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""

import os
import sys
import cv2
import time
# import gtts
import numpy as np
import pandas as pd
import mediapipe as mp
import tensorflow as tf
from datetime import datetime
from playsound import playsound
from goprocam import GoProCamera, constants
from PySide6.QtCore import QThread, Signal, Slot, QAbstractTableModel, Qt, QModelIndex
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget, QStatusBar, QToolBar, QTableView, QHeaderView)


class KeepAliveThread(QThread):
    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.gopro = GoProCamera.GoPro()

    def run(self):
        self.gopro.stream("udp://127.0.0.1:10000")


class InferThread(QThread):
    updateFrame = Signal(QImage)
    updateText = Signal(str)
    updateTable = Signal(dict)

    def __init__(self, parent=None):
        QThread.__init__(self, parent)
        self.interpreter = None
        self.signatures = None
        self.input_index = None
        self.output_index = None
        self.status = True
        self.cap = True
        self.actions = {0: 'None', 1: 'Start', 2: 'Stop'}
        self.goproCamera = GoProCamera.GoPro()
        # tts = gtts.gTTS("Enregistrement", lang="fr")
        # tts.save("tts.mp3")

    def print_config(self):
        info = self.goproCamera.infoCamera()
        self.updateText.emit("Streaming {} ({})".format(info['model_name'], info['serial_number']))

    def print_message(self, message):
        self.updateText.emit(message)

    def set_file(self, fname):
        self.interpreter = tf.lite.Interpreter(model_path=fname)
        self.interpreter.allocate_tensors()

        self.signatures = self.interpreter.get_signature_list()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]

        self.updateText.emit("Model loaded: {}".format(fname))

    def run(self):
        self.cap = cv2.VideoCapture("udp://127.0.0.1:10000?overrun_nonfatal=1&fifo_size=300000", cv2.CAP_FFMPEG)
        while self.status:
            tracking = False
            tracking_start = None
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
                                frame_landmarks[:, -1] = 255 * ((frame_landmarks[:, -1] - _min[-1]
                                                                 ) / (_max[-1] - _min[-1]))

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
                                self.interpreter.set_tensor(self.input_index, np.expand_dims(to_predict,
                                                                                             axis=0).astype(np.float32))
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
                                        if not tracking:
                                            playsound("resources/tts.mp3", block=False)
                                            tracking_start = datetime.now()
                                            self.print_message("Tracking started at {}".format(tracking_start))
                                        tracking = True
                                else:
                                    if tracking:
                                        time_diff = datetime.now()-tracking_start
                                        time_diff_formatted = "{} minutes {} seconds".format(time_diff.seconds // 60,
                                                                                             time_diff.seconds % 60)
                                        self.print_message("Tracking lasted {}".format(time_diff_formatted))
                                        self.updateTable.emit({'Date Start': [tracking_start],
                                                               'Date End': [datetime.now()],
                                                               'Date Delta': [time_diff_formatted]})
                                    tracking = False

                                if tracking:
                                    cv2.putText(image, "RECORDING", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (0, 0, 255), 3)
                        else:
                            image = cv2.flip(image, 1)

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        img = QImage(image.data, w, h, c * w, QImage.Format_RGB888)
                        self.updateFrame.emit(img)
        sys.exit(-1)


class PandasModel(QAbstractTableModel):
    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent == QModelIndex():
            return len(self._dataframe)
        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def get_df(self):
        return self._dataframe

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])
            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])
        return None

    def appendRow(self, row, index=QModelIndex()):
        self.beginInsertRows(index, self.rowCount(), self.rowCount())
        self._dataframe = pd.concat([self._dataframe, pd.DataFrame(row)], ignore_index=True)
        self.endInsertRows()
        return True


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Patterns detection")
        self.setGeometry(0, 0, 800, 500)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        tool_bar = QToolBar()
        self.addToolBar(tool_bar)

        self.menu = self.menuBar()
        self.menu_file = self.menu.addMenu("File")
        exit = QAction("Exit", self, triggered=qApp.quit)
        self.menu_file.addAction(exit)

        self.menu_about = self.menu.addMenu("&About")
        about = QAction("About Qt", self, shortcut=QKeySequence(QKeySequence.HelpContents),
                        triggered=qApp.aboutQt)
        self.menu_about.addAction(about)

        self.label = QLabel(self)

        self.th = InferThread(self)
        self.th.finished.connect(self.close)
        self.th.updateFrame.connect(self.set_image)
        self.th.updateText.connect(self.set_text)
        self.th.updateTable.connect(self.append_row)

        self.th_alive = KeepAliveThread(self)
        self.th_alive.finished.connect(self.close)

        self.group_model = QGroupBox("Trained model")
        self.group_model.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        model_layout = QHBoxLayout()

        self.combobox = QComboBox()
        for file in os.listdir("trained_models"):
            if file.endswith(".tflite"):
                self.combobox.addItem(os.path.join("trained_models", file))

        model_layout.addWidget(QLabel("File:"), 10)
        model_layout.addWidget(self.combobox, 90)
        self.group_model.setLayout(model_layout)

        buttons_layout = QHBoxLayout()
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop/Close")
        self.button1.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.button2.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        buttons_layout.addWidget(self.button2)
        buttons_layout.addWidget(self.button1)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.group_model, 1)
        right_layout.addLayout(buttons_layout, 30)

        table_layout = QHBoxLayout()
        view = QTableView()
        view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        view.setAlternatingRowColors(True)
        view.setSelectionBehavior(QTableView.SelectRows)
        ds = {'Date Start': [], 'Date End': [], 'Date Delta': []}
        df = pd.DataFrame(ds)
        self.model = PandasModel(df)
        view.setModel(self.model)
        table_layout.addWidget(view, 1)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(right_layout)
        layout.addLayout(table_layout)

        widget = QWidget(self)
        widget.setLayout(layout)
        self.setCentralWidget(widget)

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
        self.th_alive.terminate()
        # Give time for the thread to finish
        time.sleep(5)

    @Slot()
    def start(self):
        print("Starting...")
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        self.th_alive.start()
        self.th.set_file(self.combobox.currentText())
        self.th.start()
        self.th.print_config()

    @Slot(QImage)
    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    @Slot(str)
    def set_text(self, text):
        self.status_bar.showMessage(text)

    @Slot(dict)
    def append_row(self, row):
        self.model.appendRow(row)


if __name__ == "__main__":
    app = QApplication()
    w = Window()
    w.show()
    sys.exit(app.exec())
