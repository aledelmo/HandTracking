# #!/usr/bin/env python
# # coding:utf-8
#
# """
#
# Author : Alessandro Delmonte
# Contact : alessandro.delmonte@institutimagine.org
# """
#
# import cv2
# import sys
from datetime import datetime
# # from PySide6 import QtCore, QtWidgets, QtGui
#
#
# # class MyWidget(QtWidgets.QWidget):
# #     def __init__(self):
# #         super().__init__()
# #
# #         self.filename_in = None
# #         self.filename_out = None
# #         self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
# #                                                  QtWidgets.QSizePolicy.MinimumExpanding))
# #
# #         self.button_in = QtWidgets.QPushButton("/home", self)
# #         self.text_in = QtWidgets.QLineEdit("test")
# #         self.button_out = QtWidgets.QPushButton("/home", self)
# #         self.button_analyse = QtWidgets.QPushButton("Analyse", self)
# #
# #         self.progress = QtWidgets.QProgressBar()
# #         self.progress.setFormat("{}%".format(0))
# #         self.progress.setTextVisible(True)
# #
# #         self.layout = QtWidgets.QFormLayout(self)
# #
# #         self.layout.addRow(self.tr("&Input Video:"), self.button_in)
# #         self.layout.addRow(self.tr("&Output Video:"), self.button_out)
# #         self.layout.addRow(self.button_analyse)
# #         self.layout.addRow(self.progress)
# #         self.layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
# #
# #         self.button_in.clicked.connect(self.choose_file_in)
# #         self.button_out.clicked.connect(self.choose_file_out)
# #         self.button_analyse.clicked.connect(self.analyse)
# #
# #     @QtCore.Slot()
# #     def choose_file_in(self):
# #         self.filename_in = QtWidgets.QFileDialog.getOpenFileName(self,
# #                                                                  self.tr("Open video"), "/home",
# #                                                                  self.tr("Videos (*.mov *.mp4 )"))[0]
# #         if self.filename_in:
# #             self.button_in.setText(self.filename_in)
# #
# #     @QtCore.Slot()
# #     def choose_file_out(self):
# #         self.filename_out = QtWidgets.QFileDialog.getOpenFileName(self,
# #                                                                   self.tr("Open video"), "/home",
# #                                                                   self.tr("Videos (*.mov *.mp4 )"))[0]
# #         if self.filename_out:
# #             self.button_out.setText(self.filename_out)
# #
# #     @QtCore.Slot()
# #     def analyse(self):
# #         cap = cv2.VideoCapture(self.filename_in)
# #         if not cap.isOpened():
# #             print("Error opening video stream or file")
# #
# #         while cap.isOpened():
# #             success, image = cap.read()
# #             if not success:
# #                 print("Ignoring empty camera frame.")
# #
# #             print("{} - In: {} - Out: {}".format(datetime.now(), self.filename_in, self.filename_out))
# #             for i in range(1, 101):
# #                 self.progress.setValue(i)
# #
# #
# # if __name__ == "__main__":
# #     app = QtWidgets.QApplication([])
# #
# #     widget = MyWidget()
# #     widget.show()
# #
# #     sys.exit(app.exec())
# import sys
# from PySide6.QtCore import QStandardPaths, Qt, Slot
# from PySide6.QtGui import QAction, QIcon, QKeySequence
# from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow, QSlider, QStyle, QToolBar, QStatusBar, QMessageBox
# from PySide6.QtMultimedia import (QAudioOutput, QMediaFormat,
#                                   QMediaPlayer)
# from PySide6.QtMultimediaWidgets import QVideoWidget
# from PySide6.QtNetwork import QUdpSocket, QHostAddress, QAbstractSocket
#
#
# AVI = "video/x-msvideo"  # AVI
#
#
# MP4 = 'video/mp4'
#
#
# def get_supported_mime_types():
#     result = []
#     for f in QMediaFormat().supportedFileFormats(QMediaFormat.Decode):
#         mime_type = QMediaFormat(f).mimeType()
#         result.append(mime_type.name())
#     return result
#
#
# class MainWindow(QMainWindow):
#
#     def __init__(self):
#         super().__init__()
#
#         self._playlist = []  # FIXME 6.3: Replace by QMediaPlaylist?
#         self._playlist_index = -1
#         self._audio_output = QAudioOutput()
#         self._player = QMediaPlayer()
#         self._player.setAudioOutput(self._audio_output)
#
#         self._player.errorOccurred.connect(self._player_error)
#
#         self.status_bar = QStatusBar()
#         self.setStatusBar(self.status_bar)
#
#         tool_bar = QToolBar()
#         self.addToolBar(tool_bar)
#
#         file_menu = self.menuBar().addMenu("&File")
#         icon = QIcon.fromTheme("document-open")
#         open_action = QAction(icon, "&Open...", self,
#                               shortcut=QKeySequence.Open, triggered=self.open)
#         file_menu.addAction(open_action)
#         tool_bar.addAction(open_action)
#         connect_action = QAction(icon, "&Connect...", self,
#                               shortcut=QKeySequence.Copy, triggered=self.connect)
#         file_menu.addAction(connect_action)
#         tool_bar.addAction(connect_action)
#         icon = QIcon.fromTheme("application-exit")
#         exit_action = QAction(icon, "E&xit", self,
#                               shortcut="Ctrl+Q", triggered=self.close)
#         file_menu.addAction(exit_action)
#
#         play_menu = self.menuBar().addMenu("&Play")
#         style = self.style()
#         icon = QIcon.fromTheme("media-playback-start.png",
#                                style.standardIcon(QStyle.SP_MediaPlay))
#         self._play_action = tool_bar.addAction(icon, "Play")
#         self._play_action.triggered.connect(self._player.play)
#         play_menu.addAction(self._play_action)
#
#         icon = QIcon.fromTheme("media-skip-backward-symbolic.svg",
#                                style.standardIcon(QStyle.SP_MediaSkipBackward))
#         self._previous_action = tool_bar.addAction(icon, "Previous")
#         self._previous_action.triggered.connect(self.previous_clicked)
#         play_menu.addAction(self._previous_action)
#
#         icon = QIcon.fromTheme("media-playback-pause.png",
#                                style.standardIcon(QStyle.SP_MediaPause))
#         self._pause_action = tool_bar.addAction(icon, "Pause")
#         self._pause_action.triggered.connect(self._player.pause)
#         play_menu.addAction(self._pause_action)
#
#         icon = QIcon.fromTheme("media-skip-forward-symbolic.svg",
#                                style.standardIcon(QStyle.SP_MediaSkipForward))
#         self._next_action = tool_bar.addAction(icon, "Next")
#         self._next_action.triggered.connect(self.next_clicked)
#         play_menu.addAction(self._next_action)
#
#         icon = QIcon.fromTheme("media-playback-stop.png",
#                                style.standardIcon(QStyle.SP_MediaStop))
#         self._stop_action = tool_bar.addAction(icon, "Stop")
#         self._stop_action.triggered.connect(self._ensure_stopped)
#         play_menu.addAction(self._stop_action)
#
#         self._volume_slider = QSlider()
#         self._volume_slider.setOrientation(Qt.Horizontal)
#         self._volume_slider.setMinimum(0)
#         self._volume_slider.setMaximum(100)
#         available_width = self.screen().availableGeometry().width()
#         self._volume_slider.setFixedWidth(available_width / 10)
#         self._volume_slider.setValue(self._audio_output.volume())
#         self._volume_slider.setTickInterval(10)
#         self._volume_slider.setTickPosition(QSlider.TicksBelow)
#         self._volume_slider.setToolTip("Volume")
#         self._volume_slider.valueChanged.connect(self._audio_output.setVolume)
#         tool_bar.addWidget(self._volume_slider)
#
#         about_menu = self.menuBar().addMenu("&About")
#         about_qt_act = QAction("About &Qt", self, triggered=qApp.aboutQt)
#         about_menu.addAction(about_qt_act)
#
#         self._video_widget = QVideoWidget()
#         self.setCentralWidget(self._video_widget)
#         self._player.playbackStateChanged.connect(self.update_buttons)
#         self._player.setVideoOutput(self._video_widget)
#
#         self.update_buttons(self._player.playbackState())
#         self._mime_types = []
#
#     def closeEvent(self, event):
#         self._ensure_stopped()
#         event.accept()
#
#     @Slot()
#     def open(self):
#         self._ensure_stopped()
#         file_dialog = QFileDialog(self)
#
#         is_windows = sys.platform == 'win32'
#         if not self._mime_types:
#             self._mime_types = get_supported_mime_types()
#             if is_windows and AVI not in self._mime_types:
#                 self._mime_types.append(AVI)
#             elif MP4 not in self._mime_types:
#                 self._mime_types.append(MP4)
#
#         file_dialog.setMimeTypeFilters(self._mime_types)
#
#         default_mimetype = AVI if is_windows else MP4
#         if default_mimetype in self._mime_types:
#             file_dialog.selectMimeTypeFilter(default_mimetype)
#
#         movies_location = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
#         file_dialog.setDirectory(movies_location)
#         if file_dialog.exec() == QDialog.Accepted:
#             url = file_dialog.selectedUrls()[0]
#             self._playlist.append(url)
#             self._playlist_index = len(self._playlist) - 1
#             self._player.setSource(url)
#             self._player.play()
#
#     @Slot()
#     def connect(self):
#         self.status_bar.showMessage("{} - Connecting to udp://@:8554 ...".format(datetime.now()))
#
#         '''
#         cap = cv2.VideoCapture("udp://@:8554", cv2.CAP_FFMPEG)
#         while cap.isOpened():
#             success, image = cap.read()
#             if not success:
#                 self.status_bar.showMessage("{} - Ignoring empty camera frame".format(datetime.now()))
#             else:
#                 self.status_bar.showMessage("{} - Streaming".format(datetime.now()))
#         '''
#         udpSocket = QUdpSocket(self)
#         udpSocket.readyRead.connect(self.read_pending_datagrams)
#         udpSocket.errorOccurred.connect(self.display_error)
#
#         udpSocket.connectToHost(QHostAddress.LocalHost, 8554)
#
#         #udpSocket.bind(QHostAddress.LocalHost, 8554)
#         # if udpSocket.waitForConnected(5000):
#         #     self.status_bar.showMessage("Connected!")
#         # else:
#         #     self.status_bar.showMessage("Connection not available!", 5000)
#         #connect(udpSocket, QUdpSocket.readyRead,
#         #        self, Server::readPendingDatagrams)
#
#     def read_pending_datagrams(self):
#         pass
#
#     def display_error(self, socketError):
#         if socketError == QAbstractSocket.RemoteHostClosedError:
#             pass
#         elif socketError == QAbstractSocket.HostNotFoundError:
#             QMessageBox.information(self, "Client", "The host was not found. Please check the host name and "
#                                                     "port settings.")
#         elif socketError == QAbstractSocket.ConnectionRefusedError:
#             QMessageBox.information(self, "Client", "The connection was refused by the peer. Make sure the fortune "
#                                                     "server is running, and check that the host name and port settings "
#                                                     "are correct.")
#         else:
#             reason = self._tcp_socket.errorString()
#             QMessageBox.information(self, "Fortune Client", f"The following error occurred: {reason}.")
#
#     @Slot()
#     def _ensure_stopped(self):
#         if self._player.playbackState() != QMediaPlayer.StoppedState:
#             self._player.stop()
#
#     @Slot()
#     def previous_clicked(self):
#         # Go to previous track if we are within the first 5 seconds of playback
#         # Otherwise, seek to the beginning.
#         if self._player.position() <= 5000 and self._playlist_index > 0:
#             self._playlist_index -= 1
#             self._playlist.previous()
#             self._player.setSource(self._playlist[self._playlist_index])
#         else:
#             self._player.setPosition(0)
#
#     @Slot()
#     def next_clicked(self):
#         if self._playlist_index < len(self._playlist) - 1:
#             self._playlist_index += 1
#             self._player.setSource(self._playlist[self._playlist_index])
#
#     def update_buttons(self, state):
#         media_count = len(self._playlist)
#         self._play_action.setEnabled(media_count > 0 and state != QMediaPlayer.PlayingState)
#         self._pause_action.setEnabled(state == QMediaPlayer.PlayingState)
#         self._stop_action.setEnabled(state != QMediaPlayer.StoppedState)
#         self._previous_action.setEnabled(self._player.position() > 0)
#         self._next_action.setEnabled(media_count > 1)
#
#     def show_status_message(self, message):
#         self.statusBar().showMessage(message, 5000)
#
#     @Slot(QMediaPlayer.Error, str)
#     def _player_error(self, error, error_string):
#         print(error_string, file=sys.stderr)
#         self.show_status_message(error_string)
#
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     main_win = MainWindow()
#     available_geometry = main_win.screen().availableGeometry()
#     main_win.resize(available_geometry.width() / 3,
#                     available_geometry.height() / 2)
#     main_win.show()
#     sys.exit(app.exec())

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
        #self.goproCamera.livestream('start')
        #self.goproCamera.video_settings(res="1080p", fps="30")
        #self.goproCamera.gpControlSet(constants.Stream.WINDOW_SIZE, constants.Stream.WindowSize.R720)

        # tts = gtts.gTTS("Enregistrement", lang="fr")
        # tts.save("tts.mp3")

    def set_file(self, fname):
        self.trained_file = tf.keras.models.load_model(os.path.join('trained_models', fname))

    def run(self):
        self.cap = cv2.VideoCapture("udp://10.5.5.100:8554", cv2.CAP_FFMPEG)
        while self.status:
            # lum = 1
            tracking = False
            with mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                                          min_tracking_confidence=0.5) as hands:
                while self.cap.isOpened():
                    success, image = self.cap.read()
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
                            to_predict = np.reshape(to_predict, (1, to_predict.shape[0]))
                            predict_result = self.trained_file.predict(to_predict)

                            cv2.putText(image, self.actions[np.argmax(predict_result)], (w - int(_min1[0]),
                                                                                         int(_min1[1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                            to_predict = frame_landmarks[21:, :2].flatten()
                            to_predict = np.reshape(to_predict, (1, to_predict.shape[0]))
                            predict_result2 = self.trained_file.predict(to_predict)

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
        #sys.exit(-1)


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
