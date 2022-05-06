# HandTracking

Hand tracking and gestures recognition for keyframe selection during live-streaming. 

![tracking](https://i.imgur.com/ErB2m6d.gif)

The app connects to a GoPro over
a local Wi-Fi network, it streams the live video feed and it detects the presence of hands (with 21 hands key-points) 
using the [MediaPipe](https://google.github.io/mediapipe/) project. Finally, hand gestures are classified using a 
quantized TFLite model for low-latency predictions.

The predicted gestures may be used to start and stop recording of the stream during long sequences in situations where
the camera is not easily accessible (e.g. scrubbed doctor performing a long operation wanting to record certain
portions of the operation).

## Usage

Turn on the GoPro and connect the PC to the camera local Wi-Fi.

Create a training dataset annotating a pre-recorded video using a graphical interface:
```shell
$ python gestures_classification_create_ds.py
```

Train the gesture classification deep neural network:
```shell
$ python gestures_classification_training.py
```

Video streaming platform with low-latency infer:
```shell
$ python app.py
```

## System Requirements

GoPro Compatibility: HERO3, HERO3+, HERO4, HERO+, HERO5, HERO6, Fusion 1, HERO7 (Black), HERO8 Black, MAX, HERO9 Black

Everything works on CPU with minimal video delay.
Tested on MacBook Pro (15-inch, 2017)

![gopro](https://i.imgur.com/v6h6alf.png)

## Contacts

For any inquiries please contact: 
[Alessandro Delmonte](https://aledelmo.github.io) @ [alessandro.delmonte@institutimagine.org](mailto:alessandro.delmonte@institutimagine.org)

## License

This project is licensed under the [Apache License 2.0](LICENSE) - see the [LICENSE](LICENSE) file for
details
