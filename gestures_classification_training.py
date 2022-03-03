#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""

import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def parse_ds(_ds_filename):
    ds = pd.read_csv(_ds_filename, converters={'keypoints': pd.eval})

    x_dataset = ds['keypoints'].values.tolist()
    y_dataset = ds['gesture'].values.tolist()

    x_dataset = np.array(x_dataset)[:, :21, :2]
    x_dataset = np.reshape(x_dataset, (x_dataset.shape[0], x_dataset.shape[1] * x_dataset.shape[2]))
    y_dataset = np.array(y_dataset)

    _x_train, _x_test, _y_train, _y_test = train_test_split(x_dataset, y_dataset, train_size=0.75,
                                                            random_state=42)

    return _x_train, _x_test, _y_train, _y_test


def create_train_model(_x_train, _y_train, num_classes=3, quantization_aware_training=True):
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

    if quantization_aware_training:
        _model = tfmot.quantization.keras.quantize_model(_model)

    _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    _model.fit(_x_train, _y_train, epochs=1000, batch_size=128, validation_data=(x_test, y_test),
               callbacks=[es_callback])

    return _model


def convert_save_model(_model, post_float16_quant=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if post_float16_quant:
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path("trained_models/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir/"final_model_quant.tflite"
    tflite_model_file.write_bytes(tflite_model)


def evaluate_accuracy(_x_test, _y_test):
    interpreter = tf.lite.Interpreter(model_path='trained_models/final_model_quant.tflite')
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    y_pred = []
    for ex in _x_test:
        interpreter.set_tensor(input_index, np.expand_dims(ex, axis=0).astype(np.float32))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)[0]
        y_pred.append(np.argmax(predictions))

    print_confusion_matrix(_y_test, y_pred)


def print_confusion_matrix(y_true, _y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, _y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()

    if report:
        print('Classification Report')
        print(classification_report(y_test, _y_pred))


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = parse_ds('resources/ds.csv')
    model = create_train_model(x_train, y_train, num_classes=3, quantization_aware_training=True)
    convert_save_model(model, post_float16_quant=False)
    evaluate_accuracy(x_test, y_test)
