#!/usr/bin/env python
# coding:utf-8

"""

Author : Alessandro Delmonte
Contact : alessandro.delmonte@institutimagine.org
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

NUM_CLASSES = 3
RANDOM_SEED = 42
model_save_path = 'keypoint_classifier.hdf5'

ds = pd.read_csv('ds.csv', converters={'keypoints': pd.eval})

x_dataset = ds['keypoints'].values.tolist()
y_dataset = ds['gesture'].values.tolist()

x_dataset = np.array(x_dataset)[:, :21, :2]
x_dataset = np.reshape(x_dataset, (x_dataset.shape[0], x_dataset.shape[1] * x_dataset.shape[2]))
y_dataset = np.array(y_dataset)

'--------------------------------'

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75,
                                                    random_state=RANDOM_SEED)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000, batch_size=128, validation_data=(x_test, y_test),
          callbacks=[cp_callback, es_callback])

val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)

model = tf.keras.models.load_model(model_save_path)

'--------------------------------'


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


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)

