import tensorflow as tf
from tensorflow import keras
import random
import os
import numpy as np

os.environ['PYTHONHASHSEED']=str(1)
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)       
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import sys
print("python version", sys.version)
print("TF version", tf.__version__)
if tf.test.is_gpu_available():
  print("GPU available")
else:
  print("GPU unavailable")

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print(X_train_full.shape, X_test.shape)

X_valid, X_train = X_train_full[:3000] / 255., X_train_full[30000:] / 255.
y_valid, y_train = y_train_full[:3000], y_train_full[30000:]
X_test = X_test / 255.

# do not change
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print(X_valid.shape, X_train.shape)
# do not change

from functools import partial
DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")
# do not change
model = keras.models.Sequential([
    DefaultConv2D(filters=32, kernel_size=3, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=64),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(units=10, activation='softmax'),
])
# do not change
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="Adam", 
              metrics=["accuracy"])
model.summary()
# do not change

history = model.fit(X_train, y_train, epochs=20, 
                    batch_size=32,
                    validation_data=(X_valid, y_valid))

# do not change
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

org_acc = model.evaluate(X_test, y_test)  # do not change
print(org_acc[1]*100)

from keras import backend as K

org_model_size = np.sum([K.count_params(w) for w in model.trainable_weights]) # do not change
print(org_model_size)

# original FashionMNIST input shape is [28,28,1]
# if needed, data augmentation can be done here
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
from keras.models import Model, load_model

DefaultConv2D = partial(keras.layers.Conv2D,
                        kernel_size=3, activation='relu', padding="SAME")

# 수정된 모델(기존 모델보다 정확도 0.87% 향상, 기존 모델 대비 parameter 크기 69.32%)
your_model = tf.keras.Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',
               input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.20),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.30),

        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.40),

        keras.layers.Flatten(),

        Dense(64, activation='relu'),
        keras.layers.Dropout(0.20),

        Dense(units = 10, activation='softmax')
    ])

your_model.compile(loss="sparse_categorical_crossentropy",
                   optimizer="Adam",
                   metrics=["accuracy"])

your_model.summary()
your_model_size = np.sum([K.count_params(w) for w in your_model.trainable_weights])

hist=your_model.fit(X_train, y_train, epochs=20, batch_size = 64, validation_data=(X_valid, y_valid))

yours=your_model.evaluate(X_test, y_test)

print("[Acc] performance improvement: %.2f percent" % (yours[1]*100 - org_acc[1]*100))
print("[Size] size ratio: %.2f percent" % ((your_model_size / org_model_size)*100) )
if (yours[1]*100 - org_acc[1]*100) > 0:
  print("Accuracy resolved")
if (your_model_size / org_model_size)*100 < 70:
  print("Size resolved")
