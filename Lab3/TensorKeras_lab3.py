# використання tensorflow/keras

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


def create_lenet5_tf():
    model = models.Sequential([
        layers.Conv2D(6, kernel_size=5, strides=1, activation='tanh', input_shape=(28, 28, 1), padding='same'),
        layers.AvgPool2D(pool_size=2, strides=2),
        layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh'),
        layers.AvgPool2D(pool_size=2, strides=2),
        layers.Conv2D(120, kernel_size=5, strides=1, activation='tanh'),
        layers.Flatten(),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])
    return model


def train_and_evaluate_tf():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = create_lenet5_tf()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"TensorFlow Model Accuracy: {accuracy * 100:.2f}%")
    return model, x_test


if __name__ == "__main__":
    model, x_test = train_and_evaluate_tf()
