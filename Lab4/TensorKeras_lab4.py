import tensorflow as tf
from tensorflow.keras import layers, models


def create_alexnet_tf():
    model = models.Sequential([
        layers.Conv2D(64, kernel_size=3, strides=1, activation='relu', input_shape=(32, 32, 3), padding='same'),
        layers.MaxPool2D(pool_size=2, strides=2),  # Output: 16x16x64
        layers.Conv2D(128, kernel_size=3, strides=1, activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=2, strides=2),  # Output: 8x8x128
        layers.Conv2D(256, kernel_size=3, strides=1, activation='relu', padding='same'),
        layers.Conv2D(256, kernel_size=3, strides=1, activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=2, strides=2),  # Output: 4x4x256
        layers.Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='same'),
        layers.Conv2D(512, kernel_size=3, strides=1, activation='relu', padding='same'),
        layers.MaxPool2D(pool_size=2, strides=2),  # Output: 2x2x512
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model


# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize

# Compile and train model
model = create_alexnet_tf()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"TensorFlow Model Accuracy: {accuracy * 100:.2f}%")
