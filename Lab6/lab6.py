import tensorflow as tf
from tensorflow.keras import layers, models, Model, regularizers
from tensorflow.keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)


# GoogLeNet
def inception_module(x, filters):
    f1, f2_1, f2_3, f3_1, f3_5, f4 = filters

    conv1x1_1 = layers.Conv2D(f1, (1, 1), padding='same', activation='relu')(x)

    conv1x1_2 = layers.Conv2D(f2_1, (1, 1), padding='same', activation='relu')(x)
    conv3x3 = layers.Conv2D(f2_3, (3, 3), padding='same', activation='relu')(conv1x1_2)

    conv1x1_3 = layers.Conv2D(f3_1, (1, 1), padding='same', activation='relu')(x)
    conv5x5 = layers.Conv2D(f3_5, (5, 5), padding='same', activation='relu')(conv1x1_3)

    max_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    conv1x1_pool = layers.Conv2D(f4, (1, 1), padding='same', activation='relu')(max_pool)

    return layers.concatenate([conv1x1_1, conv3x3, conv5x5, conv1x1_pool], axis=-1)


def build_googlenet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Initial convolution and max-pooling layers
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception modules
    x = inception_module(x, [64, 96, 128, 16, 32, 32])
    x = inception_module(x, [128, 128, 192, 32, 96, 64])
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [192, 96, 208, 16, 48, 64])
    x = inception_module(x, [160, 112, 224, 24, 64, 64])
    x = inception_module(x, [128, 128, 256, 24, 64, 64])
    x = inception_module(x, [112, 144, 288, 32, 64, 64])
    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [256, 160, 320, 32, 128, 128])
    x = inception_module(x, [384, 192, 384, 48, 128, 128])

    # Average pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model


googlenet_model = build_googlenet((32, 32, 3), 10)
googlenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = googlenet_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64)

test_loss, test_accuracy = googlenet_model.evaluate(x_test, y_test, verbose=2)
print(f'Test Accuracy: {test_accuracy:.2f}')

predictions = googlenet_model.predict(x_test[:10])
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test[:10], axis=1)

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(x_test[i])
    axes[i].set_title(f'True: {true_labels[i]}, Pred: {predicted_labels[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
