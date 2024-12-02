import tensorflow as tf
from tensorflow.keras import layers, models, datasets, regularizers
import numpy as np
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def residual_block(input_tensor, filters, stride=1):
    shortcut = input_tensor

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding='same', kernel_regularizer=regularizers.l2(1e-4))(
        input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or input_tensor.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


# ResNet
def build_resnet34():
    input_layer = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(64, (3, 3), strides=1, padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ResNet блоки
    for filters, blocks, stride in [(64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)]:
        for block in range(blocks):
            x = residual_block(x, filters, stride if block == 0 else 1)

    x = layers.GlobalAveragePooling2D()(x)
    output_layer = layers.Dense(10, activation='softmax')(x)

    return models.Model(inputs=input_layer, outputs=output_layer)


# MirroredStrategy для GPU
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_resnet34()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# навчання
history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')

predictions = np.argmax(model.predict(x_test[:10]), axis=-1)
true_labels = np.argmax(y_test[:10], axis=-1)

plt.figure(figsize=(12, 8))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f'True: {true_labels[i]}, Pred: {predictions[i]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
