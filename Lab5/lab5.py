import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()  # завантаження датасету

# нормалізація
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def build_vgg13():
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    return model


vgg13 = build_vgg13()  # модель

# компіляція
vgg13.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# тренування
history = vgg13.fit(x_train, y_train, epochs=35, batch_size=64,
                    validation_data=(x_test, y_test))

# оцінка
test_loss, test_acc = vgg13.evaluate(x_test, y_test)
print(f'Точність: {test_acc * 100:.2f}%')


def plot_sample_predictions(model, x_test, y_test):
    predictions = model.predict(x_test[:10])
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for i, ax in enumerate(axes.flat):
        ax.imshow(x_test[i])
        pred_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[np.argmax(y_test[i])]
        ax.set_title(f'True: {true_label}\n Predict: {pred_label}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


plot_sample_predictions(vgg13, x_test, y_test)
