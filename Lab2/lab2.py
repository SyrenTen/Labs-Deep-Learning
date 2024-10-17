import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Завантаження mnist

# перетворення та нормалізація даних
x_train = x_train.reshape((x_train.shape[0], 28 * 28)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28 * 28)).astype('float32') / 255

# One-hot encoding мітки
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# будова моделі
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Компіляція

history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)  # тренування

# оцінка на тестовому наборі
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Прогноз на тест наборі
predictions = model.predict(x_test)


# фунція для відображення кількох зображень та їх прогнозованих міток
def plot_images(images, labels, preds, num=5):
    plt.figure(figsize=(10, 4))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {np.argmax(labels[i])}\nPred: {np.argmax(preds[i])}")
        plt.axis('off')
    plt.show()


# Відображення тестових зображень з справжніми та прогноз мітками
plot_images(x_test, y_test, predictions)
