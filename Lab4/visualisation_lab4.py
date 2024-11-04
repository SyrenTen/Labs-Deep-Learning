import tensorflow as tf
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from TensorKeras_lab4 import create_alexnet_tf
from PyTorch_lab4 import AlexNetTorch
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Load CIFAR-10 dataset for TensorFlow model
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test / 255.0

# Initialize and train the TensorFlow model
tf_model = create_alexnet_tf()
tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tf_model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# Initialize the PyTorch model and set it to evaluation mode
torch_model = AlexNetTorch()
torch_model.eval()

# Define transform for CIFAR-10 dataset in PyTorch
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Corrected mean and std values
])

# Visualize Predictions
tf_predictions = tf_model.predict(x_test[:5])
sample_images, sample_labels = next(iter(DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transform, download=True),
    batch_size=5
)))
torch_predictions = torch_model(sample_images).argmax(dim=1)

# Plotting predictions from both models
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"TensorF: {tf_predictions[i].argmax()}")
    plt.axis('off')

    plt.subplot(2, 5, i + 6)
    plt.imshow(sample_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5)
    plt.title(f"PyTorch: {torch_predictions[i].item()}")
    plt.axis('off')

plt.show()
