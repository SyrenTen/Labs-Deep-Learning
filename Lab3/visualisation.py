import matplotlib.pyplot as plt
import numpy as np
from TensorKeras_lab3 import train_and_evaluate_tf
from Torch_lab3 import train_and_evaluate_torch
import torch

tf_model, x_test = train_and_evaluate_tf()
torch_model, test_loader = train_and_evaluate_torch()

# TensorFlow
tf_predictions = tf_model.predict(x_test[:5])
for i in range(5):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"TensorKeras: {tf_predictions[i].argmax()}")
    plt.axis('off')

# PyTorch
sample_images, sample_labels = next(iter(test_loader))
torch_predictions = torch_model(sample_images[:5]).argmax(dim=1)
for i in range(5):
    plt.subplot(2, 5, i + 6)
    plt.imshow(sample_images[i].squeeze(), cmap='gray')
    plt.title(f"PyTorch: {torch_predictions[i].item()}")
    plt.axis('off')

plt.show()
