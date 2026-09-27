## Project Description
This project implements a convolutional neural network for handwritten digit classification using PyTorch and the MNIST dataset. It includes image preprocessing, model training, model-weight persistence, and inference for individual image files.

## Core Systems
* **Data Preprocessing:** Loads MNIST images as grayscale tensors and normalizes them using the dataset's standard mean and standard deviation. Inference images are converted to grayscale, inverted, resized to 28x28 pixels, and normalized in the same way.
* **Convolutional Neural Network (`SimpleMNISTNet`):** Applies two convolutional layers with ReLU activations and max pooling, followed by fully connected layers that produce scores for the 10 digit classes.
* **Training Engine:** Trains with the Adam optimizer and cross-entropy loss. The default run uses 5 epochs, a batch size of 64, and a learning rate of 0.001. It automatically selects CUDA, Apple MPS, or CPU when available.
* **Model Persistence (`mnist_model.pth`):** Saves the trained model's state dictionary so it can be loaded for inference without retraining.
* **Inference Pipeline:** Loads an image file, predicts a digit from 0 to 9, and prints the predicted class and its confidence.

## Usage
Install the dependencies:
```bash
pip install -r requirements.txt
```

Train the model (MNIST is downloaded to `data/` if needed):
```bash
python main.py
```

Predict a digit from an image:
```bash
python predict.py path/to/image.png
```
The image is resized to 28x28 pixels and expected to show a dark digit on a light background. If no path is provided, the script uses `test.png`.