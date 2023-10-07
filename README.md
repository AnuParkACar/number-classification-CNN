## Project Description

This project comprises two main files, `main.py` and `CNN.py`, which collectively create and evaluate a Convolutional Neural Network (CNN) for image classification tasks using PyTorch. Below is an overview of each file and their functionalities:

### `main.py`

#### Purpose
`main.py` serves as the entry point for the project. It allows you to configure and run the CNN on different datasets (currently supports MNIST and CIFAR-10) and provides performance metrics.

#### Features
- Initializes the CNN model with user-defined parameters.
- Loads the chosen dataset (MNIST or CIFAR-10) and preprocesses the data.
- Trains the model using a training dataset and evaluates it on a validation dataset.
- Computes and prints accuracy, score, and runtime metrics.

#### Usage
To run the code with default settings, execute the `main.py` script. You can also modify the `Args` class within `main.py` to change dataset selection and GPU usage preferences.

### `CNN.py`

#### Purpose
`CNN.py` contains the definition of the CNN model, dataset loading functions, and training procedures.

#### Features
- Defines a CNN model using PyTorch's `nn.Module`.
- Provides functions to load and preprocess datasets (MNIST and CIFAR-10).
- Implements training, validation, and regularization procedures.

#### Usage
This file is primarily utilized as a library by `main.py` for creating, training, and validating the CNN model.

## Getting Started

1. Ensure you have Python 3.x installed on your system.

2. Install the required dependencies by running the following command:

   ```
   pip install torch torchvision
   ```

3. Execute `main.py` to train and evaluate the CNN model. Modify the `Args` class in `main.py` to customize the dataset and GPU usage settings.

## Dataset Support

The current implementation supports two datasets: MNIST and CIFAR-10. You can select the dataset by modifying the `Args` class within `main.py`.

## Acknowledgments

This project is a basic implementation of a CNN for image classification. For more advanced applications, consider further customization and optimization.
