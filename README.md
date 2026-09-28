# Multi-Class Image Classification using CNN 🧠🖼️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Supported-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning project for multi-class image classification using Convolutional Neural Networks (CNNs). This repository includes model architecture definition, dataset processing pipeline, training & validation workflows, performance evaluation metrics, and inference scripts.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Dataset Organization](#-dataset-organization)
- [Installation & Setup](#-installation--setup)
- [Usage & Execution](#-usage--execution)
- [Model Architecture](#-model-architecture)
- [Results & Evaluation](#-results--evaluation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌌 Overview

Convolutional Neural Networks (CNNs) are the state-of-the-art architecture for visual data processing. This project demonstrates how to build, train, and evaluate a multi-class image classification model capable of categorizing images into multiple distinct classes.

Whether using TensorFlow/Keras or PyTorch, this pipeline covers:
- Image loading, resizing, normalization, and data augmentation.
- Custom CNN layer configurations (Convolution, MaxPooling, Dropout, Dense).
- Hyperparameter tuning, optimization (Adam/SGD), and loss functions (Categorical Cross-Entropy).
- Model evaluation with Confusion Matrices, Classification Reports, and Training History Plots.

---

## ✨ Key Features

- **Data Preprocessing & Augmentation**: Automated rescaling, rotation, zoom, flips, and normalization to prevent overfitting.
- **Custom CNN Architecture**: Multi-layer convolutional architecture optimized for feature extraction and pattern recognition.
- **Modular Pipeline**: Clean separation between notebook exploration (`CNN.ipynb`) and production scripts (`train.py`, `predict.py`).
- **Comprehensive Evaluation**: Metrics including Accuracy, Precision, Recall, F1-Score, and Loss curves.
- **Model Checkpointing**: Save trained weights (`.h5` / `.keras` / `.pth`) for later deployment and inference.

---

## 📁 Repository Structure

```text
Multi-Class-Image-Classification-using-CNN/
├── CNN.ipynb             # Main Jupyter Notebook containing code, outputs & analysis
├── train.py              # Modular Python script for training the CNN model
├── predict.py            # Script for running inference on single or batch images
├── requirements.txt      # List of required Python packages and dependencies
├── .gitignore            # Git ignore file to exclude datasets, checkpoints & virtual environments
├── LICENSE               # Open-source MIT License
└── README.md             # Project documentation and guide
```

---

## 📊 Dataset Organization

Organize your dataset inside a `dataset/` directory structured as follows:

```text
dataset/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
├── validation/
│   ├── class_1/
│   ├── class_2/
│   └── class_3/
└── test/
    ├── class_1/
    ├── class_2/
    └── class_3/
```

> **Note**: Avoid uploading large dataset files directly to GitHub. Large dataset files and model weights (`.h5`, `.pth`, `.keras`) are excluded via `.gitignore`. You can host datasets on [Kaggle](https://www.kaggle.com/) or [Google Drive].

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Vivritti/Multi-Class-Image-Classification-using-CNN.git
   cd Multi-Class-Image-Classification-using-CNN
   ```

2. **Create a Virtual Environment** (Recommended):
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🚀 Usage & Execution

### 1. Running the Jupyter Notebook
For interactive exploration, visualization, and step-by-step training:
```bash
jupyter notebook CNN.ipynb
```

### 2. Training via Python Script
To launch training directly from the command line:
```bash
python train.py --data_dir ./dataset --epochs 25 --batch_size 32
```

### 3. Making Predictions / Inference
To classify a new unseen image:
```bash
python predict.py --image_path path/to/sample_image.jpg --model_path model_weights.keras
```

---

## 🏗️ Model Architecture

The CNN architecture typically consists of consecutive feature-extraction blocks followed by classification layers:

1. **Input Layer**: Preprocessed image tensors (e.g., `128x128x3` or `224x224x3`).
2. **Convolutional Layers (`Conv2D`)**: Extracts low-level to high-level spatial features using ReLU activation.
3. **Pooling Layers (`MaxPooling2D`)**: Downsamples spatial dimensions to reduce computational complexity and achieve translation invariance.
4. **Dropout Layers**: Randomly deactivates neurons during training to prevent overfitting.
5. **Flatten / Global Average Pooling**: Converts 2D feature maps into a 1D feature vector.
6. **Dense Layers (`Fully Connected`)**: High-level reasoning and class probability distribution via `Softmax` output.

---

## 📈 Results & Evaluation

During model training, metrics are logged and plotted:
- **Training vs. Validation Loss**: Tracks convergence and detects underfitting/overfitting.
- **Training vs. Validation Accuracy**: Measures model performance across epochs.
- **Confusion Matrix**: Visualizes per-class classification accuracy and misclassification patterns.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📜 License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

<p center>Made with ❤️ by <a href="https://github.com/Vivritti">Vivritti</a></p>
