"""
train.py - Training Pipeline for Multi-Class Image Classification using CNN
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(input_shape, num_classes):
    """
    Constructs a Convolutional Neural Network architecture.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense Classifier Head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    return model


def main(args):
    img_height, img_width = args.img_size, args.img_size
    batch_size = args.batch_size
    epochs = args.epochs

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        print("Please structure your data into dataset/train and dataset/validation subdirectories.")
        return

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'validation')

    print("Loading datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir if os.path.exists(train_dir) else args.data_dir,
        validation_split=0.2 if not os.path.exists(val_dir) else None,
        subset="training" if not os.path.exists(val_dir) else None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir if os.path.exists(val_dir) else args.data_dir,
        validation_split=0.2 if not os.path.exists(val_dir) else None,
        subset="validation" if not os.path.exists(val_dir) else None,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # Autotune input pipelines for fast execution
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build and compile model
    model = build_cnn_model((img_height, img_width, 3), num_classes)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # Train
    print("\nStarting Training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Save trained model
    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
    model.save(args.output_model)
    print(f"Model successfully saved to {args.output_model}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN for Multi-Class Image Classification")
    parser.add_argument('--data_dir', type=str, default='./dataset', help="Path to dataset directory")
    parser.add_argument('--img_size', type=int, default=128, help="Target image height/width")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--output_model', type=str, default='model_weights.keras', help="Output model path")
    args = parser.parse_args()
    main(args)
