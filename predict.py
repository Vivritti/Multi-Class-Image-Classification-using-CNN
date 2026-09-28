"""
predict.py - Inference Script for Single Image Classification
"""

import argparse
import os
import numpy as np
import tensorflow as tf


def predict_image(image_path, model_path, img_size=128):
    """
    Loads a saved model and predicts the class of an input image.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess image
    img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) if predictions.shape[-1] > 1 else predictions[0]

    predicted_class_idx = np.argmax(predictions[0])
    confidence = 100 * np.max(score)

    print(f"\nPrediction Results:")
    print(f"Predicted Class Index: {predicted_class_idx}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict class for a single image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
    parser.add_argument('--model_path', type=str, default='model_weights.keras', help="Path to saved model")
    parser.add_argument('--img_size', type=int, default=128, help="Target image size")
    args = parser.parse_args()
    predict_image(args.image_path, args.model_path, args.img_size)
