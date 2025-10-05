import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

def load_and_preprocess_data():
    """
    Load MNIST dataset and preprocess it.
    Returns: (X_train, y_train), (X_test, y_test)
    """
    # Load the dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to 0-1 range
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return (X_train, y_train), (X_test, y_test)

if __name__ == "__main__":
    # Test the loader
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()