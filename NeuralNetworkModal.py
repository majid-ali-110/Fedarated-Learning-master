"""
Neural Network utilities for IoMT Federated Learning
Contains model definition and related functions
"""

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.optimizers import SGD  # type: ignore


def SimpleNeuralNetwork(input_size, learning_rate=0.01):
    """
    Create a simple neural network for binary classification
    Args:
        input_size: Number of input features
        learning_rate: Learning rate for SGD optimizer (must be float)
    Returns:
        Compiled Keras model
    """
    # Ensure learning_rate is float to avoid ValueError
    learning_rate = float(learning_rate)
    
    model = Sequential([
        Dense(1, input_shape=(input_size,), activation='sigmoid')
    ])
    model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model