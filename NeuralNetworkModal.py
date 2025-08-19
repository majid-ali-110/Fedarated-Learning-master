"""
Enhanced Neural Network utilities for IoMT Federated Learning
Optimized for medical data classification
"""

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore


def MedicalNeuralNetwork(input_size, learning_rate=0.001):
    """
    Create a powerful neural network optimized for medical data
    Args:
        input_size: Number of input features
        learning_rate: Learning rate for Adam optimizer
    Returns:
        Compiled Keras model optimized for medical classification
    """
    learning_rate = float(learning_rate)
    
    # Build a more powerful architecture
    model = Sequential([
        # First layer - capture complex medical patterns
        Dense(512, input_shape=(input_size,), activation='relu',
              kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Second layer
        Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third layer  
        Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Fourth layer
        Dense(64, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.2),
        
        # Fifth layer
        Dense(32, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Use advanced optimizer settings
    optimizer = Adam(
        learning_rate=learning_rate, 
        beta_1=0.9, 
        beta_2=0.999,
        epsilon=1e-7
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model