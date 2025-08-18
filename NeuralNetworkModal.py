"""
Enhanced Neural Network utilities for IoMT Federated Learning
Contains improved model definition and related functions
"""

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore
import tensorflow as tf


def SimpleNeuralNetwork(input_size, learning_rate=0.001):
    """
    Create an enhanced deep neural network for binary classification
    Args:
        input_size: Number of input features
        learning_rate: Learning rate for Adam optimizer (must be float)
    Returns:
        Compiled Keras model with improved architecture
    """
    # Ensure learning_rate is float to avoid ValueError
    learning_rate = float(learning_rate)
    
    model = Sequential([
        # Input layer with normalization
        Dense(128, input_shape=(input_size,), activation='relu', 
              kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


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


def HighPerformanceMedicalNetwork(input_size, learning_rate=0.0008):
    """
    Create an extremely powerful neural network for high accuracy medical classification
    """
    learning_rate = float(learning_rate)
    
    model = Sequential([
        # Large input layer to capture all feature interactions
        Dense(1024, input_shape=(input_size,), activation='relu',
              kernel_regularizer=l2(0.00005)),
        BatchNormalization(),
        Dropout(0.4),
        
        # Deep hidden layers for complex pattern recognition
        Dense(512, activation='relu', kernel_regularizer=l2(0.00005)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu', kernel_regularizer=l2(0.00005)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.00005)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.00005)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.00005)),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Advanced optimizer configuration
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0  # Gradient clipping
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall', 'auc']
    )
    
    return model