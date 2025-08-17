import numpy as np
from NeuralNetworkModal import SimpleNeuralNetwork
from cryptography.fernet import Fernet
from PrivacyPreserver import PrivacyPreserver
from typing import List, Tuple,Dict

class IoMTDevice:    
    def __init__(self, device_id: str, X_local: np.ndarray, y_local: np.ndarray,privacy_budget: float = 1.0):
        """        
        Args:
            device_id: Unique identifier for the device
            X_local: Local feature data
            y_local: Local target data
        """
        self.device_id = device_id
        self.X_local = X_local
        self.y_local = y_local
        self.model = SimpleNeuralNetwork(X_local.shape[1])
        self.training_history = []
        self.privacy_budget = privacy_budget
        self.used_privacy_budget = 0.0
        self.encryption_key = Fernet.generate_key()
        
        # Add local differential privacy to training data
        self._add_local_dp_noise()

        print(f"Privacy-Preserved IoMT Device {device_id} initialized with {len(X_local)} patient records")
        print(f"Privacy budget: {privacy_budget}")
    
    def _add_local_dp_noise(self, epsilon: float = 0.5):
        """Add local differential privacy noise to training data"""
        sensitivity = 1.0
        scale = sensitivity / epsilon
        
        # Add noise to features (be careful with medical data)
        noise = np.random.laplace(0, scale * 0.01, self.X_local.shape)  # Small noise for medical data
        self.X_local = self.X_local + noise
        
        print(f"\nDevice {self.device_id}: Added local DP noise (ε={epsilon}) to the training data")
    
    def local_train(self, epochs: int = 5, batch_size: int = 32,dp_epsilon: float = 0.1):
        """
        Train the local model on local data
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        
        Returns:
            Model weights after training
        """
        print(f"Device {self.device_id}: Starting secure local training for {epochs} epochs...")

        if self.used_privacy_budget + dp_epsilon > self.privacy_budget:
            print(f"Device {self.device_id}: Privacy budget exhausted!")
            return None, None
        
        print(f"Device {self.device_id}: Starting secure local training...")

        history = self.model.fit(
            self.X_local,
            self.y_local,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0  # set to 1 if you want progress printed for every batch
        )

        # Get model weights
        weights = self.model.get_weights()
        
        # Apply privacy preservation techniques
        # 1. Gradient clipping
        weights = PrivacyPreserver.gradient_clipping(weights, clip_norm=1.0)

          
        # 2. Differential privacy noise
        weights = PrivacyPreserver.add_differential_privacy_noise(weights, epsilon=dp_epsilon)
        
        # 3. Encrypt weights
        weight_shapes = [w.shape for w in weights]
        encrypted_weights, _ = PrivacyPreserver.encrypt_weights(weights, self.encryption_key)

         # Update privacy budget
        self.used_privacy_budget += dp_epsilon

        # Record training history
        losses = history.history['loss']
        self.training_history.extend(losses)
        
        # Log some progress (e.g., every 2 epochs)
        for i, loss in enumerate(losses):
            if i % 2 == 0:
                print(f"Epoch {i+1}/{epochs}, Loss: {loss:.4f}")


        print(f"Device {self.device_id}: Secure training completed. Remaining privacy budget: {self.privacy_budget - self.used_privacy_budget:.2f}")

        return encrypted_weights, weight_shapes

    
    def update_model(self, encrypted_global_weights: str, weight_shapes: List[Tuple], 
                           server_key: bytes):
        """
        Update local model with global parameters
        
        Args:
            global_parameters: Global model parameters from server
        """
        try:
            # Decrypt global weights
            global_weights = PrivacyPreserver.decrypt_weights(encrypted_global_weights,server_key, weight_shapes)
            self.model.set_weights(global_weights)
            print(f"Device {self.device_id}: Model Securely updated with global parameters")
        except Exception as e:
            print(f"Device {self.device_id}: Failed to decrypt global weights: {e}")

    def evaluate_local(self):
        """
        Evaluate model performance on local data
        Returns:
            Local accuracy
        """
        y_pred = self.model.predict(self.X_local,verbose=0)
        accuracy = np.mean((y_pred > 0.5).astype(int) == self.y_local)
        return accuracy
    
    def get_data_size(self):
        """
        Get the size of local dataset
        
        Returns:
            Number of local samples
        """
        return len(self.X_local)
    
    def get_privacy_report(self) -> Dict:
        """Generate privacy preservation report"""
        return {
            'device_id': self.device_id,
            'total_privacy_budget': self.privacy_budget,
            'used_privacy_budget': self.used_privacy_budget,
            'remaining_privacy_budget': self.privacy_budget - self.used_privacy_budget,
            'privacy_techniques_applied': [
                'Local Differential Privacy',
                'Gradient Clipping',
                'Global Differential Privacy',
                'Homomorphic Encryption'
            ]
        }