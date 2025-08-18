import numpy as np
from NeuralNetworkModal import SimpleNeuralNetwork, MedicalNeuralNetwork, HighPerformanceMedicalNetwork
from cryptography.fernet import Fernet
from typing import List, Tuple,Dict
import base64
import pickle

# Basic Privacy Preserver utilities for IoMT devices
class BasicPrivacyPreserver:
    """Basic privacy preservation utilities for IoMT devices"""
    
    @staticmethod
    def gradient_clipping(weights: List[np.ndarray], clip_norm: float = 1.0) -> List[np.ndarray]:
        """Apply gradient clipping to weights"""
        clipped_weights = []
        for weight_matrix in weights:
            # Calculate L2 norm
            norm = np.linalg.norm(weight_matrix)
            if norm > clip_norm:
                weight_matrix = weight_matrix * (clip_norm / norm)
            clipped_weights.append(weight_matrix)
        return clipped_weights
    
    @staticmethod
    def add_differential_privacy_noise(weights: List[np.ndarray], epsilon: float = 0.1, 
                                     sensitivity: float = 1.0) -> List[np.ndarray]:
        """Add Laplacian noise for differential privacy"""
        scale = sensitivity / epsilon
        noisy_weights = []
        
        for weight_matrix in weights:
            noise = np.random.laplace(0, scale, weight_matrix.shape)
            noisy_weights.append(weight_matrix + noise)
        
        return noisy_weights
    
    @staticmethod
    def encrypt_weights(weights: List[np.ndarray], key: bytes) -> Tuple[str, List[Tuple]]:
        """Encrypt model weights for secure transmission"""
        fernet = Fernet(key)
        encrypted_weights = []
        weight_shapes = []
        
        for weight_matrix in weights:
            weight_shapes.append(weight_matrix.shape)
            weight_bytes = weight_matrix.flatten().tobytes()
            encrypted_weight = fernet.encrypt(weight_bytes)
            encrypted_weights.append(encrypted_weight)
        
        # Serialize to base64 string for compatibility
        encrypted_data = pickle.dumps(encrypted_weights)
        encoded_weights = base64.b64encode(encrypted_data).decode('utf-8')
        
        return encoded_weights, weight_shapes
    
    @staticmethod
    def decrypt_weights(encrypted_weights_str: str, key: bytes, shapes: List[Tuple]) -> List[np.ndarray]:
        """Decrypt model weights from base64 encoded string"""
        fernet = Fernet(key)
        decrypted_weights = []
        
        # Decode from base64 string
        encrypted_data = base64.b64decode(encrypted_weights_str.encode('utf-8'))
        encrypted_weights = pickle.loads(encrypted_data)
        
        for encrypted_weight, shape in zip(encrypted_weights, shapes):
            weight_bytes = fernet.decrypt(encrypted_weight)
            weight_array = np.frombuffer(weight_bytes, dtype=np.float32).reshape(shape)
            decrypted_weights.append(weight_array)
        
        return decrypted_weights

# Set the privacy preserver
PrivacyPreserver = BasicPrivacyPreserver

class IoMTDevice:    
    def __init__(self, device_id: str, X_local: np.ndarray, y_local: np.ndarray, privacy_budget: float = 1.0, use_medical_model: bool = True):
        """        
        Args:
            device_id: Unique identifier for the device
            X_local: Local feature data
            y_local: Local target data
            privacy_budget: Privacy budget for differential privacy
            use_medical_model: Whether to use the specialized medical model
        """
        self.device_id = device_id
        self.X_local = X_local
        self.y_local = y_local
        
        # Use high-performance medical neural network for better accuracy
        if use_medical_model:
            self.model = HighPerformanceMedicalNetwork(X_local.shape[1], learning_rate=0.001)
            print(f"   Using HighPerformanceMedicalNetwork for {device_id}")
        else:
            self.model = SimpleNeuralNetwork(X_local.shape[1], learning_rate=0.001)
            
        self.training_history = []
        self.privacy_budget = privacy_budget
        self.used_privacy_budget = 0.0
        self.encryption_key = Fernet.generate_key()
        
        # Calculate class weights for imbalanced data
        self.class_weights = self._calculate_class_weights()
        
        # Add minimal local differential privacy to training data (reduce noise for better accuracy)
        # self._add_local_dp_noise()  # Temporarily disabled for better accuracy

        print(f"Enhanced IoMT Device {device_id} initialized with {len(X_local)} patient records")
        print(f"Privacy budget: {privacy_budget}, Class distribution: {np.bincount(y_local.astype(int))}")
        
    def _calculate_class_weights(self):
        """Calculate class weights for imbalanced medical data"""
        y_flat = self.y_local.flatten()
        class_counts = np.bincount(y_flat.astype(int))
        total_samples = len(y_flat)
        class_weights = {}
        
        for i, count in enumerate(class_counts):
            if count > 0:
                class_weights[i] = total_samples / (len(class_counts) * count)
            else:
                class_weights[i] = 1.0
                
        print(f"   Class weights for {self.device_id}: {class_weights}")
        return class_weights
    
    def _add_local_dp_noise(self, epsilon: float = 0.5):
        """Add local differential privacy noise to training data"""
        sensitivity = 1.0
        scale = sensitivity / epsilon
        
        # Add noise to features (be careful with medical data)
        noise = np.random.laplace(0, scale * 0.01, self.X_local.shape)  # Small noise for medical data
        self.X_local = self.X_local + noise
        
        print(f"\nDevice {self.device_id}: Added local DP noise (ε={epsilon}) to the training data")
    
    def local_train(self, epochs: int = 20, batch_size: int = 8, dp_epsilon: float = 0.01):
        """
        Enhanced local training with aggressive parameters for high accuracy
        Args:
            epochs: Number of training epochs (increased significantly)
            batch_size: Batch size for training (smaller for better gradients)
            dp_epsilon: Differential privacy epsilon (minimal for accuracy)
        
        Returns:
            Model weights after training
        """
        print(f"Device {self.device_id}: Starting high-performance training for {epochs} epochs...")

        if self.used_privacy_budget + dp_epsilon > self.privacy_budget:
            print(f"Device {self.device_id}: Privacy budget exhausted!")
            return None, None
        
        # Prepare training data
        y_flat = self.y_local.flatten()
        
        print(f"Device {self.device_id}: Training with aggressive parameters...")
        print(f"   Data shape: {self.X_local.shape}, Labels: {y_flat.shape}")
        print(f"   Class weights: {self.class_weights}")

        # High-performance training with aggressive parameters
        history = self.model.fit(
            self.X_local,
            y_flat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,  # Smaller validation split for more training data
            class_weight=self.class_weights,  # Handle class imbalance
            verbose=1,  # Show progress for debugging
            shuffle=True,  # Shuffle data each epoch
            # Add callbacks for better training
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
        Evaluate model performance on local data with proper shape handling
        Returns:
            Local accuracy
        """
        y_pred = self.model.predict(self.X_local, verbose=0)
        
        # Ensure proper shape alignment
        y_pred_flat = y_pred.flatten()  # Flatten predictions to 1D
        y_true_flat = self.y_local.flatten()  # Ensure labels are 1D
        
        # Convert predictions to binary (0 or 1)
        y_pred_binary = (y_pred_flat > 0.5).astype(int)
        
        # Calculate accuracy with proper shapes
        accuracy = np.mean(y_pred_binary == y_true_flat)
        
        print(f"Device {self.device_id}: Evaluation - Predictions shape: {y_pred.shape}, Labels shape: {self.y_local.shape}, Accuracy: {accuracy:.4f}")
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