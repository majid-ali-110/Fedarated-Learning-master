from typing import List, Tuple, Dict
import numpy as np
from cryptography.fernet import Fernet
import base64

class PrivacyPreserver:
    """
    Implements various privacy preservation techniques for federated learning
    """
    
    @staticmethod
    def add_differential_privacy_noise(weights: List[np.ndarray], epsilon: float = 0.1, 
                                     sensitivity: float = 1.0) -> List[np.ndarray]:
        """
        Add Laplacian noise for differential privacy
        """
        scale = sensitivity / epsilon
        noisy_weights = []
        
        for weight_matrix in weights:
            noise = np.random.laplace(0, scale, weight_matrix.shape)
            noisy_weights.append(weight_matrix + noise)
        
        return noisy_weights
    
    @staticmethod
    def gradient_clipping(weights: List[np.ndarray], clip_norm: float = 1.0) -> List[np.ndarray]:
        """
        Clip gradients to prevent gradient leakage attacks
        """
        clipped_weights = []
        
        for weight_matrix in weights:
            # Calculate L2 norm
            norm = np.linalg.norm(weight_matrix)
            if norm > clip_norm:
                # Clip the weights
                clipped_weights.append(weight_matrix * (clip_norm / norm))
            else:
                clipped_weights.append(weight_matrix)
        
        return clipped_weights
    
    @staticmethod
    def encrypt_weights(weights: List[np.ndarray], key: bytes) -> Tuple[str, bytes]:
        """
        Encrypt model weights using Fernet symmetric encryption
        """
        fernet = Fernet(key)
        
        # Serialize weights to bytes
        weights_bytes = np.concatenate([w.flatten() for w in weights]).tobytes()
        
        # Encrypt
        encrypted_weights = fernet.encrypt(weights_bytes)
        
        return base64.b64encode(encrypted_weights).decode(), weights_bytes
    
    @staticmethod
    def decrypt_weights(encrypted_weights: str, key: bytes, original_shapes: List[Tuple]) -> List[np.ndarray]:
        """
        Decrypt model weights
        """
        fernet = Fernet(key)
        
        # Decode and decrypt
        encrypted_bytes = base64.b64decode(encrypted_weights.encode())
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        
        # Reconstruct weights
        weights_flat = np.frombuffer(decrypted_bytes, dtype=np.float32)
        
        weights = []
        start_idx = 0
        for shape in original_shapes:
            size = np.prod(shape)
            weight_matrix = weights_flat[start_idx:start_idx + size].reshape(shape)
            weights.append(weight_matrix)
            start_idx += size
        
        return weights
