"""
Enhanced Privacy Preserver for Federated Learning
Implements state-of-the-art privacy techniques including Rényi DP, adaptive clipping, and comprehensive privacy accounting.
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging
from datetime import datetime

@dataclass
class PrivacyAccountingEntry:
    """Entry for privacy accounting ledger"""
    timestamp: datetime
    mechanism: str
    epsilon: float
    delta: float
    alpha: float  # Rényi DP order
    sensitivity: float
    noise_scale: float
    device_id: str

class RenyiDPAccountant:
    """
    Advanced Rényi Differential Privacy Accountant
    Provides tighter privacy bounds than basic DP composition
    """
    
    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.privacy_ledger: List[PrivacyAccountingEntry] = []
        self.alpha_orders = [1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 8., 16., 32., 64.]
        
    def add_privacy_cost(self, epsilon: float, delta: float, alpha: float, 
                        mechanism: str, device_id: str, sensitivity: float = 1.0):
        """Add privacy cost to the ledger"""
        entry = PrivacyAccountingEntry(
            timestamp=datetime.now(),
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            alpha=alpha,
            sensitivity=sensitivity,
            noise_scale=sensitivity / epsilon if epsilon > 0 else float('inf'),
            device_id=device_id
        )
        self.privacy_ledger.append(entry)
        
    def compute_rdp(self, sigma: float, alpha: float, steps: int = 1) -> float:
        """Compute Rényi Differential Privacy guarantee"""
        if alpha == 1:
            return steps * (1.0 / (2 * sigma**2))
        else:
            return steps * alpha / (2 * sigma**2)
            
    def convert_rdp_to_dp(self, rdp_values: Dict[float, float]) -> Tuple[float, float]:
        """Convert RDP to (ε, δ)-DP using optimal conversion"""
        min_epsilon = float('inf')
        
        for alpha, rdp in rdp_values.items():
            if rdp == 0:
                continue
            epsilon = rdp + math.log(1/self.target_delta) / (alpha - 1)
            min_epsilon = min(min_epsilon, epsilon)
            
        return min_epsilon, self.target_delta
        
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy spent across all mechanisms"""
        rdp_values = {}
        
        for alpha in self.alpha_orders:
            total_rdp = 0
            for entry in self.privacy_ledger:
                if entry.epsilon > 0:
                    sigma = entry.sensitivity / entry.epsilon
                    total_rdp += self.compute_rdp(sigma, alpha)
            rdp_values[alpha] = total_rdp
            
        epsilon, delta = self.convert_rdp_to_dp(rdp_values)
        return epsilon, delta
        
    def can_spend_privacy(self, additional_epsilon: float) -> bool:
        """Check if we can spend additional privacy budget"""
        current_epsilon, _ = self.get_privacy_spent()
        return (current_epsilon + additional_epsilon) <= self.target_epsilon

class AdaptiveGradientClipper:
    """
    Adaptive gradient clipping that adjusts clip norm based on gradient distributions
    """
    
    def __init__(self, initial_clip_norm: float = 1.0, adaptation_rate: float = 0.1):
        self.clip_norm = initial_clip_norm
        self.adaptation_rate = adaptation_rate
        self.gradient_norms_history = []
        
    def adaptive_clip(self, gradients: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """Perform adaptive gradient clipping"""
        # Calculate per-sample gradient norms
        per_sample_norms = []
        flattened_grads = []
        
        for grad in gradients:
            flat_grad = grad.flatten()
            flattened_grads.extend(flat_grad)
            per_sample_norms.append(np.linalg.norm(flat_grad))
            
        # Calculate global norm
        global_norm = np.linalg.norm(flattened_grads)
        self.gradient_norms_history.append(global_norm)
        
        # Adaptive clip norm adjustment
        if len(self.gradient_norms_history) > 10:
            recent_norms = self.gradient_norms_history[-10:]
            percentile_90 = float(np.percentile(recent_norms, 90))
            
            # Adapt clip norm towards 90th percentile
            target_clip_norm = min(percentile_90, 2.0)  # Cap at 2.0
            self.clip_norm += self.adaptation_rate * (target_clip_norm - self.clip_norm)
            self.clip_norm = max(0.1, self.clip_norm)  # Minimum clip norm
        
        # Apply clipping
        clipped_gradients = []
        for grad in gradients:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.clip_norm:
                clipped_gradients.append(grad * (self.clip_norm / grad_norm))
            else:
                clipped_gradients.append(grad.copy())
                
        return clipped_gradients, self.clip_norm

class EnhancedPrivacyPreserver:
    """
    Enhanced Privacy Preserver with advanced differential privacy techniques
    """
    
    def __init__(self, target_epsilon: float = 1.0, target_delta: float = 1e-5):
        self.rdp_accountant = RenyiDPAccountant(target_epsilon, target_delta)
        self.adaptive_clipper = AdaptiveGradientClipper()
        self.logger = logging.getLogger(__name__)
        
    def add_gaussian_noise_mechanism(self, weights: List[np.ndarray], 
                                   epsilon: float, delta: float,
                                   sensitivity: float = 1.0,
                                   device_id: str = "unknown") -> List[np.ndarray]:
        """
        Add Gaussian noise using the Gaussian Mechanism for (ε, δ)-DP
        """
        if epsilon == 0:
            self.logger.warning("Epsilon is 0, returning original weights")
            return weights
            
        # Calculate noise scale for Gaussian mechanism
        sigma = math.sqrt(2 * math.log(1.25/delta)) * sensitivity / epsilon
        
        # Add to privacy ledger
        self.rdp_accountant.add_privacy_cost(
            epsilon, delta, alpha=2.0, 
            mechanism="gaussian_noise", 
            device_id=device_id,
            sensitivity=sensitivity
        )
        
        noisy_weights = []
        for weight_matrix in weights:
            noise = np.random.normal(0, sigma, weight_matrix.shape)
            noisy_weights.append(weight_matrix + noise)
            
        return noisy_weights
        
    def add_laplace_noise_mechanism(self, weights: List[np.ndarray],
                                  epsilon: float, sensitivity: float = 1.0,
                                  device_id: str = "unknown") -> List[np.ndarray]:
        """
        Add Laplace noise using the Laplace Mechanism for ε-DP
        """
        if epsilon == 0:
            return weights
            
        scale = sensitivity / epsilon
        
        # Add to privacy ledger  
        self.rdp_accountant.add_privacy_cost(
            epsilon, 0.0, alpha=float('inf'),
            mechanism="laplace_noise",
            device_id=device_id,
            sensitivity=sensitivity
        )
        
        noisy_weights = []
        for weight_matrix in weights:
            noise = np.random.laplace(0, scale, weight_matrix.shape)
            noisy_weights.append(weight_matrix + noise)
            
        return noisy_weights
        
    def adaptive_gradient_clipping(self, weights: List[np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """
        Apply adaptive gradient clipping
        """
        return self.adaptive_clipper.adaptive_clip(weights)
        
    def local_differential_privacy(self, data: np.ndarray, epsilon: float,
                                 device_id: str = "unknown") -> np.ndarray:
        """
        Apply local differential privacy to training data
        Enhanced with better noise calibration for medical data
        """
        if epsilon == 0:
            return data
            
        # For medical data, use smaller sensitivity and bounded noise
        sensitivity = min(1.0, float(np.std(data)) * 0.1)  # Adaptive sensitivity
        scale = sensitivity / epsilon
        
        # Bounded Laplace noise to prevent unrealistic medical values
        noise = np.random.laplace(0, scale, data.shape)
        noise = np.clip(noise, -3*scale, 3*scale)  # Bound noise to ±3σ
        
        # Add to privacy ledger
        self.rdp_accountant.add_privacy_cost(
            epsilon, 0.0, alpha=float('inf'),
            mechanism="local_dp",
            device_id=device_id,
            sensitivity=sensitivity
        )
        
        return data + noise
        
    def homomorphic_encryption_weights(self, weights: List[np.ndarray], 
                                     public_key: bytes) -> str:
        """
        Simplified homomorphic encryption simulation
        In practice, would use libraries like SEAL or HElib
        """
        # This is a simplified simulation - real HE would use specialized libraries
        fernet = Fernet(public_key)
        
        # Serialize and encrypt weights
        weights_bytes = np.concatenate([w.flatten() for w in weights]).astype(np.float32).tobytes()
        encrypted_weights = fernet.encrypt(weights_bytes)
        
        return base64.b64encode(encrypted_weights).decode()
        
    def homomorphic_decrypt_weights(self, encrypted_weights: str, 
                                  private_key: bytes, 
                                  shapes: List[Tuple]) -> List[np.ndarray]:
        """
        Decrypt homomorphically encrypted weights
        """
        fernet = Fernet(private_key)
        
        # Decode and decrypt
        encrypted_bytes = base64.b64decode(encrypted_weights.encode())
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        
        # Reconstruct weights
        weights_flat = np.frombuffer(decrypted_bytes, dtype=np.float32)
        
        weights = []
        start_idx = 0
        for shape in shapes:
            size = np.prod(shape)
            weight_matrix = weights_flat[start_idx:start_idx + size].reshape(shape)
            weights.append(weight_matrix)
            start_idx += size
            
        return weights
        
    def get_privacy_report(self) -> Dict:
        """Generate comprehensive privacy report"""
        epsilon_spent, delta_spent = self.rdp_accountant.get_privacy_spent()
        
        return {
            'total_epsilon_spent': epsilon_spent,
            'total_delta_spent': delta_spent,
            'target_epsilon': self.rdp_accountant.target_epsilon,
            'target_delta': self.rdp_accountant.target_delta,
            'privacy_remaining': max(0, self.rdp_accountant.target_epsilon - epsilon_spent),
            'number_of_mechanisms_used': len(self.rdp_accountant.privacy_ledger),
            'mechanisms_breakdown': self._get_mechanisms_breakdown(),
            'current_clip_norm': self.adaptive_clipper.clip_norm,
            'privacy_exhausted': epsilon_spent >= self.rdp_accountant.target_epsilon
        }
        
    def _get_mechanisms_breakdown(self) -> Dict:
        """Get breakdown of privacy mechanisms used"""
        breakdown = {}
        for entry in self.rdp_accountant.privacy_ledger:
            if entry.mechanism not in breakdown:
                breakdown[entry.mechanism] = {
                    'count': 0,
                    'total_epsilon': 0.0,
                    'devices': set()
                }
            breakdown[entry.mechanism]['count'] += 1
            breakdown[entry.mechanism]['total_epsilon'] += entry.epsilon
            breakdown[entry.mechanism]['devices'].add(entry.device_id)
            
        # Convert sets to lists for JSON serialization
        for mechanism in breakdown:
            breakdown[mechanism]['devices'] = list(breakdown[mechanism]['devices'])
            
        return breakdown
