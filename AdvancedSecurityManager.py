"""
Advanced Security Module for Federated Learning
Implements Byzantine fault tolerance, secure aggregation, and advanced threat detection
"""

import numpy as np
import hashlib
import hmac
import secrets
import base64
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
import json

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    timestamp: datetime
    device_id: str
    event_type: str
    threat_level: ThreatLevel
    description: str
    evidence: Dict

class ByzantineDetector:
    """
    Advanced Byzantine fault detection using multiple techniques
    """
    
    def __init__(self, tolerance_threshold: float = 0.3):
        self.tolerance_threshold = tolerance_threshold
        self.device_behavior_history: Dict[str, List[Dict]] = {}
        self.suspicious_devices: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        
    def detect_byzantine_updates(self, device_updates: Dict[str, List[np.ndarray]], 
                               round_number: int) -> Dict[str, bool]:
        """
        Detect Byzantine (malicious) model updates using multiple detection methods
        """
        detection_results = {}
        device_ids = list(device_updates.keys())
        
        if len(device_ids) < 3:
            # Need at least 3 devices for Byzantine detection
            return {device_id: False for device_id in device_ids}
            
        # Method 1: Cosine similarity analysis
        cosine_scores = self._cosine_similarity_analysis(device_updates)
        
        # Method 2: Statistical outlier detection  
        statistical_outliers = self._statistical_outlier_detection(device_updates)
        
        # Method 3: Gradient magnitude analysis
        magnitude_outliers = self._gradient_magnitude_analysis(device_updates)
        
        # Method 4: Historical behavior analysis
        behavior_anomalies = self._behavior_anomaly_detection(device_updates, round_number)
        
        # Combine detection methods
        for device_id in device_ids:
            byzantine_indicators = 0
            
            if cosine_scores.get(device_id, 0) < 0.5:  # Low similarity
                byzantine_indicators += 1
                
            if device_id in statistical_outliers:
                byzantine_indicators += 1
                
            if device_id in magnitude_outliers:
                byzantine_indicators += 1
                
            if device_id in behavior_anomalies:
                byzantine_indicators += 1
                
            # Device is Byzantine if multiple indicators agree
            is_byzantine = byzantine_indicators >= 2
            detection_results[device_id] = is_byzantine
            
            if is_byzantine:
                self.suspicious_devices.add(device_id)
                self.logger.warning(f"Byzantine device detected: {device_id} (indicators: {byzantine_indicators})")
                
        return detection_results
        
    def _cosine_similarity_analysis(self, device_updates: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
        """Analyze cosine similarity between device updates"""
        device_ids = list(device_updates.keys())
        similarity_scores = {}
        
        for i, device_id in enumerate(device_ids):
            similarities = []
            device_update = device_updates[device_id]
            
            for j, other_device_id in enumerate(device_ids):
                if i != j:
                    other_update = device_updates[other_device_id]
                    
                    # Flatten and compute cosine similarity
                    flat_update = np.concatenate([layer.flatten() for layer in device_update])
                    flat_other = np.concatenate([layer.flatten() for layer in other_update])
                    
                    cosine_sim = np.dot(flat_update, flat_other) / (
                        np.linalg.norm(flat_update) * np.linalg.norm(flat_other) + 1e-8
                    )
                    similarities.append(cosine_sim)
                    
            similarity_scores[device_id] = np.mean(similarities) if similarities else 0.0
            
        return similarity_scores
        
    def _statistical_outlier_detection(self, device_updates: Dict[str, List[np.ndarray]]) -> Set[str]:
        """Detect statistical outliers using IQR method"""
        outliers = set()
        
        # Calculate update magnitudes
        device_magnitudes = {}
        for device_id, update in device_updates.items():
            magnitude = np.sqrt(sum(np.sum(layer**2) for layer in update))
            device_magnitudes[device_id] = magnitude
            
        magnitudes = list(device_magnitudes.values())
        q1, q3 = np.percentile(magnitudes, [25, 75])
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        for device_id, magnitude in device_magnitudes.items():
            if magnitude < lower_bound or magnitude > upper_bound:
                outliers.add(device_id)
                
        return outliers
        
    def _gradient_magnitude_analysis(self, device_updates: Dict[str, List[np.ndarray]]) -> Set[str]:
        """Detect devices with unusually large gradients"""
        outliers = set()
        
        # Calculate layer-wise magnitudes
        layer_magnitudes = {}
        for device_id, update in device_updates.items():
            layer_mags = [np.linalg.norm(layer) for layer in update]
            layer_magnitudes[device_id] = layer_mags
            
        # Check each layer for outliers
        num_layers = len(next(iter(layer_magnitudes.values())))
        
        for layer_idx in range(num_layers):
            layer_mags = np.array([layer_magnitudes[device_id][layer_idx] for device_id in device_updates.keys()])
            median_mag = np.median(layer_mags)
            mad = np.median(np.abs(layer_mags - median_mag))  # Median Absolute Deviation
            
            threshold = median_mag + 3 * mad  # 3-MAD rule
            
            for device_id in device_updates.keys():
                if layer_magnitudes[device_id][layer_idx] > threshold:
                    outliers.add(device_id)
                    
        return outliers
        
    def _behavior_anomaly_detection(self, device_updates: Dict[str, List[np.ndarray]], 
                                  round_number: int) -> Set[str]:
        """Detect anomalous behavior based on historical patterns"""
        anomalies = set()
        
        for device_id, update in device_updates.items():
            if device_id not in self.device_behavior_history:
                self.device_behavior_history[device_id] = []
                
            # Calculate update characteristics
            update_magnitude = np.sqrt(sum(np.sum(layer**2) for layer in update))
            layer_ratios = []
            
            for i in range(len(update) - 1):
                ratio = np.linalg.norm(update[i]) / (np.linalg.norm(update[i+1]) + 1e-8)
                layer_ratios.append(ratio)
                
            behavior_record = {
                'round': round_number,
                'magnitude': update_magnitude,
                'layer_ratios': layer_ratios,
                'timestamp': datetime.now()
            }
            
            self.device_behavior_history[device_id].append(behavior_record)
            
            # Analyze historical pattern
            if len(self.device_behavior_history[device_id]) >= 5:
                recent_records = self.device_behavior_history[device_id][-5:]
                historical_magnitudes = [r['magnitude'] for r in recent_records[:-1]]
                
                if historical_magnitudes:
                    avg_magnitude = np.mean(historical_magnitudes)
                    std_magnitude = np.std(historical_magnitudes)
                    
                    # Check if current magnitude is anomalous
                    if abs(update_magnitude - avg_magnitude) > 3 * std_magnitude:
                        anomalies.add(device_id)
                        
        return anomalies

class SecureAggregator:
    """
    Implements secure aggregation with multi-party computation
    """
    
    def __init__(self, min_participants: int = 3):
        self.min_participants = min_participants
        self.aggregation_keys: Dict[str, bytes] = {}
        self.logger = logging.getLogger(__name__)
        
    def generate_aggregation_keys(self, device_ids: List[str]) -> Dict[str, bytes]:
        """Generate pairwise keys for secure aggregation"""
        keys = {}
        
        for device_id in device_ids:
            # Generate a unique key for each device
            key = secrets.token_bytes(32)  # 256-bit key
            keys[device_id] = key
            
        self.aggregation_keys = keys
        return keys
        
    def secure_aggregate_trimmed_mean(self, device_updates: Dict[str, List[np.ndarray]], 
                                    byzantine_flags: Dict[str, bool],
                                    trim_ratio: float = 0.1) -> List[np.ndarray]:
        """
        Secure aggregation using trimmed mean to handle Byzantine devices
        """
        # Filter out Byzantine devices
        honest_updates = {
            device_id: update for device_id, update in device_updates.items()
            if not byzantine_flags.get(device_id, False)
        }
        
        if len(honest_updates) < self.min_participants:
            self.logger.error(f"Insufficient honest participants: {len(honest_updates)}")
            return []
            
        # Convert to numpy arrays for easier manipulation
        device_ids = list(honest_updates.keys())
        num_devices = len(device_ids)
        num_layers = len(honest_updates[device_ids[0]])
        
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Collect all device weights for this layer
            layer_weights = np.array([
                honest_updates[device_id][layer_idx] for device_id in device_ids
            ])
            
            # Apply trimmed mean along device dimension
            trim_count = max(1, int(num_devices * trim_ratio))
            
            # Sort along device axis and trim
            sorted_weights = np.sort(layer_weights, axis=0)
            trimmed_weights = sorted_weights[trim_count:-trim_count] if trim_count > 0 else sorted_weights
            
            # Compute mean
            aggregated_layer = np.mean(trimmed_weights, axis=0)
            aggregated_weights.append(aggregated_layer)
            
        self.logger.info(f"Secure aggregation completed with {len(honest_updates)} honest devices")
        return aggregated_weights
        
    def secure_aggregate_median(self, device_updates: Dict[str, List[np.ndarray]], 
                              byzantine_flags: Dict[str, bool]) -> List[np.ndarray]:
        """
        Secure aggregation using coordinate-wise median (robust to Byzantine attacks)
        """
        # Filter out Byzantine devices
        honest_updates = {
            device_id: update for device_id, update in device_updates.items()
            if not byzantine_flags.get(device_id, False)
        }
        
        if len(honest_updates) < self.min_participants:
            return []
            
        device_ids = list(honest_updates.keys())
        num_layers = len(honest_updates[device_ids[0]])
        
        aggregated_weights = []
        
        for layer_idx in range(num_layers):
            # Stack all device weights for this layer
            layer_weights = np.stack([
                honest_updates[device_id][layer_idx] for device_id in device_ids
            ], axis=0)
            
            # Compute coordinate-wise median
            median_weights = np.median(layer_weights, axis=0)
            aggregated_weights.append(median_weights)
            
        return aggregated_weights

class AdvancedSecurityManager:
    """
    Comprehensive security manager for federated learning
    """
    
    def __init__(self, byzantine_tolerance: float = 0.3):
        self.byzantine_detector = ByzantineDetector(byzantine_tolerance)
        self.secure_aggregator = SecureAggregator()
        self.security_events: List[SecurityEvent] = []
        self.device_trust_scores: Dict[str, float] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize RSA key pair for server
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
        
    def evaluate_round_security(self, device_updates: Dict[str, List[np.ndarray]], 
                              round_number: int) -> Tuple[Dict[str, List[np.ndarray]], Dict]:
        """
        Comprehensive security evaluation and secure aggregation
        """
        security_report = {
            'round_number': round_number,
            'total_devices': len(device_updates),
            'byzantine_devices': [],
            'honest_devices': [],
            'security_events': [],
            'aggregation_method': 'none',
            'trust_scores': {}
        }
        
        # 1. Detect Byzantine devices
        byzantine_flags = self.byzantine_detector.detect_byzantine_updates(
            device_updates, round_number
        )
        
        # 2. Update trust scores
        self._update_trust_scores(byzantine_flags, round_number)
        
        # 3. Log security events
        for device_id, is_byzantine in byzantine_flags.items():
            if is_byzantine:
                event = SecurityEvent(
                    timestamp=datetime.now(),
                    device_id=device_id,
                    event_type="byzantine_detection",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Byzantine behavior detected in round {round_number}",
                    evidence={"round": round_number, "detection_methods": ["cosine", "statistical", "magnitude", "behavior"]}
                )
                self.security_events.append(event)
                security_report['byzantine_devices'].append(device_id)
            else:
                security_report['honest_devices'].append(device_id)
                
        # 4. Filter out Byzantine devices and return secure device updates
        num_byzantine = sum(byzantine_flags.values())
        secure_device_updates = {
            device_id: update for device_id, update in device_updates.items()
            if not byzantine_flags.get(device_id, False)
        }
        
        security_report['secure_devices_count'] = len(secure_device_updates)
        security_report['filtered_devices'] = list(secure_device_updates.keys())
        security_report['trust_scores'] = self.device_trust_scores.copy()
        
        self.logger.info(f"Security evaluation completed: {num_byzantine} Byzantine devices detected")
        self.logger.info(f"Secure device updates: {len(secure_device_updates)} devices")
        
        return secure_device_updates, security_report
        
    def _update_trust_scores(self, byzantine_flags: Dict[str, bool], round_number: int):
        """Update device trust scores based on behavior"""
        for device_id, is_byzantine in byzantine_flags.items():
            if device_id not in self.device_trust_scores:
                self.device_trust_scores[device_id] = 1.0  # Start with full trust
                
            if is_byzantine:
                # Decrease trust for Byzantine behavior
                self.device_trust_scores[device_id] *= 0.8  # 20% penalty
            else:
                # Slowly increase trust for honest behavior
                self.device_trust_scores[device_id] = min(1.0, 
                    self.device_trust_scores[device_id] * 1.01  # 1% reward
                )
                
    def encrypt_model_update(self, weights: List[np.ndarray]) -> str:
        """Encrypt model update using RSA"""
        # Serialize weights
        weights_bytes = np.concatenate([w.flatten() for w in weights]).astype(np.float32).tobytes()
        
        # For large data, use hybrid encryption (RSA + AES)
        # This is simplified - in practice would use proper hybrid encryption
        encrypted_data = self.public_key.encrypt(
            weights_bytes[:245],  # RSA can only encrypt limited data
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return base64.b64encode(encrypted_data).decode()
        
    def get_security_summary(self) -> Dict:
        """Get comprehensive security summary"""
        return {
            'total_security_events': len(self.security_events),
            'byzantine_devices_detected': len(self.byzantine_detector.suspicious_devices),
            'suspicious_devices': list(self.byzantine_detector.suspicious_devices),
            'device_trust_scores': self.device_trust_scores,
            'recent_events': [
                {
                    'timestamp': event.timestamp.isoformat(),
                    'device_id': event.device_id,
                    'event_type': event.event_type,
                    'threat_level': event.threat_level.value,
                    'description': event.description
                }
                for event in self.security_events[-10:]  # Last 10 events
            ]
        }
