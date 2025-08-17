"""
Enhanced Medical IoMT Device for Federated Learning
Specialized for healthcare applications with advanced privacy, security, and medical data handling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
from cryptography.fernet import Fernet

from EnhancedPrivacyPreserver import EnhancedPrivacyPreserver
from PerformanceOptimizer import DeviceProfile, DeviceStatus

class MedicalDataType(Enum):
    VITAL_SIGNS = "vital_signs"
    DIAGNOSTIC_IMAGES = "diagnostic_images"
    LAB_RESULTS = "lab_results"
    CLINICAL_NOTES = "clinical_notes"
    SENSOR_DATA = "sensor_data"
    GENOMIC_DATA = "genomic_data"

class ComplianceFramework(Enum):
    HIPAA = "hipaa"
    GDPR = "gdpr"
    FDA = "fda"
    HL7 = "hl7"
    HITECH = "hitech"

@dataclass
class PatientConsent:
    patient_id: str
    consent_given: bool
    consent_timestamp: datetime
    data_types_consented: List[MedicalDataType]
    expiration_date: Optional[datetime]
    withdrawal_allowed: bool

@dataclass
class MedicalDataRecord:
    record_id: str
    patient_id: str
    data_type: MedicalDataType
    timestamp: datetime
    data: np.ndarray
    metadata: Dict[str, Any]
    sensitivity_level: int  # 1-5, 5 being most sensitive
    consent_verified: bool

class HealthcarePrivacyManager:
    """
    Specialized privacy manager for healthcare data
    """
    
    def __init__(self, compliance_frameworks: List[ComplianceFramework]):
        self.compliance_frameworks = compliance_frameworks
        self.privacy_preserver = EnhancedPrivacyPreserver(target_epsilon=5.0, target_delta=1e-6)
        self.patient_consents: Dict[str, PatientConsent] = {}
        self.data_minimization_rules = self._initialize_minimization_rules()
        
    def _initialize_minimization_rules(self) -> Dict[MedicalDataType, Dict]:
        """Initialize data minimization rules for different medical data types"""
        return {
            MedicalDataType.VITAL_SIGNS: {
                'max_precision': 2,
                'temporal_aggregation': '5min',
                'feature_selection': ['heart_rate', 'blood_pressure', 'temperature']
            },
            MedicalDataType.LAB_RESULTS: {
                'max_precision': 1,
                'outlier_capping': True,
                'reference_range_normalization': True
            },
            MedicalDataType.SENSOR_DATA: {
                'sampling_rate_reduction': 0.5,
                'noise_filtering': True,
                'dimensionality_reduction': 0.8
            },
            MedicalDataType.GENOMIC_DATA: {
                'snp_filtering': True,
                'population_frequency_threshold': 0.01,
                'linkage_disequilibrium_pruning': True
            }
        }
        
    def verify_patient_consent(self, patient_id: str, data_type: MedicalDataType) -> bool:
        """Verify patient consent for data usage"""
        if patient_id not in self.patient_consents:
            return False
            
        consent = self.patient_consents[patient_id]
        
        # Check if consent is still valid
        if consent.expiration_date and datetime.now() > consent.expiration_date:
            return False
            
        return (consent.consent_given and 
                data_type in consent.data_types_consented)
                
    def apply_medical_data_minimization(self, data: np.ndarray, 
                                      data_type: MedicalDataType) -> np.ndarray:
        """Apply data minimization specific to medical data type"""
        rules = self.data_minimization_rules.get(data_type, {})
        minimized_data = data.copy()
        
        # Apply precision reduction
        if 'max_precision' in rules:
            precision = rules['max_precision']
            minimized_data = np.round(minimized_data, precision)
            
        # Apply outlier capping for lab results
        if rules.get('outlier_capping', False):
            q1, q3 = np.percentile(minimized_data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            minimized_data = np.clip(minimized_data, lower_bound, upper_bound)
            
        # Apply dimensionality reduction for sensor data
        if 'dimensionality_reduction' in rules:
            reduction_factor = rules['dimensionality_reduction']
            if len(minimized_data.shape) > 1:
                n_features = int(minimized_data.shape[1] * reduction_factor)
                # Simple feature selection based on variance
                variances = np.var(minimized_data, axis=0)
                top_features = np.argsort(variances)[-n_features:]
                minimized_data = minimized_data[:, top_features]
                
        return minimized_data
        
    def apply_medical_differential_privacy(self, data: np.ndarray, 
                                         data_type: MedicalDataType,
                                         sensitivity_level: int,
                                         device_id: str) -> np.ndarray:
        """Apply differential privacy tailored for medical data"""
        # Adjust epsilon based on sensitivity level (higher sensitivity = lower epsilon)
        base_epsilon = 0.1
        epsilon = base_epsilon / sensitivity_level
        
        # Use Gaussian mechanism for continuous medical data
        if data_type in [MedicalDataType.VITAL_SIGNS, MedicalDataType.LAB_RESULTS, 
                        MedicalDataType.SENSOR_DATA]:
            return self.privacy_preserver.add_gaussian_noise_mechanism(
                [data], epsilon=epsilon, delta=1e-6, device_id=device_id
            )[0]
        else:
            # Use Laplace mechanism for discrete data
            return self.privacy_preserver.add_laplace_noise_mechanism(
                [data], epsilon=epsilon, device_id=device_id
            )[0]

class MedicalDeviceSimulator:
    """
    Simulates various medical device behaviors and data patterns
    """
    
    def __init__(self, device_type: str):
        self.device_type = device_type
        self.simulation_parameters = self._get_device_parameters(device_type)
        
    def _get_device_parameters(self, device_type: str) -> Dict:
        """Get simulation parameters for different medical devices"""
        parameters = {
            'wearable_monitor': {
                'sampling_rate': 1,  # Hz
                'battery_drain_rate': 0.02,  # per hour
                'connectivity_reliability': 0.95,
                'data_types': [MedicalDataType.VITAL_SIGNS, MedicalDataType.SENSOR_DATA]
            },
            'hospital_monitor': {
                'sampling_rate': 5,  # Hz
                'battery_drain_rate': 0.0,  # Always plugged in
                'connectivity_reliability': 0.99,
                'data_types': [MedicalDataType.VITAL_SIGNS, MedicalDataType.LAB_RESULTS]
            },
            'diagnostic_scanner': {
                'sampling_rate': 0.1,  # Hz (intermittent)
                'battery_drain_rate': 0.1,  # High power consumption
                'connectivity_reliability': 0.90,
                'data_types': [MedicalDataType.DIAGNOSTIC_IMAGES]
            },
            'lab_analyzer': {
                'sampling_rate': 0.05,  # Hz (batch processing)
                'battery_drain_rate': 0.05,
                'connectivity_reliability': 0.98,
                'data_types': [MedicalDataType.LAB_RESULTS]
            }
        }
        return parameters.get(device_type, parameters['wearable_monitor'])
        
    def simulate_device_behavior(self, duration_hours: float) -> Dict:
        """Simulate device behavior over time"""
        behavior = {
            'uptime_ratio': min(1.0, self.simulation_parameters['connectivity_reliability'] * 
                               np.random.uniform(0.9, 1.1)),
            'battery_level': max(0.0, 1.0 - 
                               (self.simulation_parameters['battery_drain_rate'] * duration_hours)),
            'data_quality_score': np.random.uniform(0.8, 1.0),
            'network_latency': np.random.exponential(50),  # milliseconds
            'failure_events': np.random.poisson(duration_hours * 0.01)  # failures per hour
        }
        
        return behavior

class EnhancedMedicalIoMTDevice:
    """
    Enhanced IoMT device with comprehensive medical data handling capabilities
    """
    
    def __init__(self, device_id: str, device_type: str, 
                 medical_data: List[MedicalDataRecord],
                 compliance_frameworks: Optional[List[ComplianceFramework]] = None):
        
        self.device_id = device_id
        self.device_type = device_type
        self.medical_data = medical_data
        
        if compliance_frameworks is None:
            compliance_frameworks = [ComplianceFramework.HIPAA, ComplianceFramework.GDPR]
        
        # Initialize components
        self.privacy_manager = HealthcarePrivacyManager(compliance_frameworks)
        self.device_simulator = MedicalDeviceSimulator(device_type)
        self.logger = logging.getLogger(f'medical_device_{device_id}')
        
        # Device profile for performance optimization
        self.device_profile = self._initialize_device_profile()
        
        # Prepare training data
        self.X_local, self.y_local = self._prepare_training_data()
        
        # Training state
        self.current_model_weights: Optional[List[np.ndarray]] = None
        self.training_history: List[Dict] = []
        self.privacy_budget_used = 0.0
        self.max_privacy_budget = 5.0  # Match the privacy preserver target_epsilon
        
        # Encryption
        self.encryption_key = Fernet.generate_key()
        
        self.logger.info(f"Enhanced Medical IoMT Device {device_id} initialized")
        self.logger.info(f"Device type: {device_type}")
        self.logger.info(f"Medical records: {len(medical_data)}")
        self.logger.info(f"Compliance frameworks: {[f.value for f in compliance_frameworks]}")
        
    def _initialize_device_profile(self) -> DeviceProfile:
        """Initialize device profile based on device type"""
        # Base capabilities by device type
        type_profiles = {
            'wearable_monitor': {
                'compute_capability': 1.0,
                'memory_capacity': 2.0,  # GB
                'network_bandwidth': 10.0  # Mbps
            },
            'hospital_monitor': {
                'compute_capability': 3.0,
                'memory_capacity': 8.0,
                'network_bandwidth': 100.0
            },
            'diagnostic_scanner': {
                'compute_capability': 5.0,
                'memory_capacity': 16.0,
                'network_bandwidth': 1000.0
            },
            'lab_analyzer': {
                'compute_capability': 2.0,
                'memory_capacity': 4.0,
                'network_bandwidth': 50.0
            }
        }
        
        profile_data = type_profiles.get(self.device_type, type_profiles['wearable_monitor'])
        
        return DeviceProfile(
            device_id=self.device_id,
            compute_capability=profile_data['compute_capability'],
            memory_capacity=profile_data['memory_capacity'],
            network_bandwidth=profile_data['network_bandwidth'],
            battery_level=1.0,
            availability_score=0.9,
            reliability_score=0.95,
            last_seen=datetime.now(),
            status=DeviceStatus.IDLE,
            avg_training_time=0.0,
            communication_cost=1.0
        )
        
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from medical records"""
        # Filter records based on consent (only warn for missing consents during initialization)
        consented_records = []
        for record in self.medical_data:
            if self.privacy_manager.verify_patient_consent(record.patient_id, record.data_type):
                consented_records.append(record)
            # Note: Consent may be added after device creation, so we don't warn here initially
                
        if not consented_records:
            self.logger.info("No consented data available for training - consents may be added later")
            return np.array([]).reshape(0, 1), np.array([])
            
        # Apply data minimization
        processed_data = []
        labels = []
        
        for record in consented_records:
            # Apply medical data minimization
            minimized_data = self.privacy_manager.apply_medical_data_minimization(
                record.data, record.data_type
            )
            
            # For federated learning, we save privacy budget for gradient updates
            # Instead of adding noise to each record here, we'll add noise during training
            processed_data.append(minimized_data.flatten())
            # Simulate binary labels (e.g., normal/abnormal)
            labels.append(np.random.randint(0, 2))
            
        # Ensure consistent dimensionality
        if processed_data:
            max_length = max(len(data) for data in processed_data)
            padded_data = []
            for data in processed_data:
                if len(data) < max_length:
                    padded = np.pad(data, (0, max_length - len(data)), mode='constant')
                    padded_data.append(padded)
                else:
                    padded_data.append(data[:max_length])
                    
            X = np.array(padded_data)
            y = np.array(labels)
        else:
            X = np.array([])
            y = np.array([])
            
        self.logger.info(f"Training data prepared: {X.shape[0] if X.size > 0 else 0} samples")
        return X, y
        
    def refresh_training_data(self):
        """Refresh training data after consents are added"""
        self.X_local, self.y_local = self._prepare_training_data()
        self.logger.info(f"Training data refreshed: {self.X_local.shape[0] if len(self.X_local) > 0 else 0} samples")
        
    def can_participate_in_round(self) -> Tuple[bool, str]:
        """Check if device can participate in federated learning round"""
        # Check privacy budget
        if self.privacy_budget_used >= self.max_privacy_budget:
            return False, "Privacy budget exhausted"
            
        # Check data availability
        if len(self.X_local) == 0:
            return False, "No training data available"
            
        # Simulate device behavior
        behavior = self.device_simulator.simulate_device_behavior(1.0)  # 1 hour simulation
        
        # Check battery level
        if behavior['battery_level'] < 0.1:
            return False, "Low battery level"
            
        # Check connectivity
        if behavior['uptime_ratio'] < 0.5:
            return False, "Poor connectivity"
            
        # Update device profile
        self.device_profile.battery_level = behavior['battery_level']
        self.device_profile.availability_score = behavior['uptime_ratio']
        
        return True, "Ready for participation"
        
    def secure_local_training(self, global_weights: List[np.ndarray], 
                            local_epochs: int = 3) -> Tuple[Optional[str], Optional[List[Tuple]]]:
        """
        Perform secure local training with medical data safeguards
        """
        # Check participation eligibility
        can_participate, reason = self.can_participate_in_round()
        if not can_participate:
            self.logger.warning(f"Cannot participate in training: {reason}")
            return None, None
            
        self.device_profile.status = DeviceStatus.TRAINING
        start_time = datetime.now()
        
        self.logger.info(f"Starting secure local training for {local_epochs} epochs")
        
        try:
            # Initialize local model with global weights
            self.current_model_weights = [w.copy() for w in global_weights]
            
            # Simulate local training
            final_loss = 0.0  # Initialize final_loss
            for epoch in range(local_epochs):
                # Simulate training step with privacy-preserving updates
                for i, weight_matrix in enumerate(self.current_model_weights):
                    # Add small gradient updates with medical data constraints
                    gradient_update = np.random.normal(0, 0.01, weight_matrix.shape)
                    
                    # Apply gradient clipping for medical data stability
                    clipped_gradient, _ = self.privacy_manager.privacy_preserver.adaptive_gradient_clipping([gradient_update])
                    
                    # Apply update
                    self.current_model_weights[i] += clipped_gradient[0] * 0.01  # learning rate
                    
                # Simulate loss calculation
                final_loss = np.random.exponential(1.0)  # Simulated loss
                self.logger.info(f"Epoch {epoch + 1}/{local_epochs}, Loss: {final_loss:.4f}")
                
            # Apply final privacy mechanisms to model weights
            epsilon = 0.1  # Conservative epsilon for medical data
            private_weights = self.privacy_manager.privacy_preserver.add_gaussian_noise_mechanism(
                self.current_model_weights, epsilon=epsilon, delta=1e-6, device_id=self.device_id
            )
            
            # Update privacy budget
            self.privacy_budget_used += epsilon
            
            # Encrypt weights for transmission
            weight_shapes = [w.shape for w in private_weights]
            encrypted_weights = self._encrypt_weights(private_weights)
            
            # Record training metrics
            training_duration = (datetime.now() - start_time).total_seconds()
            self.training_history.append({
                'timestamp': datetime.now(),
                'epochs': local_epochs,
                'duration': training_duration,
                'privacy_cost': epsilon,
                'final_loss': final_loss
            })
            
            self.device_profile.status = DeviceStatus.IDLE
            self.device_profile.avg_training_time = training_duration
            
            remaining_budget = self.max_privacy_budget - self.privacy_budget_used
            self.logger.info(f"Training completed. Remaining privacy budget: {remaining_budget:.2f}")
            
            return encrypted_weights, weight_shapes
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.device_profile.status = DeviceStatus.IDLE
            return None, None
            
    def _encrypt_weights(self, weights: List[np.ndarray]) -> str:
        """Encrypt model weights for secure transmission"""
        return self.privacy_manager.privacy_preserver.homomorphic_encryption_weights(
            weights, self.encryption_key
        )
        
    def decrypt_and_update_model(self, encrypted_weights: str, weight_shapes: List[Tuple]):
        """Decrypt and update local model with global weights"""
        try:
            decrypted_weights = self.privacy_manager.privacy_preserver.homomorphic_decrypt_weights(
                encrypted_weights, self.encryption_key, weight_shapes
            )
            self.current_model_weights = decrypted_weights
            self.logger.info("Model updated with global parameters")
        except Exception as e:
            self.logger.error(f"Failed to decrypt and update model: {e}")
            
    def evaluate_local_model(self) -> float:
        """Evaluate local model performance"""
        if self.current_model_weights is None or len(self.X_local) == 0:
            return 0.0
            
        # Simulate model evaluation
        accuracy = np.random.uniform(0.7, 0.95)  # Simulated accuracy
        return accuracy
        
    def get_medical_privacy_report(self) -> Dict:
        """Get comprehensive medical privacy report"""
        base_report = self.privacy_manager.privacy_preserver.get_privacy_report()
        
        # Add medical-specific information
        medical_report = {
            'device_id': self.device_id,
            'device_type': self.device_type,
            'compliance_frameworks': [f.value for f in self.privacy_manager.compliance_frameworks],
            'medical_data_types': list(set(record.data_type.value for record in self.medical_data)),
            'total_patients': len(set(record.patient_id for record in self.medical_data)),
            'consented_patients': len([
                record for record in self.medical_data
                if self.privacy_manager.verify_patient_consent(record.patient_id, record.data_type)
            ]),
            'privacy_budget_utilization': self.privacy_budget_used / self.max_privacy_budget,
            'training_rounds_participated': len(self.training_history),
            'device_performance': {
                'battery_level': self.device_profile.battery_level,
                'reliability_score': self.device_profile.reliability_score,
                'avg_training_time': self.device_profile.avg_training_time
            }
        }
        
        # Merge with base privacy report
        medical_report.update(base_report)
        
        return medical_report
        
    def get_compliance_status(self) -> Dict:
        """Get HIPAA/GDPR compliance status"""
        return {
            'hipaa_compliant': ComplianceFramework.HIPAA in self.privacy_manager.compliance_frameworks,
            'gdpr_compliant': ComplianceFramework.GDPR in self.privacy_manager.compliance_frameworks,
            'data_minimization_applied': True,
            'patient_consent_verified': all(
                self.privacy_manager.verify_patient_consent(record.patient_id, record.data_type)
                for record in self.medical_data[:5]  # Check sample
            ),
            'privacy_budget_within_limits': self.privacy_budget_used <= self.max_privacy_budget,
            'encryption_enabled': True,
            'audit_trail_maintained': len(self.training_history) > 0
        }
