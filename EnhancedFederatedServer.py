"""
Enhanced Federated Server for Medical IoMT Networks
Integrates advanced privacy, security, performance optimization, and monitoring capabilities
"""

import numpy as np
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json
from pathlib import Path

from EnhancedPrivacyPreserver import EnhancedPrivacyPreserver, RenyiDPAccountant
from AdvancedSecurityManager import AdvancedSecurityManager, ThreatLevel
from PerformanceOptimizer import PerformanceOptimizer, IntelligentDeviceSelector
from ComprehensiveMonitor import ComprehensiveMonitor
from EnhancedMedicalIoMTDevice import EnhancedMedicalIoMTDevice
from NeuralNetworkModal import MedicalNeuralNetwork

@dataclass 
class FederatedRoundConfig:
    round_number: int
    target_devices: int
    min_devices: int
    local_epochs: int
    privacy_budget_per_round: float
    security_threshold: float
    performance_target: float
    timeout_seconds: float

class EnhancedFederatedServer:
    """
    Advanced federated server with comprehensive privacy, security, and performance features
    """
    
    def __init__(self, input_size: int, output_size: int = 2, 
                 privacy_epsilon: float = 1.0, privacy_delta: float = 1e-5,
                 log_dir: str = "enhanced_fl_logs"):
        
        # Core model - Using medical-optimized neural network
        self.global_model = MedicalNeuralNetwork(input_size, learning_rate=0.001)
        
        # Enhanced components
        self.privacy_preserver = EnhancedPrivacyPreserver(privacy_epsilon, privacy_delta)
        self.security_manager = AdvancedSecurityManager(byzantine_tolerance=0.3)
        self.performance_optimizer = PerformanceOptimizer()
        self.monitor = ComprehensiveMonitor(log_dir=log_dir, plot_dir=f"{log_dir}/plots")
        
        # Rényi DP Accountant for global privacy tracking
        self.global_privacy_accountant = RenyiDPAccountant(privacy_epsilon, privacy_delta)
        
        # Device management
        self.registered_devices: Dict[str, EnhancedMedicalIoMTDevice] = {}
        self.device_selector = IntelligentDeviceSelector()
        
        # Training state
        self.current_round = 0
        self.round_history: List[Dict] = []
        self.convergence_history: List[float] = []
        
        # Configuration
        self.default_round_config = FederatedRoundConfig(
            round_number=0,
            target_devices=5,
            min_devices=3,
            local_epochs=3,
            privacy_budget_per_round=0.1,
            security_threshold=0.7,
            performance_target=0.8,
            timeout_seconds=300.0
        )
        
        self.logger = logging.getLogger('enhanced_federated_server')
        self.logger.info("Enhanced Federated Server initialized")
        
    def register_medical_device(self, device: EnhancedMedicalIoMTDevice):
        """Register a medical IoMT device with the federated server"""
        self.registered_devices[device.device_id] = device
        self.device_selector.register_device(device.device_profile)
        
        self.logger.info(f"Medical device registered: {device.device_id} ({device.device_type})")
        self.logger.info(f"Total registered devices: {len(self.registered_devices)}")
        
    def run_enhanced_federated_learning(self, num_rounds: int = 10,
                                      custom_config: Optional[FederatedRoundConfig] = None) -> Dict:
        """
        Execute enhanced federated learning with all advanced features
        """
        if custom_config is None:
            config = self.default_round_config
        else:
            config = custom_config
            
        self.logger.info(f"Starting enhanced federated learning for {num_rounds} rounds")
        self.logger.info(f"Configuration: {config}")
        
        # Pre-flight checks
        if len(self.registered_devices) < config.min_devices:
            raise ValueError(f"Insufficient devices: {len(self.registered_devices)} < {config.min_devices}")
            
        overall_start_time = datetime.now()
        
        try:
            for round_num in range(1, num_rounds + 1):
                config.round_number = round_num
                self.current_round = round_num
                
                # Execute single round with all enhancements
                round_results = self._execute_enhanced_round(config)
                
                # Check for convergence
                if self._check_convergence():
                    self.logger.info(f"Convergence achieved after {round_num} rounds")
                    break
                    
                # Adaptive configuration updates
                config = self._adapt_round_configuration(config, round_results)
                
        except Exception as e:
            self.logger.error(f"Federated learning failed: {e}")
            raise
            
        finally:
            # Generate final comprehensive report
            total_duration = (datetime.now() - overall_start_time).total_seconds()
            final_report = self._generate_final_report(total_duration)
            
            self.logger.info("Enhanced federated learning completed")
            self.logger.info(f"Total duration: {total_duration:.2f} seconds")
            
            return final_report
            
    def _execute_enhanced_round(self, config: FederatedRoundConfig) -> Dict:
        """Execute a single federated learning round with all enhancements"""
        round_start_time = datetime.now()
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Enhanced Federated Learning Round {config.round_number}")
        self.logger.info(f"{'='*60}")
        
        # Phase 1: Intelligent Device Selection
        selected_devices = self._intelligent_device_selection(config)
        if len(selected_devices) < config.min_devices:
            raise RuntimeError(f"Insufficient participating devices: {len(selected_devices)} < {config.min_devices}")
            
        self.monitor.logger.log_round_start(config.round_number, selected_devices)
        
        # Phase 2: Asynchronous Training Coordination
        device_updates, performance_metrics = self._coordinate_asynchronous_training(
            selected_devices, config
        )
        
        # Phase 3: Security Analysis & Byzantine Detection
        secure_updates, security_report = self.security_manager.evaluate_round_security(
            device_updates, config.round_number
        )
        
        # Phase 4: Privacy-Preserving Aggregation
        global_update, privacy_report = self._privacy_preserving_aggregation(
            secure_updates, config
        )
        
        # Phase 5: Global Model Update
        if global_update is not None:
            self.global_model.set_weights(global_update)
            
        # Phase 6: Model Distribution & Evaluation
        round_accuracy = self._distribute_and_evaluate(selected_devices, global_update)
        
        # Phase 7: Comprehensive Monitoring & Logging
        round_duration = (datetime.now() - round_start_time).total_seconds()
        
        round_results = {
            'round_number': config.round_number,
            'selected_devices': selected_devices,
            'participating_devices': list(device_updates.keys()),
            'round_accuracy': round_accuracy,
            'round_duration': round_duration,
            'performance_metrics': performance_metrics,
            'security_report': security_report,
            'privacy_report': privacy_report,
            'byzantine_devices_detected': security_report.get('byzantine_devices', []),
            'global_privacy_spent': self.global_privacy_accountant.get_privacy_spent()
        }
        
        self.round_history.append(round_results)
        self.convergence_history.append(round_accuracy)
        
        # Log comprehensive round completion
        self._log_round_completion(round_results)
        
        self.logger.info(f"Round {config.round_number} completed successfully")
        self.logger.info(f"Round accuracy: {round_accuracy:.4f}")
        self.logger.info(f"Participating devices: {len(device_updates)}/{len(selected_devices)}")
        
        return round_results
        
    def _intelligent_device_selection(self, config: FederatedRoundConfig) -> List[str]:
        """Perform intelligent device selection based on multiple criteria"""
        self.logger.info("Performing intelligent device selection...")
        
        # Update device profiles with current status
        available_devices = []
        for device_id, device in self.registered_devices.items():
            can_participate, reason = device.can_participate_in_round()
            if can_participate:
                available_devices.append(device_id)
                # Update device profile
                self.device_selector.update_device_profile(
                    device_id,
                    battery_level=device.device_profile.battery_level,
                    availability_score=device.device_profile.availability_score,
                    last_seen=datetime.now()
                )
            else:
                self.logger.warning(f"Device {device_id} cannot participate: {reason}")
                
        if len(available_devices) < config.min_devices:
            self.logger.error(f"Insufficient available devices: {len(available_devices)} < {config.min_devices}")
            return available_devices
            
        # Intelligent selection with medical device considerations
        minimum_requirements = {
            'compute_capability': 0.5,
            'memory_capacity': 1.0,
            'battery_level': 0.2,
            'availability_score': 0.6,
            'reliability_score': 0.7
        }
        
        selected = self.device_selector.select_devices(
            min(config.target_devices, len(available_devices)),
            minimum_requirements
        )
        
        # Filter to ensure we only select from available devices
        selected = [device_id for device_id in selected if device_id in available_devices]
        
        self.logger.info(f"Selected {len(selected)} devices from {len(available_devices)} available")
        return selected
        
    def _coordinate_asynchronous_training(self, selected_devices: List[str], 
                                        config: FederatedRoundConfig) -> Tuple[Dict, Dict]:
        """Coordinate asynchronous training across selected devices"""
        self.logger.info("Coordinating asynchronous training...")
        
        # Get current global weights
        global_weights = self.global_model.get_weights()
        
        # Use performance optimizer for asynchronous execution
        device_updates, performance_report = self.performance_optimizer.optimize_federated_round(
            global_weights, selected_devices, len(selected_devices), config.local_epochs
        )
        
        # Filter out failed updates and collect from actual devices
        actual_device_updates = {}
        for device_id in device_updates:
            if device_id in self.registered_devices:
                device = self.registered_devices[device_id]
                
                # Execute actual training
                encrypted_weights, weight_shapes = device.secure_local_training(
                    global_weights, config.local_epochs
                )
                
                if encrypted_weights is not None and weight_shapes is not None:
                    # Decrypt for aggregation (in practice would use secure multi-party computation)
                    decrypted_weights = device.privacy_manager.privacy_preserver.homomorphic_decrypt_weights(
                        encrypted_weights, device.encryption_key, weight_shapes
                    )
                    actual_device_updates[device_id] = decrypted_weights
                    
        self.logger.info(f"Collected updates from {len(actual_device_updates)} devices")
        return actual_device_updates, performance_report
        
    def _privacy_preserving_aggregation(self, device_updates: Dict[str, List[np.ndarray]], 
                                      config: FederatedRoundConfig) -> Tuple[Optional[List[np.ndarray]], Dict]:
        """Perform privacy-preserving secure aggregation"""
        self.logger.info("Performing privacy-preserving aggregation...")
        
        if not device_updates:
            return None, {'error': 'No device updates to aggregate'}
            
        # Check global privacy budget
        can_spend = self.global_privacy_accountant.can_spend_privacy(config.privacy_budget_per_round)
        if not can_spend:
            self.logger.warning("Global privacy budget exhausted")
            return None, {'error': 'Privacy budget exhausted'}
            
        # Simple federated averaging (would use more sophisticated aggregation in practice)
        num_devices = len(device_updates)
        device_weights = list(device_updates.values())
        
        # Initialize aggregated weights
        aggregated_weights = []
        for layer_idx in range(len(device_weights[0])):
            layer_sum = np.zeros_like(device_weights[0][layer_idx])
            for device_weights_list in device_weights:
                layer_sum += device_weights_list[layer_idx]
            aggregated_weights.append(layer_sum / num_devices)
            
        # Apply server-side differential privacy
        private_aggregated_weights = self.privacy_preserver.add_gaussian_noise_mechanism(
            aggregated_weights, 
            epsilon=config.privacy_budget_per_round,
            delta=self.privacy_preserver.rdp_accountant.target_delta,
            device_id="server"
        )
        
        # Update global privacy accountant
        self.global_privacy_accountant.add_privacy_cost(
            config.privacy_budget_per_round,
            self.privacy_preserver.rdp_accountant.target_delta,
            alpha=2.0,
            mechanism="server_aggregation",
            device_id="server"
        )
        
        privacy_report = {
            'privacy_budget_used': config.privacy_budget_per_round,
            'global_privacy_spent': self.global_privacy_accountant.get_privacy_spent(),
            'participating_devices': len(device_updates),
            'aggregation_method': 'secure_federated_averaging'
        }
        
        self.logger.info(f"Privacy-preserving aggregation completed with ε={config.privacy_budget_per_round}")
        return private_aggregated_weights, privacy_report
        
    def _distribute_and_evaluate(self, selected_devices: List[str], 
                               global_update: Optional[List[np.ndarray]]) -> float:
        """Distribute global model and evaluate performance"""
        if global_update is None:
            return 0.0
            
        individual_accuracies = {}
        
        for device_id in selected_devices:
            if device_id in self.registered_devices:
                device = self.registered_devices[device_id]
                
                # Update device model
                device.current_model_weights = [w.copy() for w in global_update]
                
                # Evaluate local accuracy
                accuracy = device.evaluate_local_model()
                individual_accuracies[device_id] = accuracy
                
        average_accuracy = float(np.mean(list(individual_accuracies.values()))) if individual_accuracies else 0.0
        
        self.logger.info(f"Model evaluation completed. Average accuracy: {average_accuracy:.4f}")
        return average_accuracy
        
    def _log_round_completion(self, round_results: Dict):
        """Log comprehensive round completion information"""
        # Extract individual accuracies from devices
        individual_accuracies = {}
        privacy_budget_consumed = {}
        
        for device_id in round_results['participating_devices']:
            if device_id in self.registered_devices:
                device = self.registered_devices[device_id]
                individual_accuracies[device_id] = device.evaluate_local_model()
                privacy_budget_consumed[device_id] = device.privacy_budget_used
                
        # Log to comprehensive monitor
        self.monitor.log_round_completion(
            round_number=round_results['round_number'],
            participating_devices=round_results['participating_devices'],
            individual_accuracies=individual_accuracies,
            privacy_budget_consumed=privacy_budget_consumed,
            byzantine_devices=round_results['byzantine_devices_detected'],
            communication_overhead=round_results['performance_metrics'].get('compression_ratio', 0.0),
            round_duration=round_results['round_duration'],
            security_events=round_results['security_report'].get('security_events', [])
        )
        
    def _check_convergence(self) -> bool:
        """Check if the federated learning has converged"""
        if len(self.convergence_history) < 5:
            return False
            
        # Check if accuracy has stabilized
        recent_accuracies = self.convergence_history[-5:]
        accuracy_std = np.std(recent_accuracies)
        
        # Convergence if standard deviation is low and accuracy is high
        return bool(accuracy_std < 0.01 and np.mean(recent_accuracies) > 0.85)
        
    def _adapt_round_configuration(self, config: FederatedRoundConfig, 
                                 round_results: Dict) -> FederatedRoundConfig:
        """Adapt configuration based on round performance"""
        new_config = FederatedRoundConfig(
            round_number=config.round_number + 1,
            target_devices=config.target_devices,
            min_devices=config.min_devices,
            local_epochs=config.local_epochs,
            privacy_budget_per_round=config.privacy_budget_per_round,
            security_threshold=config.security_threshold,
            performance_target=config.performance_target,
            timeout_seconds=config.timeout_seconds
        )
        
        # Adapt based on performance
        participation_ratio = len(round_results['participating_devices']) / len(round_results['selected_devices'])
        
        if participation_ratio < 0.7:
            # Low participation - relax requirements
            new_config.target_devices = max(3, new_config.target_devices - 1)
            new_config.timeout_seconds *= 1.2
            
        elif participation_ratio > 0.9 and round_results['round_accuracy'] > 0.8:
            # High participation and good accuracy - can be more aggressive
            new_config.target_devices = min(10, new_config.target_devices + 1)
            new_config.local_epochs = min(5, new_config.local_epochs + 1)
            
        # Adapt privacy budget based on remaining budget
        global_epsilon_spent, _ = self.global_privacy_accountant.get_privacy_spent()
        remaining_budget = self.global_privacy_accountant.target_epsilon - global_epsilon_spent
        
        if remaining_budget < 0.2:
            # Low remaining budget - be more conservative
            new_config.privacy_budget_per_round *= 0.8
            
        return new_config
        
    def _generate_final_report(self, total_duration: float) -> Dict:
        """Generate comprehensive final report"""
        # Generate comprehensive monitoring report
        monitoring_report = self.monitor.generate_comprehensive_report()
        
        # Privacy summary
        final_privacy_spent = self.global_privacy_accountant.get_privacy_spent()
        privacy_summary = {
            'total_epsilon_spent': final_privacy_spent[0],
            'total_delta_spent': final_privacy_spent[1],
            'privacy_budget_utilization': final_privacy_spent[0] / self.global_privacy_accountant.target_epsilon,
            'device_privacy_reports': {
                device_id: device.get_medical_privacy_report()
                for device_id, device in self.registered_devices.items()
            }
        }
        
        # Security summary
        security_summary = self.security_manager.get_security_summary()
        
        # Performance summary
        performance_summary = self.performance_optimizer._get_performance_summary()
        
        # Model performance
        final_accuracy = self.convergence_history[-1] if self.convergence_history else 0.0
        model_summary = {
            'final_accuracy': final_accuracy,
            'convergence_achieved': self._check_convergence(),
            'total_rounds': len(self.round_history),
            'accuracy_improvement': (
                final_accuracy - self.convergence_history[0] 
                if len(self.convergence_history) > 1 else 0.0
            )
        }
        
        final_report = {
            'experiment_summary': {
                'total_duration_seconds': total_duration,
                'total_rounds_completed': len(self.round_history),
                'participating_devices': len(self.registered_devices),
                'timestamp': datetime.now()
            },
            'model_performance': model_summary,
            'privacy_analysis': privacy_summary,
            'security_analysis': security_summary,
            'performance_analysis': performance_summary,
            'monitoring_report': monitoring_report,
            'device_compliance_status': {
                device_id: device.get_compliance_status()
                for device_id, device in self.registered_devices.items()
            },
            'recommendations': self._generate_recommendations()
        }
        
        # Save final report
        report_path = Path(self.monitor.logger.log_dir) / f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
            
        self.logger.info(f"Final report saved to: {report_path}")
        
        return final_report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improving federated learning"""
        recommendations = []
        
        # Privacy recommendations
        final_privacy_spent = self.global_privacy_accountant.get_privacy_spent()
        if final_privacy_spent[0] > 0.8 * self.global_privacy_accountant.target_epsilon:
            recommendations.append("Consider increasing privacy budget or reducing noise parameters")
            
        # Security recommendations
        if len(self.security_manager.byzantine_detector.suspicious_devices) > 0:
            recommendations.append("Implement additional device authentication mechanisms")
            
        # Performance recommendations
        avg_participation = np.mean([
            len(round_result['participating_devices']) / len(round_result['selected_devices'])
            for round_result in self.round_history
        ])
        
        if avg_participation < 0.7:
            recommendations.append("Consider adjusting device selection criteria or improving network reliability")
            
        # Convergence recommendations
        if not self._check_convergence():
            recommendations.append("Consider increasing local epochs or adjusting learning parameters")
            
        return recommendations
