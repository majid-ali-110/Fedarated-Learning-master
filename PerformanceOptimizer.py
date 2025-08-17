"""
Performance Enhancement Module for Federated Learning
Implements asynchronous processing, intelligent device selection, and communication optimization
"""

import asyncio
import numpy as np
import time
import threading
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, PriorityQueue
import logging
from datetime import datetime, timedelta
import json
import pickle
import zlib
from enum import Enum

class DeviceStatus(Enum):
    IDLE = "idle"
    TRAINING = "training"
    UPLOADING = "uploading"
    OFFLINE = "offline"
    BUSY = "busy"

@dataclass
class DeviceProfile:
    device_id: str
    compute_capability: float  # FLOPS rating
    memory_capacity: float     # GB
    network_bandwidth: float   # Mbps
    battery_level: float       # 0-1
    availability_score: float  # 0-1
    reliability_score: float   # 0-1
    last_seen: datetime
    status: DeviceStatus
    avg_training_time: float
    communication_cost: float

@dataclass
class TrainingTask:
    task_id: str
    device_id: str
    priority: int
    model_weights: List[np.ndarray]
    local_epochs: int
    deadline: datetime
    estimated_duration: float

class AdaptiveCompression:
    """
    Adaptive model compression for efficient communication
    """
    
    def __init__(self):
        self.compression_history: Dict[str, List[float]] = {}
        
    def compress_weights(self, weights: List[np.ndarray], 
                        compression_ratio: float = 0.1) -> Tuple[bytes, Dict]:
        """
        Compress model weights using multiple techniques
        """
        compression_info = {
            'original_size': 0,
            'compressed_size': 0,
            'techniques_used': []
        }
        
        # 1. Quantization
        quantized_weights, quantization_params = self._quantize_weights(weights)
        compression_info['techniques_used'].append('quantization')
        
        # 2. Sparsification  
        sparse_weights, sparsity_mask = self._sparsify_weights(
            quantized_weights, sparsity_ratio=compression_ratio
        )
        compression_info['techniques_used'].append('sparsification')
        
        # 3. Entropy compression
        serialized_data = pickle.dumps({
            'weights': sparse_weights,
            'quantization_params': quantization_params,
            'sparsity_mask': sparsity_mask
        })
        
        compressed_data = zlib.compress(serialized_data, level=6)
        compression_info['techniques_used'].append('entropy_compression')
        
        compression_info['original_size'] = sum(w.nbytes for w in weights)
        compression_info['compressed_size'] = len(compressed_data)
        compression_info['compression_ratio'] = compression_info['compressed_size'] / compression_info['original_size']
        
        return compressed_data, compression_info
        
    def decompress_weights(self, compressed_data: bytes, 
                          compression_info: Dict) -> List[np.ndarray]:
        """
        Decompress model weights
        """
        # Decompress
        serialized_data = zlib.decompress(compressed_data)
        data_dict = pickle.loads(serialized_data)
        
        sparse_weights = data_dict['weights']
        quantization_params = data_dict['quantization_params']
        sparsity_mask = data_dict['sparsity_mask']
        
        # Reverse sparsification
        dense_weights = self._densify_weights(sparse_weights, sparsity_mask)
        
        # Reverse quantization
        original_weights = self._dequantize_weights(dense_weights, quantization_params)
        
        return original_weights
        
    def _quantize_weights(self, weights: List[np.ndarray], 
                         bits: int = 8) -> Tuple[List[np.ndarray], Dict]:
        """
        Quantize weights to reduce precision
        """
        quantized_weights = []
        quantization_params = {}
        
        for i, weight_matrix in enumerate(weights):
            # Calculate quantization parameters
            w_min, w_max = weight_matrix.min(), weight_matrix.max()
            scale = (w_max - w_min) / (2**bits - 1)
            zero_point = -w_min / scale
            
            # Quantize
            quantized = np.round((weight_matrix - w_min) / scale).astype(np.uint8)
            quantized_weights.append(quantized)
            
            quantization_params[i] = {
                'scale': scale,
                'zero_point': zero_point,
                'min': w_min,
                'max': w_max
            }
            
        return quantized_weights, quantization_params
        
    def _dequantize_weights(self, quantized_weights: List[np.ndarray], 
                           params: Dict) -> List[np.ndarray]:
        """
        Dequantize weights back to original precision
        """
        dequantized_weights = []
        
        for i, quantized in enumerate(quantized_weights):
            scale = params[i]['scale']
            w_min = params[i]['min']
            
            # Dequantize
            dequantized = quantized.astype(np.float32) * scale + w_min
            dequantized_weights.append(dequantized)
            
        return dequantized_weights
        
    def _sparsify_weights(self, weights: List[np.ndarray], 
                         sparsity_ratio: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply magnitude-based sparsification
        """
        sparse_weights = []
        sparsity_masks = []
        
        for weight_matrix in weights:
            # Calculate magnitude threshold
            flat_weights = weight_matrix.flatten()
            threshold = np.percentile(np.abs(flat_weights), sparsity_ratio * 100)
            
            # Create sparsity mask
            mask = np.abs(weight_matrix) >= threshold
            sparsity_masks.append(mask)
            
            # Apply sparsification
            sparse_weight = weight_matrix * mask
            sparse_weights.append(sparse_weight)
            
        return sparse_weights, sparsity_masks
        
    def _densify_weights(self, sparse_weights: List[np.ndarray], 
                        masks: List[np.ndarray]) -> List[np.ndarray]:
        """
        Restore dense weights from sparse representation
        """
        # In this simple implementation, sparse weights already contain the dense structure
        # In practice, would use more sophisticated sparse representations
        return sparse_weights

class IntelligentDeviceSelector:
    """
    Intelligent device selection based on multiple criteria
    """
    
    def __init__(self, selection_strategy: str = "multi_criteria"):
        self.device_profiles: Dict[str, DeviceProfile] = {}
        self.selection_strategy = selection_strategy
        self.selection_history: List[Dict] = []
        self.logger = logging.getLogger(__name__)
        
    def register_device(self, device_profile: DeviceProfile):
        """Register a new device"""
        self.device_profiles[device_profile.device_id] = device_profile
        
    def update_device_profile(self, device_id: str, **kwargs):
        """Update device profile with new metrics"""
        if device_id in self.device_profiles:
            profile = self.device_profiles[device_id]
            for key, value in kwargs.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
                    
    def select_devices(self, num_devices: int, 
                      minimum_requirements: Optional[Dict] = None) -> List[str]:
        """
        Select optimal devices for federated learning round
        """
        if minimum_requirements is None:
            minimum_requirements = {
                'compute_capability': 1.0,
                'memory_capacity': 1.0,
                'battery_level': 0.2,
                'availability_score': 0.5
            }
            
        # Filter devices by minimum requirements
        eligible_devices = []
        for device_id, profile in self.device_profiles.items():
            if self._meets_requirements(profile, minimum_requirements):
                eligible_devices.append(device_id)
                
        if len(eligible_devices) < num_devices:
            self.logger.warning(f"Only {len(eligible_devices)} devices meet requirements, need {num_devices}")
            return eligible_devices
            
        # Apply selection strategy
        if self.selection_strategy == "multi_criteria":
            selected = self._multi_criteria_selection(eligible_devices, num_devices)
        elif self.selection_strategy == "performance_based":
            selected = self._performance_based_selection(eligible_devices, num_devices)
        elif self.selection_strategy == "fairness_aware":
            selected = self._fairness_aware_selection(eligible_devices, num_devices)
        else:
            selected = eligible_devices[:num_devices]  # Simple selection
            
        # Record selection
        selection_record = {
            'timestamp': datetime.now(),
            'selected_devices': selected,
            'eligible_devices': eligible_devices,
            'strategy': self.selection_strategy
        }
        self.selection_history.append(selection_record)
        
        return selected
        
    def _meets_requirements(self, profile: DeviceProfile, 
                          requirements: Dict) -> bool:
        """Check if device meets minimum requirements"""
        for req, min_value in requirements.items():
            if hasattr(profile, req):
                if getattr(profile, req) < min_value:
                    return False
        # Fix for enum comparison issue - compare by string value instead of enum instance
        return profile.status.value == "idle"
        
    def _multi_criteria_selection(self, eligible_devices: List[str], 
                                num_devices: int) -> List[str]:
        """
        Multi-criteria device selection using weighted scoring
        """
        device_scores = {}
        
        # Define weights for different criteria
        weights = {
            'compute_capability': 0.3,
            'memory_capacity': 0.2,
            'network_bandwidth': 0.2,
            'battery_level': 0.1,
            'availability_score': 0.1,
            'reliability_score': 0.1
        }
        
        for device_id in eligible_devices:
            profile = self.device_profiles[device_id]
            score = 0.0
            
            # Calculate weighted score
            for criterion, weight in weights.items():
                criterion_score = getattr(profile, criterion)
                # Normalize communication cost (lower is better)
                if criterion == 'communication_cost':
                    criterion_score = 1.0 / (criterion_score + 1.0)
                score += weight * criterion_score
                
            device_scores[device_id] = score
            
        # Select top devices
        sorted_devices = sorted(device_scores.items(), key=lambda x: x[1], reverse=True)
        return [device_id for device_id, _ in sorted_devices[:num_devices]]
        
    def _performance_based_selection(self, eligible_devices: List[str], 
                                   num_devices: int) -> List[str]:
        """
        Select devices based on performance metrics only
        """
        device_performance = {}
        
        for device_id in eligible_devices:
            profile = self.device_profiles[device_id]
            # Combined performance score
            performance = (profile.compute_capability * 0.6 + 
                         profile.network_bandwidth * 0.4)
            device_performance[device_id] = performance
            
        sorted_devices = sorted(device_performance.items(), key=lambda x: x[1], reverse=True)
        return [device_id for device_id, _ in sorted_devices[:num_devices]]
        
    def _fairness_aware_selection(self, eligible_devices: List[str], 
                                num_devices: int) -> List[str]:
        """
        Fair device selection considering historical participation
        """
        # Calculate participation frequency
        participation_counts = {}
        for device_id in eligible_devices:
            count = sum(1 for record in self.selection_history 
                       if device_id in record['selected_devices'])
            participation_counts[device_id] = count
            
        # Prefer devices with lower participation (fairness)
        sorted_by_fairness = sorted(participation_counts.items(), key=lambda x: x[1])
        
        # Balance fairness with capability
        selected_devices = []
        for device_id, _ in sorted_by_fairness:
            if len(selected_devices) >= num_devices:
                break
                
            profile = self.device_profiles[device_id]
            # Minimum capability threshold for fairness
            if profile.compute_capability >= 0.5 and profile.reliability_score >= 0.3:
                selected_devices.append(device_id)
                
        # Fill remaining slots with best available devices if needed
        if len(selected_devices) < num_devices:
            remaining_devices = [d for d in eligible_devices if d not in selected_devices]
            additional = self._performance_based_selection(
                remaining_devices, num_devices - len(selected_devices)
            )
            selected_devices.extend(additional)
            
        return selected_devices

class AsynchronousTrainingManager:
    """
    Manages asynchronous federated learning with intelligent scheduling
    """
    
    def __init__(self, max_concurrent_devices: int = 10):
        self.max_concurrent_devices = max_concurrent_devices
        self.task_queue = PriorityQueue()
        self.active_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: List[TrainingTask] = []
        self.device_selector = IntelligentDeviceSelector()
        self.compressor = AdaptiveCompression()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_devices)
        self.logger = logging.getLogger(__name__)
        
    async def async_federated_round(self, global_weights: List[np.ndarray], 
                                  device_ids: List[str],
                                  local_epochs: int = 5,
                                  timeout: float = 300.0) -> Dict[str, List[np.ndarray]]:
        """
        Execute asynchronous federated learning round
        """
        start_time = time.time()
        device_updates = {}
        
        # Create training tasks
        tasks = []
        for i, device_id in enumerate(device_ids):
            task = TrainingTask(
                task_id=f"round_{int(time.time())}_{device_id}",
                device_id=device_id,
                priority=1,  # Higher priority for faster devices
                model_weights=global_weights,
                local_epochs=local_epochs,
                deadline=datetime.now() + timedelta(seconds=timeout),
                estimated_duration=self._estimate_training_time(device_id, local_epochs)
            )
            tasks.append(task)
            
        # Execute tasks asynchronously
        futures = []
        future_to_task = {}
        for task in tasks:
            future = self.executor.submit(self._execute_training_task, task)
            futures.append(future)
            future_to_task[future] = task
            
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(futures, timeout=timeout):
            task = future_to_task[future]
            try:
                result = future.result()
                if result is not None:
                    device_updates[task.device_id] = result
                    completed_count += 1
                    
                    # Update device profile with actual training time
                    actual_time = time.time() - start_time
                    self.device_selector.update_device_profile(
                        task.device_id, 
                        avg_training_time=actual_time,
                        last_seen=datetime.now()
                    )
                    
            except Exception as e:
                self.logger.error(f"Training failed for device {task.device_id}: {e}")
                # Update device profile to reflect failure
                self.device_selector.update_device_profile(
                    task.device_id,
                    reliability_score=max(0.1, 
                        self.device_selector.device_profiles[task.device_id].reliability_score * 0.9
                    )
                )
                
        self.logger.info(f"Async round completed: {completed_count}/{len(device_ids)} devices participated")
        return device_updates
        
    def _execute_training_task(self, task: TrainingTask) -> Optional[List[np.ndarray]]:
        """
        Simulate execution of a training task
        """
        device_profile = self.device_selector.device_profiles.get(task.device_id)
        if not device_profile:
            return None
            
        # Update device status
        device_profile.status = DeviceStatus.TRAINING
        
        # Simulate training time based on device capability
        base_training_time = task.local_epochs * 2.0  # 2 seconds per epoch base
        actual_training_time = base_training_time / device_profile.compute_capability
        
        # Add some randomness
        actual_training_time *= np.random.uniform(0.8, 1.2)
        
        time.sleep(min(actual_training_time, 10))  # Cap at 10 seconds for simulation
        
        # Simulate generating weight updates
        updated_weights = []
        for weight in task.model_weights:
            # Add small random updates
            update = weight + np.random.normal(0, 0.01, weight.shape)
            updated_weights.append(update)
            
        # Update device status
        device_profile.status = DeviceStatus.IDLE
        
        return updated_weights
        
    def _estimate_training_time(self, device_id: str, local_epochs: int) -> float:
        """
        Estimate training time for a device
        """
        profile = self.device_selector.device_profiles.get(device_id)
        if not profile:
            return local_epochs * 5.0  # Default estimate
            
        # Use historical data if available
        if profile.avg_training_time > 0:
            return profile.avg_training_time * local_epochs
        else:
            # Estimate based on compute capability
            base_time = 3.0  # seconds per epoch
            return (base_time / profile.compute_capability) * local_epochs

class PerformanceOptimizer:
    """
    Main performance optimization coordinator
    """
    
    def __init__(self):
        self.device_selector = IntelligentDeviceSelector()
        self.training_manager = AsynchronousTrainingManager()
        self.compressor = AdaptiveCompression()
        self.logger = logging.getLogger(__name__)
        self.performance_metrics: Dict[str, List[float]] = {
            'round_duration': [],
            'communication_overhead': [],
            'device_utilization': [],
            'convergence_rate': []
        }
        
    def optimize_federated_round(self, global_weights: List[np.ndarray],
                               available_devices: List[str],
                               target_devices: int,
                               local_epochs: int) -> Tuple[Dict[str, List[np.ndarray]], Dict]:
        """
        Execute optimized federated learning round
        """
        start_time = time.time()
        
        # 1. Intelligent device selection
        selected_devices = self.device_selector.select_devices(target_devices)
        selected_devices = [d for d in selected_devices if d in available_devices]
        
        # 2. Compress global model for efficient distribution
        compressed_model, compression_info = self.compressor.compress_weights(global_weights)
        
        # 3. Execute training (Jupyter-compatible approach)
        device_updates = {}
        
        try:
            # Check if we're in a Jupyter notebook environment
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Use notebook-compatible synchronous training with proper device handling
                device_updates = self._execute_synchronous_training(
                    global_weights, selected_devices, local_epochs
                )
            else:
                # Use full asynchronous training
                device_updates = asyncio.run(
                    self.training_manager.async_federated_round(
                        global_weights, selected_devices, local_epochs
                    )
                )
        except (RuntimeError, Exception) as e:
            # Fallback to synchronous training
            self.logger.warning(f"Async training failed, using synchronous fallback: {e}")
            device_updates = self._execute_synchronous_training(
                global_weights, selected_devices, local_epochs
            )
        
        # 4. Record performance metrics
        round_duration = time.time() - start_time
        self.performance_metrics['round_duration'].append(round_duration)
        self.performance_metrics['communication_overhead'].append(compression_info['compression_ratio'])
        self.performance_metrics['device_utilization'].append(len(device_updates) / len(selected_devices))
        
        optimization_report = {
            'selected_devices': selected_devices,
            'participating_devices': list(device_updates.keys()),
            'round_duration': round_duration,
            'compression_ratio': compression_info['compression_ratio'],
            'device_utilization': len(device_updates) / len(selected_devices),
            'performance_metrics': self._get_performance_summary()
        }
        
        return device_updates, optimization_report
        
    def _get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        summary = {}
        
        for metric, values in self.performance_metrics.items():
            if values:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'latest': values[-1]
                }
            else:
                summary[metric] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'latest': 0}
                
        return summary

    def _execute_synchronous_training(self, global_weights: List[np.ndarray],
                                    device_ids: List[str],
                                    local_epochs: int) -> Dict[str, List[np.ndarray]]:
        """
        Execute synchronous training for Jupyter notebook compatibility
        """
        device_updates = {}
        
        for device_id in device_ids:
            try:
                # Get device profile for realistic simulation
                device_profile = self.device_selector.device_profiles.get(device_id)
                
                if device_profile:
                    # Update device status
                    device_profile.status = DeviceStatus.TRAINING
                    
                    # Simulate training time based on device capability
                    base_training_time = local_epochs * 0.5  # 0.5 seconds per epoch
                    actual_training_time = base_training_time / device_profile.compute_capability
                    
                    # Cap training time for notebook responsiveness
                    actual_training_time = min(actual_training_time, 2.0)
                    
                    # Simulate training
                    time.sleep(actual_training_time)
                    
                    # Generate realistic weight updates (small gradient-like changes)
                    updated_weights = []
                    for weight in global_weights:
                        # Simulate gradient descent updates
                        gradient_noise = np.random.normal(0, 0.001, weight.shape)
                        learning_rate = 0.01 / device_profile.compute_capability
                        update = weight - learning_rate * gradient_noise
                        updated_weights.append(update)
                        
                    device_updates[device_id] = updated_weights
                    
                    # Update device status back to idle
                    device_profile.status = DeviceStatus.IDLE
                    
                    # Update device reliability based on successful training
                    device_profile.reliability_score = min(1.0, device_profile.reliability_score * 1.05)
                    
                else:
                    # Fallback for devices without profiles
                    updated_weights = []
                    for weight in global_weights:
                        gradient_noise = np.random.normal(0, 0.001, weight.shape)
                        update = weight - 0.01 * gradient_noise
                        updated_weights.append(update)
                    device_updates[device_id] = updated_weights
                    
            except Exception as e:
                self.logger.warning(f"Training failed for device {device_id}: {e}")
                continue
                
        self.logger.info(f"Synchronous training completed: {len(device_updates)}/{len(device_ids)} devices participated")
        return device_updates
