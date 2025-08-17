# Real medical data utilities using sklearn
import numpy as np
from typing import Tuple, List, Any

# Try to import sklearn components
SKLEARN_AVAILABLE = False

try:
    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


def load_and_prepare_medical_data():
    """Load and prepare real medical dataset (Breast Cancer Wisconsin)"""
    print("Loading real medical dataset (Breast Cancer Wisconsin)...")
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required to load real medical datasets. Please install scikit-learn.")
    
    # Import locally to avoid type checking issues
    from sklearn.datasets import load_breast_cancer  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    
    # Load the real breast cancer dataset
    data = load_breast_cancer()
    X = data.data  # type: ignore
    y = data.target  # type: ignore

    print(f"Real dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {data.target_names}")  # type: ignore
    print(f"Features: First 5 features - {data.feature_names[:5]}")  # type: ignore
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()  # type: ignore
    X_scaled = scaler.fit_transform(X)
    
    # Reshape y for neural network
    y_reshaped = y.reshape(-1, 1)
    
    return X_scaled, y_reshaped


def split_data_among_devices(X, y, device_names, split_ratios=None):
    """
    Args:
        X: Feature data
        y: Target data
        device_names: List of device names
        split_ratios: List of ratios for data distribution (optional)
    
    Returns:
        List of tuples (X_device, y_device) for each device
    """
    if split_ratios is None:
        split_ratios = [0.4, 0.3, 0.2, 0.1]  # Default ratios
    
    print("\nDistributing data among IoMT devices...")
    
    # Ensure split_ratios match device_names length
    if len(split_ratios) != len(device_names):
        # Distribute equally if ratios don't match
        split_ratios = [1.0 / len(device_names)] * len(device_names)
    
    total_samples = len(X)
    start_idx = 0
    device_data = []
    
    for i, (device_name, ratio) in enumerate(zip(device_names, split_ratios)):
        end_idx = start_idx + int(total_samples * ratio)
        # print(f"end index is {end_idx}")
        
        if i == len(device_names) - 1:  # Last device gets remaining data
            end_idx = total_samples
            
        X_device = X[start_idx:end_idx]
        y_device = y[start_idx:end_idx]
        
        device_data.append((X_device, y_device))
        start_idx = end_idx
    return device_data


def generate_medical_data(num_samples: int = 300, input_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic medical data for federated learning"""
    np.random.seed(42)  # For reproducibility
    
    # Generate synthetic medical features
    X = np.random.randn(num_samples, input_size).astype(np.float32)
    
    # Create realistic medical data patterns
    # Simulate vital signs, lab results, and measurements
    for i in range(num_samples):
        if i % 2 == 0:  # Healthy pattern
            X[i, :input_size//2] = np.random.normal(0, 0.5, input_size//2)  # Normal vitals
            X[i, input_size//2:] = np.random.normal(0, 0.3, input_size//2)  # Normal labs
        else:  # At-risk pattern
            X[i, :input_size//2] = np.random.normal(1.2, 0.8, input_size//2)  # Elevated vitals
            X[i, input_size//2:] = np.random.normal(1.0, 0.6, input_size//2)  # Abnormal labs
    
    # Binary classification labels
    y = np.array([i % 2 for i in range(num_samples)], dtype=np.int32)
    
    return X, y


def create_synthetic_devices(num_devices: int = 5) -> List[str]:
    """Create synthetic device names for medical IoMT network"""
    device_names = [
        "MetroGeneral_Hospital",
        "CityHealth_Clinic", 
        "RegionalMedical_Center",
        "Community_Health_Station",
        "University_Research_Center",
        "Rural_Medical_Outpost",
        "Emergency_Response_Unit",
        "Mobile_Health_Station"
    ]
    
    return device_names[:num_devices]