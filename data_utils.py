# Real medical data utilities using sklearn
import numpy as np
from typing import List

# Try to import sklearn components
SKLEARN_AVAILABLE = False

try:
    from sklearn.datasets import load_breast_cancer
    SKLEARN_AVAILABLE = True
    del load_breast_cancer  # Clean up, we'll import locally where needed
except ImportError:
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


def load_and_prepare_medical_data():
    """Load and prepare real medical dataset (Breast Cancer Wisconsin) with enhanced preprocessing"""
    print("Loading and enhancing real medical dataset (Breast Cancer Wisconsin)...")
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn is required to load real medical datasets. Please install scikit-learn.")
    
    # Import locally to avoid type checking issues
    from sklearn.datasets import load_breast_cancer  # type: ignore
    from sklearn.preprocessing import RobustScaler  # type: ignore
    from sklearn.model_selection import train_test_split  # type: ignore
    
    # Load the real breast cancer dataset
    data = load_breast_cancer()
    X = data.data  # type: ignore
    y = data.target  # type: ignore

    print(f"Real dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {data.target_names}")  # type: ignore
    print(f"Class distribution: {np.bincount(y)}")
    
    # Enhanced preprocessing pipeline
    # 1. Use RobustScaler for better outlier handling in medical data
    scaler = RobustScaler()  # type: ignore
    X_scaled = scaler.fit_transform(X)
    
    # 2. Add feature engineering - polynomial features for critical medical indicators
    # Focus on the most important features (first 10 are mean values)
    X_enhanced = X_scaled.copy()
    
    # Add some interaction features for better medical pattern recognition
    # Mean radius * mean texture (tumor characteristics interaction)
    radius_texture = X_scaled[:, 0:1] * X_scaled[:, 1:2]  
    # Mean perimeter * mean area (size consistency)
    perimeter_area = X_scaled[:, 2:3] * X_scaled[:, 3:4]
    
    # Concatenate enhanced features
    X_final = np.concatenate([X_enhanced, radius_texture, perimeter_area], axis=1)
    
    print(f"Enhanced features: {X_final.shape[1]} total features (original: {X.shape[1]})")
    
    # 3. Ensure balanced representation by stratified shuffling
    X_shuffled, _, y_shuffled, _ = train_test_split(X_final, y, test_size=0.01, 
                                                    stratify=y, random_state=42)
    
    # Reshape y for neural network compatibility
    y_reshaped = y_shuffled.reshape(-1, 1).astype(np.float32)
    X_final_float = X_shuffled.astype(np.float32)
    
    print(f"Final preprocessed data: {X_final_float.shape}, labels: {y_reshaped.shape}")
    print(f"Data type: {X_final_float.dtype}, Labels type: {y_reshaped.dtype}")
    
    return X_final_float, y_reshaped


def create_medical_device_names(num_devices: int = 5) -> List[str]:
    """Create realistic medical device names for IoMT network"""
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