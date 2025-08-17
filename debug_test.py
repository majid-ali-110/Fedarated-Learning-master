"""
Simple test to debug the enhanced federated learning system
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Test basic imports first
print("Testing basic imports...")
try:
    from data_utils import load_and_prepare_medical_data
    print("✅ data_utils imported successfully")
except Exception as e:
    print(f"❌ data_utils import failed: {e}")

try:
    from NeuralNetworkModal import SimpleNeuralNetwork
    print("✅ NeuralNetworkModal imported successfully")
except Exception as e:
    print(f"❌ NeuralNetworkModal import failed: {e}")

# Test enhanced components
print("\nTesting enhanced components...")
try:
    from EnhancedPrivacyPreserver import EnhancedPrivacyPreserver
    print("✅ EnhancedPrivacyPreserver imported successfully")
except Exception as e:
    print(f"❌ EnhancedPrivacyPreserver import failed: {e}")
    EnhancedPrivacyPreserver = None

# Import medical components
try:
    from EnhancedMedicalIoMTDevice import (
        EnhancedMedicalIoMTDevice, MedicalDataRecord, MedicalDataType, 
        ComplianceFramework, PatientConsent
    )
    print("✅ EnhancedMedicalIoMTDevice imported successfully")
    medical_imports_available = True
except Exception as e:
    print(f"❌ EnhancedMedicalIoMTDevice import failed: {e}")
    EnhancedMedicalIoMTDevice = None
    MedicalDataRecord = None 
    MedicalDataType = None
    ComplianceFramework = None
    PatientConsent = None
    medical_imports_available = False

# Test basic data creation
print("\nTesting basic data creation...")
if medical_imports_available and MedicalDataRecord is not None and MedicalDataType is not None and PatientConsent is not None:
    # Import again within the conditional to satisfy type checker
    from EnhancedMedicalIoMTDevice import (
        MedicalDataRecord as MDR, MedicalDataType as MDT, 
        PatientConsent as PC
    )
    
    try:
        # Create a simple medical record
        test_data = np.array([120.0, 80.0, 36.5, 75.0])  # Simple vital signs
        
        record = MDR(
            record_id="test_001",
            patient_id="patient_001",
            data_type=MDT.VITAL_SIGNS,
            timestamp=datetime.now(),
            data=test_data,
            metadata={"source": "test"},
            sensitivity_level=2,
            consent_verified=True
        )
        
        print(f"✅ Medical record created: {record.record_id}")
        print(f"   Data shape: {record.data.shape}")
        print(f"   Data type: {record.data_type}")
        
        # Create a simple consent
        consent = PC(
            patient_id="patient_001",
            consent_given=True,
            consent_timestamp=datetime.now(),
            data_types_consented=[MDT.VITAL_SIGNS],
            expiration_date=datetime.now() + timedelta(days=365),
            withdrawal_allowed=True
        )
        
        print(f"✅ Patient consent created for: {consent.patient_id}")
        
    except Exception as e:
        print(f"❌ Basic data creation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  Skipping basic data creation - medical imports not available")

# Test device creation
print("\nTesting device creation...")
if medical_imports_available and all(cls is not None for cls in [MedicalDataRecord, MedicalDataType, EnhancedMedicalIoMTDevice, ComplianceFramework, PatientConsent]):
    # Import again within the conditional to satisfy type checker
    from EnhancedMedicalIoMTDevice import (
        MedicalDataRecord as MDR, MedicalDataType as MDT, 
        EnhancedMedicalIoMTDevice as EMID, ComplianceFramework as CF,
        PatientConsent as PC
    )
    
    try:
        # Create multiple test records
        test_records = []
        for i in range(5):
            test_data = np.random.normal([120, 80, 36.5, 75], [5, 5, 0.5, 5], 4).astype(np.float32)
            
            record = MDR(
                record_id=f"test_{i:03d}",
                patient_id=f"patient_{i:03d}",
                data_type=MDT.VITAL_SIGNS,
                timestamp=datetime.now(),
                data=test_data,
                metadata={"source": "test"},
                sensitivity_level=2,
                consent_verified=True
            )
            test_records.append(record)
        
        print(f"✅ Created {len(test_records)} test records")
        
        # Create test device
        device = EMID(
            device_id="test_device",
            device_type="wearable_monitor",
            medical_data=test_records,
            compliance_frameworks=[CF.HIPAA]
        )
        
        # Add consents
        for i in range(5):
            consent = PC(
                patient_id=f"patient_{i:03d}",
                consent_given=True,
                consent_timestamp=datetime.now(),
                data_types_consented=[MDT.VITAL_SIGNS],
                expiration_date=datetime.now() + timedelta(days=365),
                withdrawal_allowed=True
            )
            device.privacy_manager.patient_consents[f"patient_{i:03d}"] = consent
        
        print(f"✅ Device created: {device.device_id}")
        print(f"   Training data shape: {device.X_local.shape if len(device.X_local) > 0 else 'No data'}")
        print(f"   Number of patients: {len(set(r.patient_id for r in device.medical_data))}")
        print(f"   Consents registered: {len(device.privacy_manager.patient_consents)}")
        
        # Test participation check
        can_participate, reason = device.can_participate_in_round()
        print(f"   Can participate: {can_participate} ({reason})")
        
    except Exception as e:
        print(f"❌ Device creation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠️  Skipping device creation - medical imports not available")

print("\n🎯 Debugging complete!")
