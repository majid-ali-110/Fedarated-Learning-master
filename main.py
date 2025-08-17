from data_utils import load_and_prepare_medical_data, split_data_among_devices

if __name__ == "__main__":
    # Load and prepare the medical data
    X, y = load_and_prepare_medical_data()
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")