

from data_utils import load_and_prepare_medical_data, split_data_among_devices
from IoMTDevice import IoMTDevice
from Server import FederatedServer
import matplotlib.pyplot as plt

def simulate_federated_iomt_scenario(
    device_names=None,
    split_ratios=None,
    num_rounds=10,
    local_epochs=3
):
    """
    Simulate a federated learning scenario with multiple IoMT devices
    Each device represents a different hospital/clinic with local patient data
    
    Args:
        device_names: List of device names (default: 4 medical institutions)
        split_ratios: Data distribution ratios (default: [0.4, 0.3, 0.2, 0.1])
        num_rounds: Number of federated learning rounds
        local_epochs: Number of local training epochs per round
    
    Returns:
        Tuple of (server, devices) for further analysis
    """
    print("=== IoMT Federated Learning Simulation ===\n")
    
    # Set defaults
    if device_names is None:
        device_names = ["Hospital_A", "Clinic_B", "Medical_Center_C", "Health_Station_D"]
    if split_ratios is None:
        split_ratios = [0.4, 0.3, 0.2, 0.1]
    
    # Load and prepare data
    X, y = load_and_prepare_medical_data()
    
    # Split data among devices
    device_data = split_data_among_devices(X, y, device_names, split_ratios)
    
    # Create IoMT devices
    devices = []
    for device_name, (X_device, y_device) in zip(device_names, device_data):
        device = IoMTDevice(device_name, X_device, y_device, privacy_budget=2.0)
        devices.append(device)

    # Initialize federated server
    print(f"\nInitializing Federated Server...")
    server = FederatedServer(X.shape[1])

    # Register devices with server
    for device in devices:
        server.register_device(device)

    # Run federated learning rounds
    print("\nStarting Privacy-Preserved Federated Learning...")

    
    for round_num in range(num_rounds):
        accuracy = server.train_round(local_epochs)
        
        if round_num % 2 == 1:  # Print detailed results every other round
            print(f"Individual device accuracies:")
            device_accuracies = server.get_device_accuracies()
            for device_id, acc in device_accuracies.items():
                print(f"  {device_id}: {acc:.4f}")
    
    # Generate privacy audit report
    print("\n=== Privacy Audit Report ===")
    audit_report = server.generate_privacy_audit_report()
    
    print(f"Privacy techniques implemented: {audit_report['privacy_techniques_implemented']}")
    print(f"Devices with exhausted privacy budget: {audit_report['privacy_budget_exhausted_devices']}")
    
    for device_report in audit_report['device_privacy_status']:
        print(f"Device {device_report['device_id']}: "
              f"Used {device_report['used_privacy_budget']:.2f}/"
              f"{device_report['total_privacy_budget']:.2f} privacy budget")
    
    return server, devices


def analyze_results(server, devices):
    """
    Analyze and display results from federated learning simulation
    
    Args:
        server: FederatedServer instance
        devices: List of IoMTDevice instances
    """
    print("\n=== Final Results Analysis ===")
    
    # Final accuracies
    final_accuracies = server.get_device_accuracies()
    print("\nFinal Device Accuracies:")
    for device_id, accuracy in final_accuracies.items():
        print(f"  {device_id}: {accuracy:.4f}")
    
    # Training history
    history = server.get_training_history()

    print("\nTraining History is:")
    print(history)
    print(f"\nTraining Progress:")
    print(f"  Initial Average Accuracy: {history[0]:.4f}")
    print(f"  Final Average Accuracy: {history[-1]:.4f}")
    print(f"  Improvement: {history[-1] - history[0]:.4f}")
    
    round_numbers = list(range(1, len(history) + 1))
    plt.legend(['Average Accuracy'])
    plt.xlabel('Number of Rounds')
    plt.ylabel('Accuracy')
    plt.title('Federated Learning Training Progress')
    plt.grid()
    plt.plot(round_numbers, history, marker='o', label='Average Accuracy')

    # Data distribution
    print(f"\nData Distribution:")
    total_samples = sum(device.get_data_size() for device in devices)
    for device in devices:
        ratio = device.get_data_size() / total_samples
        print(f"  {device.device_id}: {device.get_data_size()} samples ({ratio:.1%})")


if __name__ == "__main__":
    # Run simulation
    server, devices = simulate_federated_iomt_scenario()
    
    # Analyze results
    analyze_results(server, devices)