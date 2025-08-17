# 🏥 Multi-Algorithm Enhanced Federated Learning for Medical IoMT Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-red)

## 🎯 Overview

A comprehensive **multi-algorithm federated learning system** specifically designed for medical IoMT (Internet of Medical Things) networks. This system implements 5 different federated learning algorithms with enhanced privacy, security, and comprehensive visualization capabilities for healthcare applications.

### ✨ Key Features

- 🧠 **Multi-Algorithm Support**: FedAvg, FedProx, SCAFFOLD, FedNova, Personalized FL
- 📊 **Comprehensive Visualization**: Detailed charts comparing all algorithms performance  
- 🔒 **Advanced Privacy**: Enhanced privacy preservation with differential privacy
- 🛡️ **Robust Security**: Advanced security manager with threat detection
- ⚡ **Performance Optimization**: Intelligent device selection and optimization
- 🏥 **Medical Focus**: HIPAA/GDPR compliance and healthcare-specific requirements
- 📈 **Real Medical Data**: Uses authentic Breast Cancer Wisconsin dataset
- 🎨 **Interactive Dashboard**: Complete visualization dashboard with algorithm comparison

## 🏗️ System Architecture

### Core Components

#### 1. **Multi-Algorithm Federated Server** (`Server.py`)
- **FederatedServer**: Main server coordinating federated learning
- **5 Federated Learning Algorithms**:
  - `FedAvg`: Standard federated averaging
  - `FedProx`: Proximal term for handling heterogeneity
  - `SCAFFOLD`: Control variates for reduced variance
  - `FedNova`: Normalized averaging with step correction
  - `Personalized FL`: Personalized model layers
- **Algorithm Configuration**: Flexible algorithm-specific parameter tuning

#### 2. **Enhanced Privacy & Security**
- **EnhancedPrivacyPreserver**: Rényi Differential Privacy with adaptive clipping
- **AdvancedSecurityManager**: Byzantine fault tolerance and threat detection
- **Privacy Budgeting**: Sophisticated privacy budget management
- **Secure Aggregation**: Cryptographic protection of model updates

#### 3. **Medical IoMT Device Network**
- **IoMTDevice**: Basic federated learning device
- **EnhancedMedicalIoMTDevice**: Medical-specific device with healthcare compliance
- **Real Medical Data**: Breast Cancer Wisconsin dataset integration
- **Privacy-Preserving Training**: Local differential privacy mechanisms

#### 4. **Performance & Monitoring**
- **PerformanceOptimizer**: Asynchronous training and intelligent device selection
- **ComprehensiveMonitor**: Real-time logging, privacy auditing, and performance tracking
- **Detailed Analytics**: Algorithm comparison and convergence analysis

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
pip install -r requirements.txt
```

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Fedarated-Learning-master
```

2. **Create virtual environment**:
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Option 1: Jupyter Notebook (Recommended)
```bash
jupyter notebook main.ipynb
```

Run the comprehensive notebook for:
- Interactive algorithm comparison
- Real-time visualization
- Detailed performance analysis
- Complete system demonstration

#### Option 2: Python Script
```python
from Server import FederatedServer, FederatedAlgorithm, AlgorithmConfig
from data_utils import load_and_prepare_medical_data, split_data_among_devices
from IoMTDevice import IoMTDevice

# Load real medical dataset
X, y = load_and_prepare_medical_data()

# Create federated learning server
server = FederatedServer(input_size=30, output_size=2)

# Configure algorithm (example: FedProx)
config = AlgorithmConfig(FederatedAlgorithm.FEDPROX, mu=0.1)
server.set_algorithm(FederatedAlgorithm.FEDPROX, config)

# Create and register devices
devices = []
device_data = split_data_among_devices(X, y, device_names=["Hospital_A", "Clinic_B"])
for i, (X_device, y_device) in enumerate(device_data):
    device = IoMTDevice(f"Medical_Device_{i+1}", X_device, y_device)
    devices.append(device)
    server.register_device(device)

# Train federated model
results = server.train_round_multi_algorithm(local_epochs=3)
print(f"Training accuracy: {results.accuracy:.4f}")
```

## 📊 Federated Learning Algorithms

### 1. **FedAvg** (Federated Averaging)
- **Description**: Standard federated learning approach
- **Use Case**: Balanced, IID data distribution
- **Parameters**: Basic configuration

### 2. **FedProx** (Federated Proximal)
- **Description**: Adds proximal term to handle client heterogeneity
- **Use Case**: Non-IID data, system heterogeneity
- **Parameters**: `mu` (proximal term strength)

### 3. **SCAFFOLD**
- **Description**: Uses control variates to reduce client drift
- **Use Case**: Highly heterogeneous environments
- **Parameters**: `server_learning_rate` (server-side learning rate)

### 4. **FedNova**
- **Description**: Normalized averaging with step size correction
- **Use Case**: Varying local computation capabilities
- **Parameters**: `tau` (momentum parameter)

### 5. **Personalized FL**
- **Description**: Personalized model layers for each client
- **Use Case**: Highly personalized medical applications
- **Parameters**: Personalization ratio configuration

## 🏥 Medical Dataset Integration

### Real Medical Data
- **Dataset**: Breast Cancer Wisconsin Diagnostic Dataset
- **Features**: 30 real medical features (radius, texture, perimeter, area, smoothness, etc.)
- **Samples**: 569 patient records
- **Labels**: Malignant vs Benign classification
- **Privacy**: All data processing follows medical privacy standards

### Data Distribution
- **Federated Split**: Realistic hospital/clinic distribution simulation
- **Non-IID Handling**: Algorithms specifically designed for medical data heterogeneity
- **Privacy Preservation**: Local differential privacy during training

## 📈 Visualization & Analysis

### Comprehensive Dashboard
- **Algorithm Comparison**: Side-by-side performance analysis
- **Convergence Plots**: Real-time training progress
- **Performance Metrics**: Accuracy, loss, convergence rate
- **Privacy Analysis**: Privacy budget utilization tracking
- **Security Monitoring**: Threat detection and mitigation logs

### Generated Visualizations
- Algorithm accuracy progression
- Final performance comparison  
- Performance distribution analysis
- Round-by-round detailed metrics
- Privacy and security audit reports

## 🔒 Privacy & Security Features

### Enhanced Privacy
- **Rényi Differential Privacy**: Advanced privacy accounting
- **Adaptive Noise**: Dynamic noise calibration
- **Privacy Budget Management**: Sophisticated budget allocation
- **HIPAA/GDPR Compliance**: Healthcare regulatory compliance

### Advanced Security
- **Byzantine Fault Tolerance**: Robust against malicious clients
- **Secure Aggregation**: Cryptographic model update protection
- **Threat Detection**: Real-time security monitoring
- **Audit Trails**: Complete system activity logging

## 📁 Project Structure

```
Fedarated-Learning-master/
├── Server.py                          # Multi-algorithm federated server
├── IoMTDevice.py                      # Basic IoMT device
├── EnhancedMedicalIoMTDevice.py       # Medical-specific device
├── data_utils.py                      # Real medical data utilities
├── NeuralNetworkModal.py              # Neural network model
├── EnhancedPrivacyPreserver.py        # Advanced privacy mechanisms
├── AdvancedSecurityManager.py         # Security management
├── PerformanceOptimizer.py            # Performance optimization
├── ComprehensiveMonitor.py            # System monitoring
├── main.ipynb                         # Comprehensive demo notebook
├── main.py                            # Python script entry point
├── requirements.txt                   # Python dependencies
├── README.md                          # This documentation
└── .gitignore                         # Git ignore rules
```

## 🧪 Research & Development

### Implemented Research
- **Multi-Algorithm Comparison**: Empirical analysis of 5 FL algorithms
- **Medical IoMT Applications**: Healthcare-specific federated learning
- **Privacy-Preserving ML**: Differential privacy in federated settings
- **System Heterogeneity**: Handling diverse medical device capabilities

### Future Enhancements
- Additional federated learning algorithms (FedMA, FedBN, etc.)
- Advanced attack scenarios and defenses
- Real-time deployment capabilities
- Integration with actual medical device networks
- Extended privacy accounting mechanisms

## 📊 Performance Benchmarks

### Algorithm Performance (Breast Cancer Dataset)
- **FedAvg**: Baseline federated averaging performance
- **FedProx**: Improved handling of data heterogeneity
- **SCAFFOLD**: Reduced variance in heterogeneous settings
- **FedNova**: Better convergence with varying local updates
- **Personalized FL**: Enhanced performance for client-specific patterns

### System Capabilities
- **Scalability**: Tested with multiple simulated medical devices
- **Privacy**: ε-differential privacy with configurable privacy budgets
- **Security**: Robust against Byzantine attacks and data poisoning
- **Efficiency**: Optimized communication and computation

## 🤝 Contributing

We welcome contributions to enhance the federated learning system:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines
- Follow PEP 8 coding standards
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for changes
- Ensure privacy and security compliance

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Federated Learning Research Community**: For foundational algorithms and concepts
- **Medical AI Researchers**: For healthcare-specific federated learning insights
- **Privacy-Preserving ML Community**: For differential privacy implementations
- **Open Source Contributors**: For tools and libraries that made this possible

## 📞 Contact & Support

- **Issues**: [GitHub Issues](issues)
- **Discussions**: [GitHub Discussions](discussions)
- **Documentation**: Available in notebook and code comments
- **Research Papers**: References available in code documentation

---

**🚀 Ready to revolutionize medical federated learning with multi-algorithm support and comprehensive privacy protection!**
