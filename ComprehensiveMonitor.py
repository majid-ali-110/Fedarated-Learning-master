"""
Comprehensive Monitoring and Visualization Module for Federated Learning
Provides detailed logging, privacy auditing, performance tracking, and real-time visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RoundMetrics:
    round_number: int
    timestamp: datetime
    participating_devices: List[str]
    average_accuracy: float
    individual_accuracies: Dict[str, float]
    privacy_budget_consumed: Dict[str, float]
    byzantine_devices_detected: List[str]
    communication_overhead: float
    round_duration: float
    convergence_metric: float
    security_events: List[Dict]

class FederatedLearningLogger:
    """
    Advanced logging system for federated learning
    """
    
    def __init__(self, log_dir: str = "fl_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup different loggers
        self.setup_loggers()
        
        # Metrics storage
        self.round_metrics: List[RoundMetrics] = []
        self.device_metrics: Dict[str, List[Dict]] = defaultdict(list)
        self.security_events: List[Dict] = []
        self.privacy_events: List[Dict] = []
        
    def setup_loggers(self):
        """Setup structured logging"""
        # Main FL logger
        self.fl_logger = logging.getLogger('federated_learning')
        self.fl_logger.setLevel(logging.INFO)
        
        # Privacy logger
        self.privacy_logger = logging.getLogger('privacy')
        self.privacy_logger.setLevel(logging.INFO)
        
        # Security logger  
        self.security_logger = logging.getLogger('security')
        self.security_logger.setLevel(logging.WARNING)
        
        # Performance logger
        self.performance_logger = logging.getLogger('performance')
        self.performance_logger.setLevel(logging.INFO)
        
        # Setup file handlers
        handlers = [
            ('federated_learning.log', self.fl_logger),
            ('privacy.log', self.privacy_logger),
            ('security.log', self.security_logger),
            ('performance.log', self.performance_logger)
        ]
        
        for filename, logger in handlers:
            handler = logging.FileHandler(self.log_dir / filename)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
    def log_round_start(self, round_number: int, participating_devices: List[str]):
        """Log start of federated learning round"""
        self.fl_logger.info(
            f"Round {round_number} started with {len(participating_devices)} devices: {participating_devices}"
        )
        
    def log_round_completion(self, metrics: RoundMetrics):
        """Log completion of federated learning round"""
        self.round_metrics.append(metrics)
        
        self.fl_logger.info(
            f"Round {metrics.round_number} completed - "
            f"Avg Accuracy: {metrics.average_accuracy:.4f}, "
            f"Duration: {metrics.round_duration:.2f}s, "
            f"Devices: {len(metrics.participating_devices)}"
        )
        
        # Log privacy information
        total_privacy_consumed = sum(metrics.privacy_budget_consumed.values())
        self.privacy_logger.info(
            f"Round {metrics.round_number} privacy consumption: {total_privacy_consumed:.4f}"
        )
        
        # Log security events
        if metrics.byzantine_devices_detected:
            self.security_logger.warning(
                f"Round {metrics.round_number} - Byzantine devices detected: {metrics.byzantine_devices_detected}"
            )
            
    def log_device_metrics(self, device_id: str, metrics: Dict):
        """Log device-specific metrics"""
        metrics['timestamp'] = datetime.now()
        self.device_metrics[device_id].append(metrics)
        
        self.performance_logger.info(
            f"Device {device_id} metrics: {json.dumps(metrics, default=str)}"
        )
        
    def log_privacy_event(self, device_id: str, mechanism: str, 
                         epsilon: float, delta: float = 0.0):
        """Log privacy mechanism usage"""
        event = {
            'timestamp': datetime.now(),
            'device_id': device_id,
            'mechanism': mechanism,
            'epsilon': epsilon,
            'delta': delta
        }
        self.privacy_events.append(event)
        
        self.privacy_logger.info(
            f"Privacy mechanism applied - Device: {device_id}, "
            f"Mechanism: {mechanism}, ε: {epsilon}, δ: {delta}"
        )
        
    def log_security_event(self, event_type: str, device_id: str, 
                          description: str, threat_level: str):
        """Log security events"""
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'device_id': device_id,
            'description': description,
            'threat_level': threat_level
        }
        self.security_events.append(event)
        
        self.security_logger.warning(
            f"Security event - Type: {event_type}, Device: {device_id}, "
            f"Threat: {threat_level}, Description: {description}"
        )
        
    def export_logs(self, format: str = 'json') -> str:
        """Export all logs to specified format"""
        export_data = {
            'round_metrics': [asdict(m) for m in self.round_metrics],
            'device_metrics': dict(self.device_metrics),
            'security_events': self.security_events,
            'privacy_events': self.privacy_events,
            'export_timestamp': datetime.now()
        }
        
        filename = None
        
        if format == 'json':
            filename = self.log_dir / f"fl_logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        elif format == 'csv':
            # Export as separate CSV files
            filename_base = self.log_dir / f"fl_logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filename = f"{filename_base}_main"
            
            # Round metrics
            pd.DataFrame([asdict(m) for m in self.round_metrics]).to_csv(
                f"{filename_base}_rounds.csv", index=False
            )
            
            # Security events
            pd.DataFrame(self.security_events).to_csv(
                f"{filename_base}_security.csv", index=False
            )
            
            # Privacy events
            pd.DataFrame(self.privacy_events).to_csv(
                f"{filename_base}_privacy.csv", index=False
            )
        else:
            # Default case
            filename = self.log_dir / f"fl_logs_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
        return str(filename)

class PrivacyAuditor:
    """
    Advanced privacy auditing and compliance checking
    """
    
    def __init__(self, privacy_budget_limit: float = 1.0):
        self.privacy_budget_limit = privacy_budget_limit
        self.audit_results: List[Dict] = []
        
    def audit_privacy_compliance(self, round_metrics: List[RoundMetrics], 
                                privacy_events: List[Dict]) -> Dict:
        """
        Comprehensive privacy compliance audit
        """
        audit_timestamp = datetime.now()
        
        # 1. Privacy budget analysis
        budget_analysis = self._analyze_privacy_budgets(round_metrics)
        
        # 2. Mechanism usage analysis
        mechanism_analysis = self._analyze_mechanism_usage(privacy_events)
        
        # 3. Device-level privacy analysis
        device_analysis = self._analyze_device_privacy(round_metrics, privacy_events)
        
        # 4. Temporal privacy analysis
        temporal_analysis = self._analyze_temporal_privacy(privacy_events)
        
        # 5. Compliance assessment
        compliance_status = self._assess_compliance(budget_analysis, mechanism_analysis)
        
        audit_report = {
            'audit_timestamp': audit_timestamp,
            'compliance_status': compliance_status,
            'budget_analysis': budget_analysis,
            'mechanism_analysis': mechanism_analysis,
            'device_analysis': device_analysis,
            'temporal_analysis': temporal_analysis,
            'recommendations': self._generate_recommendations(budget_analysis, compliance_status)
        }
        
        self.audit_results.append(audit_report)
        return audit_report
        
    def _analyze_privacy_budgets(self, round_metrics: List[RoundMetrics]) -> Dict:
        """Analyze privacy budget consumption patterns"""
        total_budget_consumed = {}
        budget_per_round = []
        
        for metrics in round_metrics:
            round_total = sum(metrics.privacy_budget_consumed.values())
            budget_per_round.append(round_total)
            
            for device_id, budget in metrics.privacy_budget_consumed.items():
                if device_id not in total_budget_consumed:
                    total_budget_consumed[device_id] = 0
                total_budget_consumed[device_id] += budget
                
        return {
            'total_budget_by_device': total_budget_consumed,
            'average_budget_per_round': np.mean(budget_per_round) if budget_per_round else 0,
            'max_budget_per_round': np.max(budget_per_round) if budget_per_round else 0,
            'budget_exhausted_devices': [
                device_id for device_id, budget in total_budget_consumed.items()
                if budget >= self.privacy_budget_limit
            ]
        }
        
    def _analyze_mechanism_usage(self, privacy_events: List[Dict]) -> Dict:
        """Analyze usage patterns of different privacy mechanisms"""
        mechanism_counts = defaultdict(int)
        mechanism_epsilon = defaultdict(list)
        
        for event in privacy_events:
            mechanism = event['mechanism']
            mechanism_counts[mechanism] += 1
            mechanism_epsilon[mechanism].append(event['epsilon'])
            
        return {
            'mechanism_usage_counts': dict(mechanism_counts),
            'mechanism_epsilon_stats': {
                mechanism: {
                    'mean': np.mean(epsilons),
                    'std': np.std(epsilons),
                    'total': np.sum(epsilons)
                }
                for mechanism, epsilons in mechanism_epsilon.items()
            }
        }
        
    def _analyze_device_privacy(self, round_metrics: List[RoundMetrics], 
                               privacy_events: List[Dict]) -> Dict:
        """Analyze per-device privacy patterns"""
        device_privacy_stats = {}
        
        # Get unique device IDs
        all_devices = set()
        for metrics in round_metrics:
            all_devices.update(metrics.participating_devices)
            
        for device_id in all_devices:
            device_events = [e for e in privacy_events if e['device_id'] == device_id]
            device_rounds = [m for m in round_metrics if device_id in m.participating_devices]
            
            total_epsilon = sum(e['epsilon'] for e in device_events)
            participation_rounds = len(device_rounds)
            
            device_privacy_stats[device_id] = {
                'total_epsilon_consumed': total_epsilon,
                'participation_rounds': participation_rounds,
                'average_epsilon_per_round': total_epsilon / participation_rounds if participation_rounds > 0 else 0,
                'privacy_mechanisms_used': list(set(e['mechanism'] for e in device_events))
            }
            
        return device_privacy_stats
        
    def _analyze_temporal_privacy(self, privacy_events: List[Dict]) -> Dict:
        """Analyze privacy consumption over time"""
        if not privacy_events:
            return {}
            
        # Sort events by timestamp
        sorted_events = sorted(privacy_events, key=lambda x: x['timestamp'])
        
        # Calculate cumulative privacy consumption
        cumulative_epsilon = 0
        epsilon_over_time = []
        
        for event in sorted_events:
            cumulative_epsilon += event['epsilon']
            epsilon_over_time.append({
                'timestamp': event['timestamp'],
                'cumulative_epsilon': cumulative_epsilon,
                'mechanism': event['mechanism']
            })
            
        return {
            'epsilon_over_time': epsilon_over_time,
            'total_epsilon_consumed': cumulative_epsilon,
            'privacy_consumption_rate': cumulative_epsilon / len(privacy_events)
        }
        
    def _assess_compliance(self, budget_analysis: Dict, mechanism_analysis: Dict) -> Dict:
        """Assess overall privacy compliance"""
        compliance_issues = []
        compliance_score = 100  # Start with perfect score
        
        # Check for budget exhaustion
        exhausted_devices = budget_analysis.get('budget_exhausted_devices', [])
        if exhausted_devices:
            compliance_issues.append(f"Privacy budget exhausted for {len(exhausted_devices)} device(s)")
            compliance_score -= 30
            
        # Check for excessive epsilon values
        mechanism_stats = mechanism_analysis.get('mechanism_epsilon_stats', {})
        for mechanism, stats in mechanism_stats.items():
            if stats['mean'] > 1.0:  # High epsilon values
                compliance_issues.append(f"High epsilon values detected for {mechanism}")
                compliance_score -= 10
                
        return {
            'compliance_score': max(0, compliance_score),
            'compliance_level': 'HIGH' if compliance_score >= 80 else 'MEDIUM' if compliance_score >= 60 else 'LOW',
            'compliance_issues': compliance_issues
        }
        
    def _generate_recommendations(self, budget_analysis: Dict, compliance_status: Dict) -> List[str]:
        """Generate privacy improvement recommendations"""
        recommendations = []
        
        if budget_analysis.get('budget_exhausted_devices'):
            recommendations.append("Consider excluding devices with exhausted privacy budgets")
            recommendations.append("Implement device rotation to distribute privacy consumption")
            
        if compliance_status['compliance_score'] < 70:
            recommendations.append("Reduce epsilon values for better privacy protection")
            recommendations.append("Implement more sophisticated privacy mechanisms")
            
        if budget_analysis.get('max_budget_per_round', 0) > 0.1:
            recommendations.append("Consider reducing privacy budget consumption per round")
            
        return recommendations

class FederatedLearningVisualizer:
    """
    Advanced visualization system for federated learning
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: str = "fl_plots"):
        self.save_plots = save_plots
        self.plot_dir = Path(plot_dir)
        if save_plots:
            self.plot_dir.mkdir(exist_ok=True)
            
    def create_training_dashboard(self, round_metrics: List[RoundMetrics]) -> go.Figure:
        """
        Create comprehensive training dashboard
        """
        if not round_metrics:
            return go.Figure()
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy Over Rounds', 'Device Participation',
                           'Privacy Budget Consumption', 'Round Duration'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract data
        rounds = [m.round_number for m in round_metrics]
        accuracies = [m.average_accuracy for m in round_metrics]
        durations = [m.round_duration for m in round_metrics]
        participation_counts = [len(m.participating_devices) for m in round_metrics]
        
        # Privacy budget data
        privacy_budgets = []
        for m in round_metrics:
            total_privacy = sum(m.privacy_budget_consumed.values())
            privacy_budgets.append(total_privacy)
        
        # 1. Accuracy over rounds
        fig.add_trace(
            go.Scatter(x=rounds, y=accuracies, mode='lines+markers',
                      name='Average Accuracy', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 2. Device participation
        fig.add_trace(
            go.Scatter(x=rounds, y=participation_counts, mode='lines+markers',
                      name='Participating Devices', line=dict(color='green')),
            row=1, col=2
        )
        
        # 3. Privacy budget consumption
        fig.add_trace(
            go.Scatter(x=rounds, y=privacy_budgets, mode='lines+markers',
                      name='Privacy Budget', line=dict(color='red')),
            row=2, col=1
        )
        
        # 4. Round duration
        fig.add_trace(
            go.Scatter(x=rounds, y=durations, mode='lines+markers',
                      name='Round Duration (s)', line=dict(color='orange')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Federated Learning Training Dashboard",
            showlegend=False,
            height=600
        )
        
        if self.save_plots:
            fig.write_html(self.plot_dir / "training_dashboard.html")
            
        return fig
        
    def create_device_analysis_plot(self, round_metrics: List[RoundMetrics]) -> go.Figure:
        """
        Create device-level analysis visualization
        """
        if not round_metrics:
            return go.Figure()
            
        # Collect device data
        device_data = defaultdict(lambda: {'accuracies': [], 'rounds': [], 'privacy_consumed': []})
        
        for metrics in round_metrics:
            for device_id, accuracy in metrics.individual_accuracies.items():
                device_data[device_id]['accuracies'].append(accuracy)
                device_data[device_id]['rounds'].append(metrics.round_number)
                device_data[device_id]['privacy_consumed'].append(
                    metrics.privacy_budget_consumed.get(device_id, 0)
                )
        
        # Create subplot for each device
        num_devices = len(device_data)
        fig = make_subplots(
            rows=(num_devices + 1) // 2, cols=2,
            subplot_titles=[f'Device {device_id}' for device_id in device_data.keys()]
        )
        
        colors = px.colors.qualitative.Set1
        
        row, col = 1, 1
        for i, (device_id, data) in enumerate(device_data.items()):
            color = colors[i % len(colors)]
            
            # Accuracy line
            fig.add_trace(
                go.Scatter(x=data['rounds'], y=data['accuracies'],
                          mode='lines+markers', name=f'{device_id} Accuracy',
                          line=dict(color=color)),
                row=row, col=col
            )
            
            # Move to next subplot position
            col += 1
            if col > 2:
                col = 1
                row += 1
                
        fig.update_layout(
            title_text="Device-Level Performance Analysis",
            height=300 * ((num_devices + 1) // 2),
            showlegend=False
        )
        
        if self.save_plots:
            fig.write_html(self.plot_dir / "device_analysis.html")
            
        return fig
        
    def create_security_analysis_plot(self, round_metrics: List[RoundMetrics]) -> go.Figure:
        """
        Create security analysis visualization
        """
        if not round_metrics:
            return go.Figure()
            
        rounds = [m.round_number for m in round_metrics]
        byzantine_counts = [len(m.byzantine_devices_detected) for m in round_metrics]
        security_event_counts = [len(m.security_events) for m in round_metrics]
        
        fig = go.Figure()
        
        # Byzantine devices detected
        fig.add_trace(
            go.Bar(x=rounds, y=byzantine_counts, name='Byzantine Devices Detected',
                   marker_color='red', opacity=0.7)
        )
        
        # Security events
        fig.add_trace(
            go.Scatter(x=rounds, y=security_event_counts, mode='lines+markers',
                      name='Security Events', line=dict(color='orange'),
                      yaxis='y2')
        )
        
        fig.update_layout(
            title="Security Analysis Over Rounds",
            xaxis_title="Round Number",
            yaxis=dict(title="Byzantine Devices", side="left"),
            yaxis2=dict(title="Security Events", side="right", overlaying="y"),
            legend=dict(x=0.01, y=0.99)
        )
        
        if self.save_plots:
            fig.write_html(self.plot_dir / "security_analysis.html")
            
        return fig
        
    def create_privacy_audit_plot(self, audit_results: List[Dict]) -> go.Figure:
        """
        Create privacy audit visualization
        """
        if not audit_results:
            return go.Figure()
            
        # Extract audit data
        timestamps = [r['audit_timestamp'] for r in audit_results]
        compliance_scores = [r['compliance_status']['compliance_score'] for r in audit_results]
        
        fig = go.Figure()
        
        # Compliance score over time
        fig.add_trace(
            go.Scatter(x=timestamps, y=compliance_scores, mode='lines+markers',
                      name='Compliance Score', line=dict(color='blue'))
        )
        
        # Add compliance thresholds
        fig.add_hline(y=80, line_dash="dash", line_color="green", 
                     annotation_text="High Compliance")
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Medium Compliance")
        
        fig.update_layout(
            title="Privacy Compliance Score Over Time",
            xaxis_title="Audit Timestamp",
            yaxis_title="Compliance Score",
            yaxis=dict(range=[0, 100])
        )
        
        if self.save_plots:
            fig.write_html(self.plot_dir / "privacy_audit.html")
            
        return fig

class ComprehensiveMonitor:
    """
    Main monitoring coordinator that brings all components together
    """
    
    def __init__(self, log_dir: str = "fl_logs", plot_dir: str = "fl_plots"):
        self.logger = FederatedLearningLogger(log_dir)
        self.privacy_auditor = PrivacyAuditor()
        self.visualizer = FederatedLearningVisualizer(save_plots=True, plot_dir=plot_dir)
        
    def log_round_completion(self, round_number: int, 
                           participating_devices: List[str],
                           individual_accuracies: Dict[str, float],
                           privacy_budget_consumed: Dict[str, float],
                           byzantine_devices: List[str],
                           communication_overhead: float,
                           round_duration: float,
                           security_events: Optional[List[Dict]] = None):
        """
        Comprehensive logging of round completion
        """
        if security_events is None:
            security_events = []
            
        metrics = RoundMetrics(
            round_number=round_number,
            timestamp=datetime.now(),
            participating_devices=participating_devices,
            average_accuracy=float(np.mean(list(individual_accuracies.values()))),
            individual_accuracies=individual_accuracies,
            privacy_budget_consumed=privacy_budget_consumed,
            byzantine_devices_detected=byzantine_devices,
            communication_overhead=communication_overhead,
            round_duration=round_duration,
            convergence_metric=0.0,  # Can be computed based on accuracy trends
            security_events=security_events
        )
        
        self.logger.log_round_completion(metrics)
        
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive monitoring report with all visualizations
        """
        # Generate privacy audit
        audit_report = self.privacy_auditor.audit_privacy_compliance(
            self.logger.round_metrics, self.logger.privacy_events
        )
        
        # Create visualizations
        training_dashboard = self.visualizer.create_training_dashboard(self.logger.round_metrics)
        device_analysis = self.visualizer.create_device_analysis_plot(self.logger.round_metrics)
        security_analysis = self.visualizer.create_security_analysis_plot(self.logger.round_metrics)
        privacy_audit_plot = self.visualizer.create_privacy_audit_plot(self.privacy_auditor.audit_results)
        
        # Export logs
        log_export_path = self.logger.export_logs('json')
        
        return {
            'report_timestamp': datetime.now(),
            'privacy_audit': audit_report,
            'log_export_path': log_export_path,
            'total_rounds': len(self.logger.round_metrics),
            'total_security_events': len(self.logger.security_events),
            'total_privacy_events': len(self.logger.privacy_events),
            'visualizations_created': [
                'training_dashboard.html',
                'device_analysis.html', 
                'security_analysis.html',
                'privacy_audit.html'
            ]
        }
