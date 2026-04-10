"""
Financial institution tab for the Climate Risk Integration Platform.
Displays climate risk exposure for financial institutions.
"""

import logging
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                            QSplitter, QTabWidget, QGridLayout, QScrollArea,
                            QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QSizePolicy, QGroupBox, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette, QBrush

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data_manager import DataManager
from core.risk_engine import RiskEngine


class FinancialInstitutionTab(QWidget):
    """
    Financial institution tab showing climate risk exposure for banks and funds
    with significant exposure to oil and gas companies
    """
    
    def __init__(self, data_manager: DataManager, risk_engine: RiskEngine):
        super().__init__()
        
        self.logger = logging.getLogger("financial_institution_tab")
        self.data_manager = data_manager
        self.risk_engine = risk_engine
        
        # Initialize state
        self.institution_id = None
        self.exposure_results = None
        self.optimization_results = None
        
        # Set up the UI
        self._init_ui()
        
        self.logger.info("FinancialInstitutionTab initialized")
    
    def _init_ui(self):
        """Initialize the financial institution UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header section
        header_layout = QHBoxLayout()
        
        # Institution info panel
        self.institution_info = QFrame()
        self.institution_info.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.institution_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        
        institution_layout = QVBoxLayout(self.institution_info)
        
        self.institution_name_label = QLabel("No Institution Selected")
        self.institution_name_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        institution_layout.addWidget(self.institution_name_label)
        
        self.institution_details_label = QLabel("")
        institution_layout.addWidget(self.institution_details_label)
        
        header_layout.addWidget(self.institution_info)
        
        # Action buttons
        action_layout = QVBoxLayout()
        
        self.analyze_button = QPushButton("Analyze Exposure")
        self.analyze_button.clicked.connect(self._analyze_exposure)
        action_layout.addWidget(self.analyze_button)
        
        self.optimize_button = QPushButton("Optimize Portfolio")
        self.optimize_button.clicked.connect(self._optimize_portfolio)
        self.optimize_button.setEnabled(False)  # Disabled until analysis is done
        action_layout.addWidget(self.optimize_button)
        
        header_layout.addLayout(action_layout)
        
        layout.addLayout(header_layout)
        
        # Main content in a splitter
        splitter = QSplitter(Qt.Vertical)
        
        # Upper section - Exposure Overview
        upper_widget = QWidget()
        upper_layout = QHBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel - Exposure metrics and charts
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        left_layout = QVBoxLayout(left_panel)
        
        # Exposure metrics group
        metrics_group = QGroupBox("Exposure Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        metrics_layout.addWidget(QLabel("Total O&G Exposure:"), 0, 0)
        self.total_exposure_label = QLabel("--")
        metrics_layout.addWidget(self.total_exposure_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Average Risk Score:"), 1, 0)
        self.risk_score_label = QLabel("--")
        metrics_layout.addWidget(self.risk_score_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Value at Risk:"), 2, 0)
        self.var_label = QLabel("--")
        metrics_layout.addWidget(self.var_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("Risk Concentration:"), 3, 0)
        self.concentration_label = QLabel("--")
        metrics_layout.addWidget(self.concentration_label, 3, 1)
        
        left_layout.addWidget(metrics_group)
        
        # Portfolio risk chart
        risk_chart_label = QLabel("Portfolio Risk by Scenario")
        risk_chart_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(risk_chart_label)
        
        self.risk_figure = Figure(figsize=(5, 4), dpi=100)
        self.risk_canvas = FigureCanvas(self.risk_figure)
        left_layout.addWidget(self.risk_canvas)
        
        upper_layout.addWidget(left_panel)
        
        # Right panel - Network visualization
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        right_layout = QVBoxLayout(right_panel)
        
        network_label = QLabel("Exposure Network")
        network_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(network_label)
        
        self.network_figure = Figure(figsize=(6, 5), dpi=100)
        self.network_canvas = FigureCanvas(self.network_figure)
        right_layout.addWidget(self.network_canvas)
        
        upper_layout.addWidget(right_panel)
        
        splitter.addWidget(upper_widget)
        
        # Lower section - Exposure Details and Optimization
        lower_widget = QTabWidget()
        
        # Exposure Details tab
        exposure_tab = QWidget()
        exposure_layout = QVBoxLayout(exposure_tab)
        
        # Company exposure table
        self.company_table = QTableWidget(0, 5)
        self.company_table.setHorizontalHeaderLabels(
            ["Company", "Exposure Amount", "Exposure Type", "Risk Score", "% of Portfolio"]
        )
        self.company_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.company_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.company_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.company_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.company_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.company_table.verticalHeader().setVisible(False)
        self.company_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        exposure_layout.addWidget(self.company_table)
        
        lower_widget.addTab(exposure_tab, "Exposure Details")
        
        # Portfolio Optimization tab
        optimization_tab = QWidget()
        optimization_layout = QVBoxLayout(optimization_tab)
        
        # Optimization controls
        controls_layout = QHBoxLayout()
        
        # Optimization goal selector
        goal_group = QGroupBox("Optimization Goal")
        goal_layout = QVBoxLayout(goal_group)
        
        self.goal_combo = QComboBox()
        self.goal_combo.addItems(["Risk Reduction", "Return Preservation", "Balanced"])
        goal_layout.addWidget(self.goal_combo)
        
        controls_layout.addWidget(goal_group)
        
        # Constraint level selector
        constraint_group = QGroupBox("Constraint Level")
        constraint_layout = QVBoxLayout(constraint_group)
        
        self.constraint_combo = QComboBox()
        self.constraint_combo.addItems(["Conservative", "Moderate", "Aggressive"])
        constraint_layout.addWidget(self.constraint_combo)
        
        controls_layout.addWidget(constraint_group)
        
        # Run optimization button
        self.run_button = QPushButton("Run Optimization")
        self.run_button.clicked.connect(self._run_optimization)
        self.run_button.setEnabled(False)  # Disabled until exposure analysis is done
        
        controls_layout.addWidget(self.run_button)
        controls_layout.addStretch(1)
        
        optimization_layout.addLayout(controls_layout)
        
        # Optimization results
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout(results_group)
        
        # Before/After metrics
        metrics_layout = QGridLayout()
        
        metrics_layout.addWidget(QLabel("Metric"), 0, 0)
        metrics_layout.addWidget(QLabel("Before"), 0, 1)
        metrics_layout.addWidget(QLabel("After"), 0, 2)
        metrics_layout.addWidget(QLabel("Change"), 0, 3)
        
        metrics_layout.addWidget(QLabel("Risk Score:"), 1, 0)
        self.before_risk_label = QLabel("--")
        metrics_layout.addWidget(self.before_risk_label, 1, 1)
        self.after_risk_label = QLabel("--")
        metrics_layout.addWidget(self.after_risk_label, 1, 2)
        self.change_risk_label = QLabel("--")
        metrics_layout.addWidget(self.change_risk_label, 1, 3)
        
        metrics_layout.addWidget(QLabel("Diversification:"), 2, 0)
        self.before_div_label = QLabel("--")
        metrics_layout.addWidget(self.before_div_label, 2, 1)
        self.after_div_label = QLabel("--")
        metrics_layout.addWidget(self.after_div_label, 2, 2)
        self.change_div_label = QLabel("--")
        metrics_layout.addWidget(self.change_div_label, 2, 3)
        
        metrics_layout.addWidget(QLabel("Value at Risk:"), 3, 0)
        self.before_var_label = QLabel("--")
        metrics_layout.addWidget(self.before_var_label, 3, 1)
        self.after_var_label = QLabel("--")
        metrics_layout.addWidget(self.after_var_label, 3, 2)
        self.change_var_label = QLabel("--")
        metrics_layout.addWidget(self.change_var_label, 3, 3)
        
        results_layout.addLayout(metrics_layout)
        
        # Optimization chart
        self.opt_figure = Figure(figsize=(6, 4), dpi=100)
        self.opt_canvas = FigureCanvas(self.opt_figure)
        results_layout.addWidget(self.opt_canvas)
        
        # Recommendations table
        recommendations_label = QLabel("Recommended Actions")
        recommendations_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        results_layout.addWidget(recommendations_label)
        
        self.recommendations_table = QTableWidget(0, 4)
        self.recommendations_table.setHorizontalHeaderLabels(
            ["Company", "Action", "Current Exposure", "Suggested Exposure"]
        )
        self.recommendations_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.recommendations_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.recommendations_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.recommendations_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.recommendations_table.verticalHeader().setVisible(False)
        
        results_layout.addWidget(self.recommendations_table)
        
        optimization_layout.addWidget(results_group)
        
        lower_widget.addTab(optimization_tab, "Portfolio Optimization")
        
        splitter.addWidget(lower_widget)
        
        # Set initial splitter sizes
        splitter.setSizes([int(self.height() * 0.4), int(self.height() * 0.6)])
        
        layout.addWidget(splitter)
        
        # Initialize charts
        self._initialize_charts()
    
    def _initialize_charts(self):
        """Initialize empty charts"""
        # Portfolio risk figure
        self.risk_figure.clear()
        risk_ax = self.risk_figure.add_subplot(111)
        risk_ax.set_title("Portfolio Risk by Scenario")
        risk_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=risk_ax.transAxes)
        self.risk_figure.tight_layout()
        self.risk_canvas.draw()
        
        # Network figure
        self.network_figure.clear()
        network_ax = self.network_figure.add_subplot(111)
        network_ax.set_title("Exposure Network")
        network_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=network_ax.transAxes)
        self.network_figure.tight_layout()
        self.network_canvas.draw()
        
        # Optimization figure
        self.opt_figure.clear()
        opt_ax = self.opt_figure.add_subplot(111)
        opt_ax.set_title("Exposure Optimization")
        opt_ax.text(0.5, 0.5, "Run optimization to see results", ha='center', va='center', transform=opt_ax.transAxes)
        self.opt_figure.tight_layout()
        self.opt_canvas.draw()
    
    def set_institution(self, institution_id: str):
        """Set the institution to display"""
        self.institution_id = institution_id
        
        # Clear previous results
        self.exposure_results = None
        self.optimization_results = None
        self._clear_ui()
        
        # Update institution name and enable analysis button
        if institution_id:
            self.institution_name_label.setText(f"Institution: {institution_id}")
            self.analyze_button.setEnabled(True)
            
            # In a real implementation, this would fetch institution details
            # For demo, we'll use placeholder data
            institution_types = {
                "JPM": "Bank", "BAC": "Bank", "WFC": "Bank", "C": "Bank",
                "GS": "Investment Bank", "MS": "Investment Bank",
                "BLK": "Asset Manager", "BK": "Bank", "STT": "Asset Manager",
                "PNC": "Bank"
            }
            
            inst_type = institution_types.get(institution_id, "Financial Institution")
            self.institution_details_label.setText(f"Type: {inst_type}")
        else:
            self.institution_name_label.setText("No Institution Selected")
            self.institution_details_label.setText("")
            self.analyze_button.setEnabled(False)
        
        self.logger.info(f"Set institution: {institution_id}")
    
    def update_exposure_results(self, results: Dict[str, Any]):
        """Update the tab with exposure calculation results"""
        if not results or "institution_id" not in results:
            return
        
        # Store results
        self.exposure_results = results
        institution_id = results["institution_id"]
        
        # Update UI if it matches current selected institution
        if institution_id == self.institution_id:
            self._update_ui_with_results()
    
    def update_optimization_results(self, results: Dict[str, Any]):
        """Update the tab with portfolio optimization results"""
        if not results or "institution_id" not in results:
            return
            
        # Store results
        self.optimization_results = results
        institution_id = results["institution_id"]
        
        # Update UI if it matches current selected institution
        if institution_id == self.institution_id:
            self._update_optimization_ui()
    
    def update_data(self, data_type: str, data: Any):
        """Update with new data"""
        # This method would update specific parts of the UI based on data updates
        pass
    
    def _clear_ui(self):
        """Clear UI elements"""
        # Clear metrics
        self.total_exposure_label.setText("--")
        self.risk_score_label.setText("--")
        self.var_label.setText("--")
        self.concentration_label.setText("--")
        
        # Clear tables
        self.company_table.setRowCount(0)
        self.recommendations_table.setRowCount(0)
        
        # Clear optimization metrics
        self.before_risk_label.setText("--")
        self.after_risk_label.setText("--")
        self.change_risk_label.setText("--")
        self.before_div_label.setText("--")
        self.after_div_label.setText("--")
        self.change_div_label.setText("--")
        self.before_var_label.setText("--")
        self.after_var_label.setText("--")
        self.change_var_label.setText("--")
        
        # Reset charts
        self._initialize_charts()
        
        # Disable buttons that require results
        self.optimize_button.setEnabled(False)
        self.run_button.setEnabled(False)
    
    def _update_ui_with_results(self):
        """Update UI with exposure results"""
        if not self.exposure_results:
            return
        
        # Enable optimization button
        self.optimize_button.setEnabled(True)
        self.run_button.setEnabled(True)
        
        # Extract key data
        exposure_summary = self.exposure_results.get("exposure_summary", {})
        company_exposures = self.exposure_results.get("company_exposures", [])
        
        # Update metrics
        self._update_exposure_metrics(exposure_summary)
        
        # Update tables
        self._update_company_table(company_exposures)
        
        # Update charts
        self._update_risk_chart(exposure_summary)
        self._update_network_chart(company_exposures, exposure_summary)
    
    def _update_exposure_metrics(self, exposure_summary: Dict[str, Any]):
        """Update the exposure metrics section"""
        # Total exposure
        total_exposure = exposure_summary.get("total_exposure", 0)
        if total_exposure >= 1e9:
            self.total_exposure_label.setText(f"${total_exposure/1e9:.1f}B")
        elif total_exposure >= 1e6:
            self.total_exposure_label.setText(f"${total_exposure/1e6:.1f}M")
        else:
            self.total_exposure_label.setText(f"${total_exposure:.0f}")
        
        # Risk score metrics
        scenario_metrics = exposure_summary.get("scenario_metrics", {})
        
        if scenario_metrics:
            # Get the average risk score across scenarios
            risk_scores = [metrics.get("weighted_risk_score", 0) for metrics in scenario_metrics.values()]
            avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
            
            self.risk_score_label.setText(f"{avg_risk:.1f} / 10")
            
            # Value at Risk
            var_values = [metrics.get("portfolio_var", 0) for metrics in scenario_metrics.values()]
            avg_var = sum(var_values) / len(var_values) if var_values else 0
            
            if avg_var >= 1e9:
                self.var_label.setText(f"${avg_var/1e9:.1f}B")
            elif avg_var >= 1e6:
                self.var_label.setText(f"${avg_var/1e6:.1f}M")
            else:
                self.var_label.setText(f"${avg_var:.0f}")
        
        # Risk concentration
        concentration = exposure_summary.get("risk_concentration", {})
        hhi = concentration.get("hhi", 0)
        top_5_percent = concentration.get("top_5_percent", 0)
        
        self.concentration_label.setText(f"HHI: {hhi:.2f} | Top 5: {top_5_percent*100:.1f}%")
    
    def _update_company_table(self, company_exposures: List[Dict[str, Any]]):
        """Update the company exposure table"""
        # Clear table
        self.company_table.setRowCount(0)
        
        if not company_exposures:
            return
        
        # Add rows for each company
        self.company_table.setRowCount(len(company_exposures))
        
        # Calculate total exposure for percentage calculation
        total_exposure = sum(company.get("exposure_amount", 0) for company in company_exposures)
        
        for i, company in enumerate(company_exposures):
            # Company name
            name = f"{company.get('company_id', '')} - {company.get('company_name', 'Unknown')}"
            name_item = QTableWidgetItem(name)
            self.company_table.setItem(i, 0, name_item)
            
            # Exposure amount
            amount = company.get("exposure_amount", 0)
            if amount >= 1e9:
                amount_text = f"${amount/1e9:.1f}B"
            elif amount >= 1e6:
                amount_text = f"${amount/1e6:.1f}M"
            else:
                amount_text = f"${amount:.0f}"
            
            amount_item = QTableWidgetItem(amount_text)
            amount_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.company_table.setItem(i, 1, amount_item)
            
            # Exposure type
            # This would come from real data - using placeholder
            exposure_types = ["Loan", "Equity", "Bond"]
            exposure_type = exposure_types[i % len(exposure_types)]
            type_item = QTableWidgetItem(exposure_type)
            self.company_table.setItem(i, 2, type_item)
            
            # Risk score
            risk_scores = company.get("risk_scores", {})
            score = 0
            
            if risk_scores:
                # Use first available scenario score
                for scenario, scenario_score in risk_scores.items():
                    score = scenario_score
                    break
            
            score_item = QTableWidgetItem(f"{score:.1f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            
            # Set color based on score
            if score >= 7:
                score_item.setBackground(QColor("#d9534f"))
                score_item.setForeground(QColor("white"))
            elif score >= 5:
                score_item.setBackground(QColor("#f0ad4e"))
            elif score >= 3:
                score_item.setBackground(QColor("#5bc0de"))
            else:
                score_item.setBackground(QColor("#5cb85c"))
                score_item.setForeground(QColor("white"))
            
            self.company_table.setItem(i, 3, score_item)
            
            # Portfolio percentage
            if total_exposure > 0:
                percent = amount / total_exposure * 100
            else:
                percent = 0
            
            percent_item = QTableWidgetItem(f"{percent:.1f}%")
            percent_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.company_table.setItem(i, 4, percent_item)
        
        # Resize rows to content
        self.company_table.resizeRowsToContents()
    
    def _update_risk_chart(self, exposure_summary: Dict[str, Any]):
        """Update the portfolio risk chart"""
        scenario_metrics = exposure_summary.get("scenario_metrics", {})
        
        if not scenario_metrics:
            return
        
        # Clear figure
        self.risk_figure.clear()
        risk_ax = self.risk_figure.add_subplot(111)
        
        # Scenario labels mapping
        scenario_names = {
            "ipcc_ssp119": "Low 1.5°C",
            "ipcc_ssp126": "Low-Med 2°C",
            "ipcc_ssp245": "Med 3°C",
            "ipcc_ssp370": "Med-High 4°C",
            "ipcc_ssp585": "High 5°C"
        }
        
        # Extract data for chart
        scenarios = []
        risk_scores = []
        var_percentages = []
        
        for scenario, metrics in scenario_metrics.items():
            scenario_label = scenario_names.get(scenario, scenario)
            scenarios.append(scenario_label)
            risk_scores.append(metrics.get("weighted_risk_score", 0))
            var_percentages.append(metrics.get("var_percentage", 0) * 100)  # Convert to percent
        
        # Sort by scenario intensity (assuming order matches the dict keys)
        scenario_order = list(scenario_names.values())
        sorted_data = sorted(zip(scenarios, risk_scores, var_percentages), 
                          key=lambda x: scenario_order.index(x[0]) if x[0] in scenario_order else 999)
        
        if sorted_data:
            scenarios, risk_scores, var_percentages = zip(*sorted_data)
        
            # Create two y-axes
            ax1 = risk_ax
            ax2 = ax1.twinx()
            
            # Set positions for grouped bar chart
            x = np.arange(len(scenarios))
            width = 0.35
            
            # Plot
            bars1 = ax1.bar(x - width/2, risk_scores, width, label='Risk Score', color='#5bc0de')
            bars2 = ax2.bar(x + width/2, var_percentages, width, label='VaR %', color='#f0ad4e')
            
            # Add data labels to bars
            for i, bar in enumerate(bars1):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{risk_scores[i]:.1f}", ha='center', va='bottom')
            
            for i, bar in enumerate(bars2):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{var_percentages[i]:.1f}%", ha='center', va='bottom')
            
            # Set labels and styling
            ax1.set_xlabel('Scenario')
            ax1.set_ylabel('Risk Score (0-10)')
            ax2.set_ylabel('Value at Risk (% of Exposure)')
            
            ax1.set_title('Portfolio Risk by Climate Scenario')
            ax1.set_xticks(x)
            ax1.set_xticklabels(scenarios, rotation=45, ha='right')
            
            # Set reasonable y-axis ranges
            ax1.set_ylim(0, max(risk_scores) * 1.2 if risk_scores else 10)
            ax2.set_ylim(0, max(var_percentages) * 1.2 if var_percentages else 20)
            
            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            # Add grid
            ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # Redraw
        self.risk_figure.tight_layout()
        self.risk_canvas.draw()
    
    def _update_network_chart(self, company_exposures: List[Dict[str, Any]], exposure_summary: Dict[str, Any]):
        """Update the exposure network chart"""
        if not company_exposures:
            return
        
        # Clear figure
        self.network_figure.clear()
        network_ax = self.network_figure.add_subplot(111)
        
        # This is a simplified network visualization - in a real implementation,
        # you would use a proper network analysis library like networkx
        
        # Extract data
        companies = []
        exposures = []
        risk_scores = []
        
        for company in company_exposures:
            companies.append(company.get("company_name", "Unknown"))
            exposures.append(company.get("exposure_amount", 0))
            
            # Get risk score
            company_risk_scores = company.get("risk_scores", {})
            avg_score = 0
            if company_risk_scores:
                scores = list(company_risk_scores.values())
                avg_score = sum(scores) / len(scores) if scores else 0
            
            risk_scores.append(avg_score)
        
        # Normalize exposures for node size
        max_exposure = max(exposures) if exposures else 1
        node_sizes = [100 + (exp / max_exposure) * 1000 for exp in exposures]
        
        # Create star-like layout (central institution with companies around)
        n_companies = len(companies)
        angles = np.linspace(0, 2*np.pi, n_companies, endpoint=False)
        
        # Calculate node positions
        node_x = [0]  # Institution at center
        node_y = [0]
        radius = 5
        for angle in angles:
            node_x.append(radius * np.cos(angle))
            node_y.append(radius * np.sin(angle))
        
        # Colors based on risk score
        node_colors = ['#5cb85c']  # Green for institution
        for score in risk_scores:
            if score >= 7:
                node_colors.append('#d9534f')  # Red
            elif score >= 5:
                node_colors.append('#f0ad4e')  # Orange
            elif score >= 3:
                node_colors.append('#5bc0de')  # Blue
            else:
                node_colors.append('#5cb85c')  # Green
        
        # Draw nodes
        network_ax.scatter(node_x, node_y, s=node_sizes, c=node_colors, alpha=0.7, edgecolors='black')
        
        # Draw edges (connections from institution to companies)
        for i in range(n_companies):
            network_ax.plot([0, node_x[i+1]], [0, node_y[i+1]], 'k-', alpha=0.3)
        
        # Add labels
        network_ax.text(0, 0, self.institution_id, ha='center', va='center', fontweight='bold')
        
        for i, company in enumerate(companies):
            network_ax.text(node_x[i+1], node_y[i+1], company, ha='center', va='center', fontsize=8)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#d9534f', label='High Risk (7-10)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#f0ad4e', label='Medium Risk (5-7)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#5bc0de', label='Low-Medium Risk (3-5)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#5cb85c', label='Low Risk (0-3)', markersize=10),
        ]
        network_ax.legend(handles=legend_elements, loc='upper right')
        
        # Remove axes
        network_ax.set_xticks([])
        network_ax.set_yticks([])
        network_ax.set_title('Exposure Network')
        
        # Set equal aspect ratio to prevent distortion
        network_ax.set_aspect('equal')
        
        # Redraw
        self.network_figure.tight_layout()
        self.network_canvas.draw()
    
    def _update_optimization_ui(self):
        """Update UI with optimization results"""
        if not self.optimization_results or not self.exposure_results:
            return
        
        # Check if optimization was successful
        if self.optimization_results.get("status") != "success":
            error_message = self.optimization_results.get("message", "Optimization failed")
            self.logger.warning(f"Optimization error: {error_message}")
            return
        
        # Extract optimization data
        optimization_result = self.optimization_results.get("optimization_result", {})
        recommendations = self.optimization_results.get("recommendations", [])
        
        # Update metrics
        self._update_optimization_metrics(optimization_result)
        
        # Update recommendations table
        self._update_recommendations_table(recommendations)
        
        # Update optimization chart
        self._update_optimization_chart(optimization_result)
    
    def _update_optimization_metrics(self, optimization_result: Dict[str, Any]):
        """Update the optimization metrics display"""
        metrics = optimization_result.get("metrics", {})
        
        # Risk score metrics
        risk_metrics = metrics.get("risk_score", {})
        before_risk = risk_metrics.get("initial", 0)
        after_risk = risk_metrics.get("optimized", 0)
        risk_reduction = risk_metrics.get("reduction", 0)
        
        self.before_risk_label.setText(f"{before_risk:.2f}")
        self.after_risk_label.setText(f"{after_risk:.2f}")
        self.change_risk_label.setText(f"{risk_reduction*100:.1f}%")
        
        # Diversification metrics
        div_metrics = metrics.get("diversification", {})
        before_div = div_metrics.get("initial_positions", 0)
        after_div = div_metrics.get("optimized_positions", 0)
        
        self.before_div_label.setText(f"{before_div}")
        self.after_div_label.setText(f"{after_div}")
        change_div = after_div - before_div
        self.change_div_label.setText(f"{change_div:+d}")
        
        # VaR metrics - using exposure_summary from exposure_results
        if self.exposure_results:
            exposure_summary = self.exposure_results.get("exposure_summary", {})
            scenario_metrics = exposure_summary.get("scenario_metrics", {})
            
            if scenario_metrics:
                # Get average VaR across scenarios
                var_values = [metrics.get("portfolio_var", 0) for metrics in scenario_metrics.values()]
                before_var = sum(var_values) / len(var_values) if var_values else 0
                
                # Estimate optimized VaR using risk reduction percentage
                after_var = before_var * (1 - risk_reduction)
                var_change = before_var - after_var
                
                # Format for display
                if before_var >= 1e9:
                    self.before_var_label.setText(f"${before_var/1e9:.1f}B")
                elif before_var >= 1e6:
                    self.before_var_label.setText(f"${before_var/1e6:.1f}M")
                else:
                    self.before_var_label.setText(f"${before_var:.0f}")
                
                if after_var >= 1e9:
                    self.after_var_label.setText(f"${after_var/1e9:.1f}B")
                elif after_var >= 1e6:
                    self.after_var_label.setText(f"${after_var/1e6:.1f}M")
                else:
                    self.after_var_label.setText(f"${after_var:.0f}")
                
                if var_change >= 1e9:
                    self.change_var_label.setText(f"-${var_change/1e9:.1f}B")
                elif var_change >= 1e6:
                    self.change_var_label.setText(f"-${var_change/1e6:.1f}M")
                else:
                    self.change_var_label.setText(f"-${var_change:.0f}")
    
    def _update_recommendations_table(self, recommendations: List[Dict[str, Any]]):
        """Update the recommendations table"""
        # Clear table
        self.recommendations_table.setRowCount(0)
        
        if not recommendations:
            return
        
        # Add rows for each recommendation
        self.recommendations_table.setRowCount(len(recommendations))
        
        for i, recommendation in enumerate(recommendations):
            # Company name
            company_name = recommendation.get("company_name", "Unknown")
            name_item = QTableWidgetItem(company_name)
            self.recommendations_table.setItem(i, 0, name_item)
            
            # Action
            action = recommendation.get("action", "")
            action_item = QTableWidgetItem(action.title())
            
            # Set color based on action
            if action == "reduce":
                action_item.setForeground(QColor("#d9534f"))  # Red
            elif action == "increase":
                action_item.setForeground(QColor("#5cb85c"))  # Green
            
            self.recommendations_table.setItem(i, 1, action_item)
            
            # Current exposure
            current_amount = recommendation.get("current_amount", 0)
            if current_amount >= 1e9:
                current_text = f"${current_amount/1e9:.1f}B"
            elif current_amount >= 1e6:
                current_text = f"${current_amount/1e6:.1f}M"
            else:
                current_text = f"${current_amount:.0f}"
            
            current_item = QTableWidgetItem(current_text)
            current_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.recommendations_table.setItem(i, 2, current_item)
            
            # Suggested exposure
            suggested_amount = recommendation.get("suggested_amount", 0)
            if suggested_amount >= 1e9:
                suggested_text = f"${suggested_amount/1e9:.1f}B"
            elif suggested_amount >= 1e6:
                suggested_text = f"${suggested_amount/1e6:.1f}M"
            else:
                suggested_text = f"${suggested_amount:.0f}"
            
            suggested_item = QTableWidgetItem(suggested_text)
            suggested_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.recommendations_table.setItem(i, 3, suggested_item)
        
        # Resize rows to content
        self.recommendations_table.resizeRowsToContents()
    
    def _update_optimization_chart(self, optimization_result: Dict[str, Any]):
        """Update the optimization chart"""
        optimized_exposure = optimization_result.get("optimized_exposure", [])
        
        if not optimized_exposure:
            return
        
        # Clear figure
        self.opt_figure.clear()
        
        # Prepare data for chart
        companies = []
        before_values = []
        after_values = []
        
        # Sort by current exposure (descending)
        sorted_exposures = sorted(optimized_exposure, key=lambda x: x.get("initial_weight", 0), reverse=True)
        
        # Take top 10 for readability
        top_exposures = sorted_exposures[:10]
        
        for exposure in top_exposures:
            companies.append(exposure.get("company_name", "Unknown"))
            before_values.append(exposure.get("initial_weight", 0) * 100)  # Convert to percentage
            after_values.append(exposure.get("optimized_weight", 0) * 100)  # Convert to percentage
        
        # Create horizontal bar chart
        ax = self.opt_figure.add_subplot(111)
        
        # Plot bars
        y_pos = np.arange(len(companies))
        width = 0.35
        
        ax.barh(y_pos - width/2, before_values, width, label='Before', color='#5bc0de')
        ax.barh(y_pos + width/2, after_values, width, label='After', color='#5cb85c')
        
        # Add labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(companies)
        ax.set_xlabel('Portfolio Weight (%)')
        ax.set_title('Before vs. After Optimization')
        ax.legend()
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6, axis='x')
        
        # Redraw
        self.opt_figure.tight_layout()
        self.opt_canvas.draw()
    
    @pyqtSlot()
    def _analyze_exposure(self):
        """Analyze exposure for the selected institution"""
        if not self.institution_id:
            return
        
        self.logger.info(f"Analyzing exposure for {self.institution_id}")
        
        # Clear previous results
        self.exposure_results = None
        self._clear_ui()
        
        # Update status display
        self.total_exposure_label.setText("Calculating...")
        
        # Call risk engine to calculate exposure
        self.risk_engine.calculate_financial_institution_exposure(
            self.institution_id,
            scenarios=["ipcc_ssp245", "ipcc_ssp370", "ipcc_ssp585"]
        )
    
    @pyqtSlot()
    def _optimize_portfolio(self):
        """Optimize portfolio for the selected institution"""
        if not self.institution_id or not self.exposure_results:
            return
        
        self.logger.info(f"Optimizing portfolio for {self.institution_id}")
        
        # Show optimization tab
        parent_tab_widget = self.parent()
        if isinstance(parent_tab_widget, QTabWidget):
            tab_index = parent_tab_widget.indexOf(self)
            if tab_index >= 0:
                parent_tab_widget.setCurrentIndex(tab_index)
        
        # Find the optimization tab
        for i in range(self.children()[-1].count()):
            if self.children()[-1].tabText(i) == "Portfolio Optimization":
                self.children()[-1].setCurrentIndex(i)
                break
    
    @pyqtSlot()
    def _run_optimization(self):
        """Run the optimization process"""
        if not self.institution_id or not self.exposure_results:
            return
        
        self.logger.info(f"Running optimization for {self.institution_id}")
        
        # Get optimization parameters
        goal_text = self.goal_combo.currentText().lower().replace(" ", "_")
        constraint_text = self.constraint_combo.currentText().lower()
        
        # Update button state
        self.run_button.setText("Optimizing...")
        self.run_button.setEnabled(False)
        
        # Call risk engine to optimize portfolio
        self.risk_engine.optimize_portfolio(
            self.institution_id,
            optimization_goal=goal_text,
            constraint_level=constraint_text
        )
        
        # Button will be re-enabled when results are received