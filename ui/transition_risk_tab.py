"""
Transition risk tab for the Climate Risk Integration Platform.
Displays transition climate risks for companies.
"""

import logging
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                            QSplitter, QTabWidget, QGridLayout, QScrollArea,
                            QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QSizePolicy, QGroupBox, QSlider)
from PyQt5.QtCore import Qt, QSize, pyqtSlot, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QBrush

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data_manager import DataManager
from core.risk_engine import RiskEngine


class TransitionRiskTab(QWidget):
    """Transition risk tab showing climate policy and market risks for companies"""
    
    def __init__(self, data_manager: DataManager, risk_engine: RiskEngine):
        super().__init__()
        
        self.logger = logging.getLogger("transition_risk_tab")
        self.data_manager = data_manager
        self.risk_engine = risk_engine
        
        # Initialize state
        self.company_id = None
        self.scenario = None
        self.risk_results = None
        self.include_physical = True
        self.include_transition = True
        
        # Set up the UI
        self._init_ui()
        
        self.logger.info("TransitionRiskTab initialized")
    
    def _init_ui(self):
        """Initialize the transition risk UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header section with controls
        header_layout = QHBoxLayout()
        
        # Risk driver selector
        risk_driver_group = QGroupBox("Risk Driver")
        risk_driver_layout = QHBoxLayout(risk_driver_group)
        
        self.risk_driver_combo = QComboBox()
        self.risk_driver_combo.addItems(["All Drivers", "Carbon Price", "Energy Demand", "Policy Timeline"])
        self.risk_driver_combo.currentIndexChanged.connect(self._risk_driver_changed)
        risk_driver_layout.addWidget(self.risk_driver_combo)
        
        header_layout.addWidget(risk_driver_group)
        
        # Time horizon selector
        time_horizon_group = QGroupBox("Time Horizon")
        time_horizon_layout = QHBoxLayout(time_horizon_group)
        
        self.time_horizon_combo = QComboBox()
        self.time_horizon_combo.addItems(["2030", "2050", "2100"])
        self.time_horizon_combo.currentIndexChanged.connect(self._time_horizon_changed)
        time_horizon_layout.addWidget(self.time_horizon_combo)
        
        header_layout.addWidget(time_horizon_group)
        
        # Carbon price slider
        carbon_price_group = QGroupBox("Carbon Price Simulation")
        carbon_price_layout = QVBoxLayout(carbon_price_group)
        
        self.carbon_price_slider = QSlider(Qt.Horizontal)
        self.carbon_price_slider.setMinimum(0)
        self.carbon_price_slider.setMaximum(200)
        self.carbon_price_slider.setValue(50)
        self.carbon_price_slider.setTickInterval(25)
        self.carbon_price_slider.setTickPosition(QSlider.TicksBelow)
        self.carbon_price_slider.valueChanged.connect(self._carbon_price_changed)
        
        slider_labels_layout = QHBoxLayout()
        slider_labels_layout.addWidget(QLabel("$0"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("$100"))
        slider_labels_layout.addStretch()
        slider_labels_layout.addWidget(QLabel("$200"))
        
        carbon_price_layout.addWidget(self.carbon_price_slider)
        carbon_price_layout.addLayout(slider_labels_layout)
        
        header_layout.addWidget(carbon_price_group)
        
        # Add spacer
        header_layout.addStretch(1)
        
        # Overall risk score display
        score_group = QGroupBox("Transition Risk Score")
        score_layout = QVBoxLayout(score_group)
        
        self.risk_score_label = QLabel("--")
        self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #777;")
        self.risk_score_label.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(self.risk_score_label)
        
        header_layout.addWidget(score_group)
        
        layout.addLayout(header_layout)
        
        # Main content in a splitter
        self.main_splitter = QSplitter(Qt.Vertical)
        
        # Upper section - Policy and Market Drivers
        upper_widget = QWidget()
        upper_layout = QHBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        
        # Policy drivers - left side
        policy_group = QGroupBox("Policy Drivers")
        policy_layout = QVBoxLayout(policy_group)
        
        # Carbon price chart
        self.carbon_figure = Figure(figsize=(5, 4), dpi=100)
        self.carbon_canvas = FigureCanvas(self.carbon_figure)
        policy_layout.addWidget(self.carbon_canvas)
        
        upper_layout.addWidget(policy_group, 1)
        
        # Market drivers - right side
        market_group = QGroupBox("Market Drivers")
        market_layout = QVBoxLayout(market_group)
        
        # Demand projection chart
        self.demand_figure = Figure(figsize=(5, 4), dpi=100)
        self.demand_canvas = FigureCanvas(self.demand_figure)
        market_layout.addWidget(self.demand_canvas)
        
        upper_layout.addWidget(market_group, 1)
        
        self.main_splitter.addWidget(upper_widget)
        
        # Lower section - Financial Impacts
        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        
        # Financial impacts in tabs
        self.financial_tabs = QTabWidget()
        
        # NPV Impact tab
        npv_tab = QWidget()
        npv_layout = QVBoxLayout(npv_tab)
        
        self.npv_figure = Figure(figsize=(6, 4), dpi=100)
        self.npv_canvas = FigureCanvas(self.npv_figure)
        npv_layout.addWidget(self.npv_canvas)
        
        self.financial_tabs.addTab(npv_tab, "NPV Impact")
        
        # Cash Flow Impact tab
        cashflow_tab = QWidget()
        cashflow_layout = QVBoxLayout(cashflow_tab)
        
        self.cashflow_figure = Figure(figsize=(6, 4), dpi=100)
        self.cashflow_canvas = FigureCanvas(self.cashflow_figure)
        cashflow_layout.addWidget(self.cashflow_canvas)
        
        self.financial_tabs.addTab(cashflow_tab, "Cash Flow Impact")
        
        # Balance Sheet Impact tab
        balance_tab = QWidget()
        balance_layout = QVBoxLayout(balance_tab)
        
        # Balance sheet impact table
        self.balance_table = QTableWidget(5, 3)
        self.balance_table.setHorizontalHeaderLabels(["Category", "Current Value", "Projected Impact"])
        self.balance_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.balance_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.balance_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.balance_table.verticalHeader().setVisible(False)
        
        balance_layout.addWidget(self.balance_table)
        
        self.financial_tabs.addTab(balance_tab, "Balance Sheet Impact")
        
        # Add financial tabs to lower layout
        lower_layout.addWidget(self.financial_tabs)
        
        # Risk details section
        risk_details_group = QGroupBox("Transition Risk Breakdown")
        risk_details_layout = QHBoxLayout(risk_details_group)
        
        # Risk metrics grid - left side
        metrics_layout = QGridLayout()
        metrics_layout.addWidget(QLabel("Carbon Price Risk:"), 0, 0)
        self.carbon_risk_label = QLabel("--")
        metrics_layout.addWidget(self.carbon_risk_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Market Demand Risk:"), 1, 0)
        self.market_risk_label = QLabel("--")
        metrics_layout.addWidget(self.market_risk_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("Policy Timeline Risk:"), 2, 0)
        self.policy_risk_label = QLabel("--")
        metrics_layout.addWidget(self.policy_risk_label, 2, 1)
        
        metrics_layout.addWidget(QLabel("Stranded Asset Risk:"), 3, 0)
        self.stranded_risk_label = QLabel("--")
        metrics_layout.addWidget(self.stranded_risk_label, 3, 1)
        
        left_widget = QWidget()
        left_widget.setLayout(metrics_layout)
        
        risk_details_layout.addWidget(left_widget)
        
        # Risk breakdown chart - right side
        self.breakdown_figure = Figure(figsize=(4, 3), dpi=100)
        self.breakdown_canvas = FigureCanvas(self.breakdown_figure)
        risk_details_layout.addWidget(self.breakdown_canvas)
        
        lower_layout.addWidget(risk_details_group)
        
        self.main_splitter.addWidget(lower_widget)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([int(self.height() * 0.5), int(self.height() * 0.5)])
        
        layout.addWidget(self.main_splitter)
        
        # Initialize charts
        self._initialize_charts()
    
    def _initialize_charts(self):
        """Initialize empty charts"""
        # Carbon price figure
        self.carbon_figure.clear()
        carbon_ax = self.carbon_figure.add_subplot(111)
        carbon_ax.set_title("Carbon Price Projections")
        carbon_ax.set_xlabel("Year")
        carbon_ax.set_ylabel("Carbon Price ($/tCO2)")
        carbon_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=carbon_ax.transAxes)
        self.carbon_figure.tight_layout()
        self.carbon_canvas.draw()
        
        # Demand projection figure
        self.demand_figure.clear()
        demand_ax = self.demand_figure.add_subplot(111)
        demand_ax.set_title("Energy Demand Projections")
        demand_ax.set_xlabel("Year")
        demand_ax.set_ylabel("Demand (% of 2020 level)")
        demand_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=demand_ax.transAxes)
        self.demand_figure.tight_layout()
        self.demand_canvas.draw()
        
        # NPV impact figure
        self.npv_figure.clear()
        npv_ax = self.npv_figure.add_subplot(111)
        npv_ax.set_title("NPV Impact of Transition Risks")
        npv_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=npv_ax.transAxes)
        self.npv_figure.tight_layout()
        self.npv_canvas.draw()
        
        # Cash flow impact figure
        self.cashflow_figure.clear()
        cashflow_ax = self.cashflow_figure.add_subplot(111)
        cashflow_ax.set_title("Annual Cash Flow Impact")
        cashflow_ax.set_xlabel("Year")
        cashflow_ax.set_ylabel("Impact ($M)")
        cashflow_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=cashflow_ax.transAxes)
        self.cashflow_figure.tight_layout()
        self.cashflow_canvas.draw()
        
        # Risk breakdown figure
        self.breakdown_figure.clear()
        breakdown_ax = self.breakdown_figure.add_subplot(111)
        breakdown_ax.set_title("Risk Driver Breakdown")
        breakdown_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=breakdown_ax.transAxes)
        self.breakdown_figure.tight_layout()
        self.breakdown_canvas.draw()
        
        # Initialize balance sheet table with default rows
        row_labels = ["Asset Valuation", "Revenue", "Operating Costs", "Capital Expenditure", "Debt Metrics"]
        for i, label in enumerate(row_labels):
            self.balance_table.setItem(i, 0, QTableWidgetItem(label))
            self.balance_table.setItem(i, 1, QTableWidgetItem("--"))
            self.balance_table.setItem(i, 2, QTableWidgetItem("--"))
    
    def set_company(self, company_id: str):
        """Set the company to display"""
        self.company_id = company_id
        
        # Clear previous results
        self.risk_results = None
        self._clear_ui()
        
        self.logger.info(f"Set company in transition risk tab: {company_id}")
    
    def set_scenario(self, scenario: str):
        """Set the scenario to display"""
        self.scenario = scenario
        
        self.logger.info(f"Set scenario in transition risk tab: {scenario}")
        
        # Update the UI if we have results
        if self.risk_results:
            self._update_ui_with_results()
    
    def update_risk_results(self, results: Dict[str, Any]):
        """Update the tab with risk calculation results"""
        if not results or "company_id" not in results:
            return
        
        # Store results
        self.risk_results = results
        company_id = results["company_id"]
        
        # Update UI if it matches current selected company
        if company_id == self.company_id:
            self._update_ui_with_results()
    
    def update_data(self, data_type: str, data: Any):
        """Update with new data"""
        # This method would update specific parts of the UI based on data updates
        pass
    
    def _clear_ui(self):
        """Clear UI elements"""
        # Clear risk score
        self.risk_score_label.setText("--")
        self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #777;")
        
        # Clear risk metrics
        self.carbon_risk_label.setText("--")
        self.market_risk_label.setText("--")
        self.policy_risk_label.setText("--")
        self.stranded_risk_label.setText("--")
        
        # Reset charts
        self._initialize_charts()
    
    def _update_ui_with_results(self):
        """Update UI with current risk results"""
        # Check if we have transition risk results
        if not self.risk_results or "transition_risks" not in self.risk_results:
            return
        
        transition_risks = self.risk_results["transition_risks"]
        
        # If specific scenario selected, show only that one
        if self.scenario and self.scenario in transition_risks:
            self._show_scenario_results(self.scenario, transition_risks[self.scenario])
        else:
            # Find the first available scenario
            if transition_risks:
                scenario = next(iter(transition_risks))
                self._show_scenario_results(scenario, transition_risks[scenario])
    
    def _show_scenario_results(self, scenario: str, risk_data: Dict[str, Any]):
        """Show results for a specific scenario"""
        # Update risk score
        overall_score = risk_data.get("overall_score", 0)
        self.risk_score_label.setText(f"{overall_score:.1f}")
        
        # Set color based on score
        if overall_score >= 7:
            self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #d9534f;")
        elif overall_score >= 5:
            self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #f0ad4e;")
        elif overall_score >= 3:
            self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #5bc0de;")
        else:
            self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #5cb85c;")
        
        # Extract risk drivers
        risk_drivers = risk_data.get("risk_drivers", {})
        financial_impact = risk_data.get("financial_impact", {})
        monte_carlo = risk_data.get("monte_carlo_results", {})
        
        # Update risk metrics
        self._update_risk_metrics(risk_data)
        
        # Update charts
        self._update_carbon_chart(risk_drivers.get("carbon_price", {}))
        self._update_demand_chart(risk_drivers.get("demand_changes", {}))
        self._update_npv_chart(monte_carlo)
        self._update_cashflow_chart(monte_carlo.get("yearly_impacts", []))
        self._update_breakdown_chart(risk_data)
        
        # Update balance sheet table
        self._update_balance_sheet(financial_impact, monte_carlo)
    
    def _update_risk_metrics(self, risk_data: Dict[str, Any]):
        """Update the risk metrics display"""
        # Extract risk components
        risk_drivers = risk_data.get("risk_drivers", {})
        financial_impact = risk_data.get("financial_impact", {})
        
        # Carbon price risk
        carbon_prices = risk_drivers.get("carbon_price", {})
        carbon_price_2050 = carbon_prices.get("2050", 0)
        # Scale to 0-10 risk score
        carbon_risk = min(10, carbon_price_2050 / 20)
        self.carbon_risk_label.setText(f"{carbon_risk:.1f} / 10")
        
        # Market demand risk
        demand_changes = risk_drivers.get("demand_changes", {})
        demand_2050 = demand_changes.get("2050", 1.0)
        # Convert to risk score (lower demand = higher risk)
        market_risk = min(10, (1 - demand_2050) * 10)
        self.market_risk_label.setText(f"{market_risk:.1f} / 10")
        
        # Policy timeline risk - compute from other factors
        # Implementation acceleration/delay effect
        transition_speed = risk_data.get("overall_score", 5) / 5 # normalize to 0-2 range
        policy_risk = min(10, transition_speed * 5)
        self.policy_risk_label.setText(f"{policy_risk:.1f} / 10")
        
        # Stranded asset risk
        stranded_asset_value = financial_impact.get("stranded_asset_risk", 0)
        if stranded_asset_value >= 1e9:
            stranded_text = f"${stranded_asset_value/1e9:.1f}B"
        elif stranded_asset_value >= 1e6:
            stranded_text = f"${stranded_asset_value/1e6:.1f}M"
        else:
            stranded_text = f"${stranded_asset_value:.0f}"
        
        self.stranded_risk_label.setText(stranded_text)
    
    def _update_carbon_chart(self, carbon_prices: Dict[str, float]):
        """Update the carbon price projection chart"""
        if not carbon_prices:
            return
        
        # Clear figure
        self.carbon_figure.clear()
        carbon_ax = self.carbon_figure.add_subplot(111)
        
        # Extract years and prices
        years = []
        prices = []
        
        for year, price in carbon_prices.items():
            years.append(int(year))
            prices.append(price)
        
        # Sort by year
        year_prices = sorted(zip(years, prices))
        years, prices = zip(*year_prices) if year_prices else ([], [])
        
        # Plot line
        carbon_ax.plot(years, prices, 'b-o', linewidth=2)
        
        # Add user-adjusted price line if significantly different
        user_price = self.carbon_price_slider.value()
        if abs(user_price - prices[-1]) > 10:  # Only show if significantly different
            # Create a user-adjusted projection
            user_prices = [p * (user_price / prices[-1]) for p in prices]
            carbon_ax.plot(years, user_prices, 'r--o', linewidth=2, alpha=0.7, label=f"Custom (${user_price}/tCO2)")
            carbon_ax.legend()
        
        # Add labels and styling
        carbon_ax.set_xlabel("Year")
        carbon_ax.set_ylabel("Carbon Price ($/tCO2)")
        carbon_ax.set_title("Carbon Price Projections")
        carbon_ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set reasonable y-axis range
        carbon_ax.set_ylim(0, max(prices) * 1.2)
        
        # Redraw
        self.carbon_figure.tight_layout()
        self.carbon_canvas.draw()
    
    def _update_demand_chart(self, demand_changes: Dict[str, float]):
        """Update the energy demand projection chart"""
        if not demand_changes:
            return
        
        # Clear figure
        self.demand_figure.clear()
        demand_ax = self.demand_figure.add_subplot(111)
        
        # Extract years and demand factors
        years = []
        factors = []
        
        for year, factor in demand_changes.items():
            years.append(int(year))
            factors.append(factor * 100)  # Convert to percentage
        
        # Sort by year
        year_factors = sorted(zip(years, factors))
        years, factors = zip(*year_factors) if year_factors else ([], [])
        
        # Add reference line at 100%
        demand_ax.axhline(y=100, color='k', linestyle='--', alpha=0.3)
        
        # Plot line
        demand_ax.plot(years, factors, 'g-o', linewidth=2)
        
        # Add area fill below line
        demand_ax.fill_between(years, 0, factors, alpha=0.2, color='green')
        
        # Add labels and styling
        demand_ax.set_xlabel("Year")
        demand_ax.set_ylabel("Demand (% of 2020 level)")
        demand_ax.set_title("Oil & Gas Demand Projections")
        demand_ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set reasonable y-axis range
        min_y = min(min(factors) * 0.9, 90)
        max_y = max(max(factors) * 1.1, 110)
        demand_ax.set_ylim(min_y, max_y)
        
        # Redraw
        self.demand_figure.tight_layout()
        self.demand_canvas.draw()
    
    def _update_npv_chart(self, monte_carlo: Dict[str, Any]):
        """Update the NPV impact chart"""
        if not monte_carlo:
            return
        
        # Extract NPV impact data
        mean_impact = monte_carlo.get("npv_impact_mean", 0)
        ci_lower, ci_upper = monte_carlo.get("npv_impact_95ci", [0, 0])
        impact_pct = monte_carlo.get("impact_percentage", 0)
        
        # Clear figure
        self.npv_figure.clear()
        
        # Create figure with two subplots
        gs = self.npv_figure.add_gridspec(1, 2, width_ratios=[3, 1])
        
        # NPV impact chart
        npv_ax = self.npv_figure.add_subplot(gs[0, 0])
        
        # Create a bar with error bars
        x = ['NPV Impact']
        
        # Determine color based on impact direction
        color = 'r' if mean_impact < 0 else 'g'
        
        npv_ax.bar(x, [mean_impact / 1e9], color=color, alpha=0.7)
        npv_ax.errorbar(x, [mean_impact / 1e9], yerr=[[abs(mean_impact - ci_lower) / 1e9], [abs(ci_upper - mean_impact) / 1e9]],
                     fmt='none', capsize=10, color='k')
        
        # Add value labels
        npv_ax.text(0, mean_impact / 1e9, f"${mean_impact/1e9:.1f}B", ha='center', va='bottom' if mean_impact >= 0 else 'top')
        
        # Add zero line
        npv_ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add labels and styling
        npv_ax.set_ylabel("NPV Impact ($ Billions)")
        npv_ax.set_title("NPV Impact of Transition Risks")
        npv_ax.grid(True, linestyle='--', alpha=0.6, axis='y')
        
        # Set y-axis limits with some padding
        max_abs = max(abs(ci_lower), abs(ci_upper)) / 1e9
        npv_ax.set_ylim(-max_abs * 1.2 if mean_impact < 0 else -max_abs * 0.2, 
                      max_abs * 0.2 if mean_impact < 0 else max_abs * 1.2)
        
        # Impact percentage chart (pie chart)
        pie_ax = self.npv_figure.add_subplot(gs[0, 1])
        
        # Convert to absolute percentage for visualization
        abs_impact_pct = abs(impact_pct) * 100
        remaining_pct = 100 - abs_impact_pct
        
        # Create pie chart
        colors = ['r' if mean_impact < 0 else 'g', '#f8f9fa']
        pie_ax.pie([abs_impact_pct, remaining_pct], colors=colors, shadow=False, startangle=90,
                wedgeprops={'alpha': 0.8})
        
        # Add percentage in center
        pie_ax.text(0, 0, f"{abs_impact_pct:.1f}%", ha='center', va='center', fontsize=14, fontweight='bold')
        
        pie_ax.set_title("% of Company Value")
        
        # Redraw
        self.npv_figure.tight_layout()
        self.npv_canvas.draw()
    
    def _update_cashflow_chart(self, yearly_impacts: List[Dict[str, float]]):
        """Update the cash flow impact chart"""
        if not yearly_impacts:
            return
        
        # Clear figure
        self.cashflow_figure.clear()
        cashflow_ax = self.cashflow_figure.add_subplot(111)
        
        # Extract data
        years = []
        means = []
        lower_bounds = []
        upper_bounds = []
        
        for impact in yearly_impacts:
            years.append(impact.get("year", 0))
            means.append(impact.get("mean_impact", 0) / 1e6)  # Convert to millions
            lower_bounds.append(impact.get("ci_lower", 0) / 1e6)
            upper_bounds.append(impact.get("ci_upper", 0) / 1e6)
        
        # Plot line for mean impact
        cashflow_ax.plot(years, means, 'b-', linewidth=2, label='Expected Impact')
        
        # Add confidence interval as shaded area
        cashflow_ax.fill_between(years, lower_bounds, upper_bounds, alpha=0.2, color='blue', label='95% Confidence Interval')
        
        # Add zero line
        cashflow_ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add labels and styling
        cashflow_ax.set_xlabel("Year")
        cashflow_ax.set_ylabel("Cash Flow Impact ($ Millions)")
        cashflow_ax.set_title("Annual Cash Flow Impact")
        cashflow_ax.grid(True, linestyle='--', alpha=0.6)
        cashflow_ax.legend()
        
        # Set reasonable y-axis range
        min_y = min(min(lower_bounds), 0) * 1.2
        max_y = max(max(upper_bounds), 0) * 1.2
        cashflow_ax.set_ylim(min_y, max_y)
        
        # Redraw
        self.cashflow_figure.tight_layout()
        self.cashflow_canvas.draw()
    
    def _update_breakdown_chart(self, risk_data: Dict[str, Any]):
        """Update the risk breakdown chart"""
        # Calculate risk components based on available data
        risk_drivers = risk_data.get("risk_drivers", {})
        financial_impact = risk_data.get("financial_impact", {})
        
        # Extract carbon price risk
        carbon_prices = risk_drivers.get("carbon_price", {})
        carbon_price_2050 = carbon_prices.get("2050", 0)
        carbon_risk = min(10, carbon_price_2050 / 20)
        
        # Extract market demand risk
        demand_changes = risk_drivers.get("demand_changes", {})
        demand_2050 = demand_changes.get("2050", 1.0)
        market_risk = min(10, (1 - demand_2050) * 10)
        
        # Policy timeline risk
        transition_speed = risk_data.get("overall_score", 5) / 5  # Normalize to 0-2 range
        policy_risk = min(10, transition_speed * 5)
        
        # Stranded asset risk
        stranded_asset_value = financial_impact.get("stranded_asset_risk", 0)
        company_value = 1e10  # Placeholder assumption
        stranded_risk = min(10, (stranded_asset_value / company_value) * 50)
        
        # Clear figure
        self.breakdown_figure.clear()
        breakdown_ax = self.breakdown_figure.add_subplot(111)
        
        # Create data for radar chart
        categories = ['Carbon Price', 'Market Demand', 'Policy Timeline', 'Stranded Assets']
        values = [carbon_risk, market_risk, policy_risk, stranded_risk]
        
        # Number of variables
        N = len(categories)
        
        # Angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Values need to be repeated to close the loop
        values += values[:1]
        
        # Plot radar chart
        breakdown_ax.plot(angles, values, 'b-', linewidth=2)
        breakdown_ax.fill(angles, values, 'b', alpha=0.2)
        
        # Add category labels
        breakdown_ax.set_xticks(angles[:-1])
        breakdown_ax.set_xticklabels(categories)
        
        # Set y ticks and limit
        breakdown_ax.set_yticks([2, 4, 6, 8, 10])
        breakdown_ax.set_ylim(0, 10)
        
        # Add title
        breakdown_ax.set_title("Risk Driver Breakdown")
        
        # Redraw
        self.breakdown_figure.tight_layout()
        self.breakdown_canvas.draw()
    
    def _update_balance_sheet(self, financial_impact: Dict[str, float], monte_carlo: Dict[str, Any]):
        """Update the balance sheet impact table"""
        # Extract financial impact data
        stranded_asset_risk = financial_impact.get("stranded_asset_risk", 0)
        revenue_impact = financial_impact.get("revenue_impact", 0)
        carbon_cost_impact = financial_impact.get("carbon_cost_impact", 0)
        
        # Extract NPV impact
        npv_impact = monte_carlo.get("npv_impact_mean", 0)
        
        # Generate placeholder data for demonstration
        # In a real implementation, this would use actual financial data
        
        # Row 0: Asset Valuation
        self.balance_table.setItem(0, 1, QTableWidgetItem("$10.0B"))  # Current value
        
        asset_impact = -stranded_asset_risk / 1e9
        asset_impact_text = f"${asset_impact:.1f}B" if asset_impact < 0 else f"+${asset_impact:.1f}B"
        asset_item = QTableWidgetItem(asset_impact_text)
        if asset_impact < 0:
            asset_item.setForeground(QColor("red"))
        self.balance_table.setItem(0, 2, asset_item)
        
        # Row 1: Revenue
        self.balance_table.setItem(1, 1, QTableWidgetItem("$5.2B"))  # Current value
        
        revenue_impact_annual = revenue_impact / 1e9
        revenue_impact_text = f"${revenue_impact_annual:.1f}B" if revenue_impact_annual < 0 else f"+${revenue_impact_annual:.1f}B"
        revenue_item = QTableWidgetItem(revenue_impact_text)
        if revenue_impact_annual < 0:
            revenue_item.setForeground(QColor("red"))
        self.balance_table.setItem(1, 2, revenue_item)
        
        # Row 2: Operating Costs
        self.balance_table.setItem(2, 1, QTableWidgetItem("$3.8B"))  # Current value
        
        cost_impact = carbon_cost_impact / 1e9
        cost_impact_text = f"+${cost_impact:.1f}B" if cost_impact > 0 else f"${cost_impact:.1f}B"
        cost_item = QTableWidgetItem(cost_impact_text)
        if cost_impact > 0:
            cost_item.setForeground(QColor("red"))
        self.balance_table.setItem(2, 2, cost_item)
        
        # Row 3: Capital Expenditure
        self.balance_table.setItem(3, 1, QTableWidgetItem("$1.5B"))  # Current value
        
        # Assume some capex reduction due to transition
        capex_impact = -0.3
        capex_item = QTableWidgetItem(f"${capex_impact:.1f}B")
        capex_item.setForeground(QColor("red"))
        self.balance_table.setItem(3, 2, capex_item)
        
        # Row 4: Debt Metrics
        self.balance_table.setItem(4, 1, QTableWidgetItem("3.2x EBITDA"))  # Current value
        
        # Assume debt metrics worsen
        debt_impact = "+0.5x"
        debt_item = QTableWidgetItem(debt_impact)
        debt_item.setForeground(QColor("red"))
        self.balance_table.setItem(4, 2, debt_item)
    
    @pyqtSlot(int)
    def _risk_driver_changed(self, index):
        """Handle risk driver selection"""
        risk_driver = self.risk_driver_combo.currentText()
        self.logger.info(f"Selected risk driver: {risk_driver}")
        
        # Update visualization based on selected driver
        if self.risk_results:
            self._update_ui_with_results()
    
    @pyqtSlot(int)
    def _time_horizon_changed(self, index):
        """Handle time horizon selection"""
        time_horizon = self.time_horizon_combo.currentText()
        self.logger.info(f"Selected time horizon: {time_horizon}")
        
        # Update visualization based on selected time horizon
        if self.risk_results:
            self._update_ui_with_results()
    
    @pyqtSlot(int)
    def _carbon_price_changed(self, value):
        """Handle carbon price slider change"""
        self.logger.info(f"Carbon price adjusted to: ${value}")
        
        # Update carbon price chart with custom value
        if self.risk_results:
            transition_risks = self.risk_results.get("transition_risks", {})
            
            # Determine which scenario data to use
            scenario_data = None
            
            if self.scenario and self.scenario in transition_risks:
                scenario_data = transition_risks[self.scenario]
            elif transition_risks:
                # Use the first scenario
                scenario_data = next(iter(transition_risks.values()))
            
            if scenario_data:
                risk_drivers = scenario_data.get("risk_drivers", {})
                self._update_carbon_chart(risk_drivers.get("carbon_price", {}))