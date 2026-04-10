"""
Dashboard tab for the Climate Risk Integration Platform.
Provides overview of climate risks and key metrics.
"""

import logging
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                            QSplitter, QTabWidget, QGridLayout, QScrollArea,
                            QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QSizePolicy)
from PyQt5.QtCore import Qt, QSize, pyqtSlot, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QBrush, QLinearGradient

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data_manager import DataManager
from core.risk_engine import RiskEngine


class DashboardTab(QWidget):
    """Dashboard tab showing overview of climate risks and key metrics"""
    
    def __init__(self, data_manager: DataManager, risk_engine: RiskEngine):
        super().__init__()
        
        self.logger = logging.getLogger("dashboard_tab")
        self.data_manager = data_manager
        self.risk_engine = risk_engine
        
        # Initialize state
        self.company_id = None
        self.risk_results = None
        
        # Set up the UI
        self._init_ui()
        
        self.logger.info("DashboardTab initialized")
    
    def _init_ui(self):
        """Initialize the dashboard UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header section
        header_layout = QHBoxLayout()
        
        # Company info panel
        self.company_info = QFrame()
        self.company_info.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.company_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        
        company_layout = QVBoxLayout(self.company_info)
        
        self.company_name_label = QLabel("No Company Selected")
        self.company_name_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        company_layout.addWidget(self.company_name_label)
        
        self.company_details_label = QLabel("")
        company_layout.addWidget(self.company_details_label)
        
        header_layout.addWidget(self.company_info)
        
        # Overall risk score
        self.risk_score_frame = QFrame()
        self.risk_score_frame.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.risk_score_frame.setMinimumWidth(150)
        self.risk_score_frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        
        risk_score_layout = QVBoxLayout(self.risk_score_frame)
        
        risk_score_header = QLabel("Overall Risk")
        risk_score_header.setAlignment(Qt.AlignCenter)
        risk_score_layout.addWidget(risk_score_header)
        
        self.risk_score_label = QLabel("--")
        self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #777;")
        self.risk_score_label.setAlignment(Qt.AlignCenter)
        risk_score_layout.addWidget(self.risk_score_label)
        
        header_layout.addWidget(self.risk_score_frame)
        
        layout.addLayout(header_layout)
        
        # Main dashboard sections in a splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Left panel - Key Metrics
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        left_layout = QVBoxLayout(left_panel)
        
        metrics_header = QLabel("Key Risk Metrics")
        metrics_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(metrics_header)
        
        # Risk breakdown
        self.metrics_grid = QGridLayout()
        self.metrics_grid.setColumnStretch(1, 1)
        
        # Physical risk score
        self.metrics_grid.addWidget(QLabel("Physical Risk:"), 0, 0)
        self.physical_risk_label = QLabel("--")
        self.metrics_grid.addWidget(self.physical_risk_label, 0, 1)
        
        # Transition risk score
        self.metrics_grid.addWidget(QLabel("Transition Risk:"), 1, 0)
        self.transition_risk_label = QLabel("--")
        self.metrics_grid.addWidget(self.transition_risk_label, 1, 1)
        
        # Financial impact
        self.metrics_grid.addWidget(QLabel("Financial Impact:"), 2, 0)
        self.financial_impact_label = QLabel("--")
        self.metrics_grid.addWidget(self.financial_impact_label, 2, 1)
        
        # Worst case scenario
        self.metrics_grid.addWidget(QLabel("Worst Case Scenario:"), 3, 0)
        self.worst_case_label = QLabel("--")
        self.metrics_grid.addWidget(self.worst_case_label, 3, 1)
        
        left_layout.addLayout(self.metrics_grid)
        
        # Risk comparison chart
        risk_chart_label = QLabel("Risk Comparison by Scenario")
        risk_chart_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        left_layout.addWidget(risk_chart_label)
        
        self.risk_chart_widget = QWidget()
        risk_chart_layout = QVBoxLayout(self.risk_chart_widget)
        
        self.risk_figure = Figure(figsize=(5, 4), dpi=100)
        self.risk_canvas = FigureCanvas(self.risk_figure)
        risk_chart_layout.addWidget(self.risk_canvas)
        
        left_layout.addWidget(self.risk_chart_widget)
        
        # Add left panel to splitter
        splitter.addWidget(left_panel)
        
        # Right panel - Risk Factors & Explanations
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        right_layout = QVBoxLayout(right_panel)
        
        # Risk factor table
        factors_header = QLabel("Key Risk Factors")
        factors_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(factors_header)
        
        self.factors_table = QTableWidget(0, 3)
        self.factors_table.setHorizontalHeaderLabels(["Factor", "Type", "Score"])
        self.factors_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.factors_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.factors_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.factors_table.verticalHeader().setVisible(False)
        self.factors_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        right_layout.addWidget(self.factors_table)
        
        # Risk explanation
        explanation_header = QLabel("Risk Explanation")
        explanation_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(explanation_header)
        
        self.explanation_label = QLabel("No risk assessment available.")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.explanation_label.setMinimumHeight(100)
        self.explanation_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.explanation_label.setTextFormat(Qt.RichText)
        
        explanation_scroll = QScrollArea()
        explanation_scroll.setWidget(self.explanation_label)
        explanation_scroll.setWidgetResizable(True)
        explanation_scroll.setFrameShape(QFrame.NoFrame)
        
        right_layout.addWidget(explanation_scroll)
        
        # Recommendations
        recommendations_header = QLabel("Recommended Actions")
        recommendations_header.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(recommendations_header)
        
        self.recommendations_table = QTableWidget(0, 2)
        self.recommendations_table.setHorizontalHeaderLabels(["Category", "Recommendation"])
        self.recommendations_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.recommendations_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.recommendations_table.verticalHeader().setVisible(False)
        
        right_layout.addWidget(self.recommendations_table)
        
        # Add right panel to splitter
        splitter.addWidget(right_panel)
        
        # Set initial sizes
        splitter.setSizes([int(self.width() * 0.4), int(self.width() * 0.6)])
        
        layout.addWidget(splitter)
    
    def set_company(self, company_id: str):
        """Set the company to display"""
        self.company_id = company_id
        
        # Update company name and details
        self.company_name_label.setText(f"Company: {company_id}")
        
        # Clear previous results
        self.risk_results = None
        self._clear_ui()
        
        # In a real implementation, this would fetch company details
        # For demo, we'll use placeholder data
        self.company_details_label.setText("Sector: Oil & Gas | Industry: Integrated Oil & Gas")
    
    def update_risk_results(self, results: Dict[str, Any]):
        """Update the dashboard with risk calculation results"""
        if not results or "company_id" not in results:
            return
        
        # Store results
        self.risk_results = results
        company_id = results["company_id"]
        
        # Update company info if it matches current selected company
        if company_id == self.company_id:
            self._update_risk_metrics(results)
            self._update_risk_chart(results)
            self._update_risk_factors(results)
            self._update_explanations(results)
            self._update_recommendations(results)
    
    def update_data(self, data_type: str, data: Any):
        """Update with new data"""
        # This method would update specific parts of the UI based on data updates
        # For the dashboard, we're mostly concerned with complete risk results
        pass
    
    def _clear_ui(self):
        """Clear UI elements"""
        # Clear risk scores
        self.risk_score_label.setText("--")
        self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #777;")
        
        self.physical_risk_label.setText("--")
        self.transition_risk_label.setText("--")
        self.financial_impact_label.setText("--")
        self.worst_case_label.setText("--")
        
        # Clear tables
        self.factors_table.setRowCount(0)
        self.recommendations_table.setRowCount(0)
        
        # Clear explanation
        self.explanation_label.setText("No risk assessment available.")
        
        # Clear chart
        self.risk_figure.clear()
        self.risk_canvas.draw()
    
    def _update_risk_metrics(self, results: Dict[str, Any]):
        """Update the risk metrics section"""
        # Extract combined metrics
        combined_metrics = results.get("combined_metrics", {})
        overall = combined_metrics.get("overall", {})
        
        # Update overall risk score
        avg_score = overall.get("average_score", 0)
        self.risk_score_label.setText(f"{avg_score:.1f}")
        
        # Set color based on score
        if avg_score >= 7:
            self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #d9534f;")
        elif avg_score >= 5:
            self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #f0ad4e;")
        elif avg_score >= 3:
            self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #5bc0de;")
        else:
            self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #5cb85c;")
        
        # Update detailed metrics
        # Calculate average physical and transition risk scores
        physical_scores = []
        transition_scores = []
        
        physical_risks = results.get("physical_risks", {})
        transition_risks = results.get("transition_risks", {})
        
        for scenario, risk_data in physical_risks.items():
            physical_scores.append(risk_data.get("overall_score", 0))
        
        for scenario, risk_data in transition_risks.items():
            transition_scores.append(risk_data.get("overall_score", 0))
        
        avg_physical = sum(physical_scores) / len(physical_scores) if physical_scores else 0
        avg_transition = sum(transition_scores) / len(transition_scores) if transition_scores else 0
        
        self.physical_risk_label.setText(f"{avg_physical:.1f} / 10")
        self.transition_risk_label.setText(f"{avg_transition:.1f} / 10")
        
        # Financial impact
        financial_impact = overall.get("average_financial_impact", 0)
        if financial_impact >= 1e9:
            self.financial_impact_label.setText(f"${financial_impact/1e9:.1f}B")
        elif financial_impact >= 1e6:
            self.financial_impact_label.setText(f"${financial_impact/1e6:.1f}M")
        else:
            self.financial_impact_label.setText(f"${financial_impact:.0f}")
        
        # Worst case scenario
        worst_case = overall.get("worst_case_scenario", "Unknown")
        
        # Map scenario IDs to readable names
        scenario_names = {
            "ipcc_ssp119": "Low Emissions (1.5°C)",
            "ipcc_ssp126": "Low-Medium Emissions (2°C)",
            "ipcc_ssp245": "Medium Emissions (2.5-3°C)",
            "ipcc_ssp370": "Medium-High Emissions (3-4°C)",
            "ipcc_ssp585": "High Emissions (4-5°C)"
        }
        
        worst_case_name = scenario_names.get(worst_case, worst_case)
        self.worst_case_label.setText(worst_case_name)
    
    def _update_risk_chart(self, results: Dict[str, Any]):
        """Update the risk comparison chart"""
        # Extract data for chart
        combined_metrics = results.get("combined_metrics", {})
        
        # Prepare data
        scenarios = []
        physical_scores = []
        transition_scores = []
        
        # Scenario name mapping
        scenario_names = {
            "ipcc_ssp119": "Low 1.5°C",
            "ipcc_ssp126": "Low-Med 2°C",
            "ipcc_ssp245": "Med 3°C",
            "ipcc_ssp370": "Med-High 4°C",
            "ipcc_ssp585": "High 5°C"
        }
        
        for scenario, metrics in combined_metrics.items():
            if scenario != "overall":
                scenario_label = scenario_names.get(scenario, scenario)
                scenarios.append(scenario_label)
                physical_scores.append(metrics.get("physical_score", 0))
                transition_scores.append(metrics.get("transition_score", 0))
        
        if not scenarios:
            return
        
        # Create chart
        self.risk_figure.clear()
        ax = self.risk_figure.add_subplot(111)
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        ax.bar(x - width/2, physical_scores, width, label='Physical Risk', color='#5bc0de')
        ax.bar(x + width/2, transition_scores, width, label='Transition Risk', color='#f0ad4e')
        
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha='right')
        ax.set_ylabel('Risk Score (0-10)')
        ax.set_title('Risk Comparison by Scenario')
        ax.legend()
        ax.set_ylim(0, 10)
        
        self.risk_figure.tight_layout()
        self.risk_canvas.draw()
    
    def _update_risk_factors(self, results: Dict[str, Any]):
        """Update the risk factors table"""
        # Get explanations
        explanations = results.get("explanations", {})
        risk_factors = explanations.get("factor_contributions", [])
        
        # Clear table
        self.factors_table.setRowCount(0)
        
        # Add rows for each factor
        self.factors_table.setRowCount(len(risk_factors))
        
        for i, factor in enumerate(risk_factors):
            name_item = QTableWidgetItem(factor.get("name", "Unknown"))
            
            factor_type = factor.get("type", "")
            type_item = QTableWidgetItem(factor_type.title())
            
            score = factor.get("score", 0)
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
            
            self.factors_table.setItem(i, 0, name_item)
            self.factors_table.setItem(i, 1, type_item)
            self.factors_table.setItem(i, 2, score_item)
        
        # Resize rows to content
        self.factors_table.resizeRowsToContents()
    
    def _update_explanations(self, results: Dict[str, Any]):
        """Update the explanations section"""
        explanations = results.get("explanations", {})
        summary = explanations.get("summary", "No explanation available.")
        
        self.explanation_label.setText(summary)
    
    def _update_recommendations(self, results: Dict[str, Any]):
        """Update the recommendations table"""
        # Get recommendations
        explanations = results.get("explanations", {})
        recommendations = explanations.get("recommendations", [])
        
        # Clear table
        self.recommendations_table.setRowCount(0)
        
        # Add rows for each recommendation
        self.recommendations_table.setRowCount(len(recommendations))
        
        for i, recommendation in enumerate(recommendations):
            category_item = QTableWidgetItem(recommendation.get("category", "General"))
            
            rec_text = recommendation.get("recommendation", "")
            recommendation_item = QTableWidgetItem(rec_text)
            recommendation_item.setToolTip(recommendation.get("rationale", ""))
            
            self.recommendations_table.setItem(i, 0, category_item)
            self.recommendations_table.setItem(i, 1, recommendation_item)
        
        # Resize rows to content
        self.recommendations_table.resizeRowsToContents()