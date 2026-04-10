"""
Physical risk tab for the Climate Risk Integration Platform.
Displays physical climate risks for company assets.
"""

import logging
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                            QSplitter, QTabWidget, QGridLayout, QScrollArea,
                            QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QSizePolicy, QGroupBox)
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette, QBrush

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data_manager import DataManager
from core.risk_engine import RiskEngine


class PhysicalRiskTab(QWidget):
    """Physical risk tab showing climate risks for company assets"""
    
    def __init__(self, data_manager: DataManager, risk_engine: RiskEngine):
        super().__init__()
        
        self.logger = logging.getLogger("physical_risk_tab")
        self.data_manager = data_manager
        self.risk_engine = risk_engine
        
        # Initialize state
        self.company_id = None
        self.scenario = None
        self.risk_results = None
        
        # Set up the UI
        self._init_ui()
        
        self.logger.info("PhysicalRiskTab initialized")
    
    def _init_ui(self):
        """Initialize the physical risk UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header section with controls
        header_layout = QHBoxLayout()
        
        # Risk type selector
        risk_type_group = QGroupBox("Risk Type")
        risk_type_layout = QHBoxLayout(risk_type_group)
        
        self.risk_type_combo = QComboBox()
        self.risk_type_combo.addItems(["All Risks", "Flood", "Hurricane", "Wildfire", "Drought", "Extreme Heat"])
        self.risk_type_combo.currentIndexChanged.connect(self._risk_type_changed)
        risk_type_layout.addWidget(self.risk_type_combo)
        
        header_layout.addWidget(risk_type_group)
        
        # Time horizon selector
        time_horizon_group = QGroupBox("Time Horizon")
        time_horizon_layout = QHBoxLayout(time_horizon_group)
        
        self.time_horizon_combo = QComboBox()
        self.time_horizon_combo.addItems(["2030", "2050", "2100"])
        self.time_horizon_combo.currentIndexChanged.connect(self._time_horizon_changed)
        time_horizon_layout.addWidget(self.time_horizon_combo)
        
        header_layout.addWidget(time_horizon_group)
        
        # Add spacer
        header_layout.addStretch(1)
        
        # Overall risk score display
        score_group = QGroupBox("Physical Risk Score")
        score_layout = QVBoxLayout(score_group)
        
        self.risk_score_label = QLabel("--")
        self.risk_score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #777;")
        self.risk_score_label.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(self.risk_score_label)
        
        header_layout.addWidget(score_group)
        
        layout.addLayout(header_layout)
        
        # Main content in a splitter
        self.main_splitter = QSplitter(Qt.Vertical)
        
        # Upper section - Map and Charts
        upper_widget = QWidget()
        upper_layout = QHBoxLayout(upper_widget)
        upper_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left side - Map or asset visualization
        map_group = QGroupBox("Asset Risk Map")
        map_layout = QVBoxLayout(map_group)
        
        # Placeholder for map (would be replaced with a real map widget)
        self.map_figure = Figure(figsize=(5, 4), dpi=100)
        self.map_canvas = FigureCanvas(self.map_figure)
        map_layout.addWidget(self.map_canvas)
        
        upper_layout.addWidget(map_group, 3)  # 3:2 ratio for map vs charts
        
        # Right side - Charts
        charts_group = QGroupBox("Risk Analysis")
        charts_layout = QVBoxLayout(charts_group)
        
        # Risk breakdown chart
        self.breakdown_figure = Figure(figsize=(4, 3), dpi=100)
        self.breakdown_canvas = FigureCanvas(self.breakdown_figure)
        charts_layout.addWidget(self.breakdown_canvas)
        
        # Risk trend chart
        self.trend_figure = Figure(figsize=(4, 3), dpi=100)
        self.trend_canvas = FigureCanvas(self.trend_figure)
        charts_layout.addWidget(self.trend_canvas)
        
        upper_layout.addWidget(charts_group, 2)
        
        self.main_splitter.addWidget(upper_widget)
        
        # Lower section - Asset Details
        lower_widget = QWidget()
        lower_layout = QVBoxLayout(lower_widget)
        lower_layout.setContentsMargins(0, 0, 0, 0)
        
        # Asset risk table
        self.assets_table = QTableWidget(0, 6)
        self.assets_table.setHorizontalHeaderLabels(
            ["Asset Name", "Type", "Location", "Risk Score", "Highest Risk", "Value at Risk"]
        )
        self.assets_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.assets_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.assets_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.assets_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.assets_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.assets_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.assets_table.verticalHeader().setVisible(False)
        self.assets_table.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Improve selection handling with a proper connection method
        self.assets_table.itemSelectionChanged.connect(self._asset_selected)
        
        asset_table_group = QGroupBox("Asset Risk Details")
        asset_table_layout = QVBoxLayout(asset_table_group)
        asset_table_layout.addWidget(self.assets_table)
        
        lower_layout.addWidget(asset_table_group)
        
        # Asset detail section
        self.asset_detail_widget = QWidget()
        asset_detail_layout = QHBoxLayout(self.asset_detail_widget)
        
        # Asset info
        asset_info_group = QGroupBox("Asset Information")
        asset_info_layout = QGridLayout(asset_info_group)
        
        asset_info_layout.addWidget(QLabel("Name:"), 0, 0)
        self.asset_name_label = QLabel("--")
        asset_info_layout.addWidget(self.asset_name_label, 0, 1)
        
        asset_info_layout.addWidget(QLabel("Type:"), 1, 0)
        self.asset_type_label = QLabel("--")
        asset_info_layout.addWidget(self.asset_type_label, 1, 1)
        
        asset_info_layout.addWidget(QLabel("Location:"), 2, 0)
        self.asset_location_label = QLabel("--")
        asset_info_layout.addWidget(self.asset_location_label, 2, 1)
        
        asset_info_layout.addWidget(QLabel("Status:"), 3, 0)
        self.asset_status_label = QLabel("--")
        asset_info_layout.addWidget(self.asset_status_label, 3, 1)
        
        asset_detail_layout.addWidget(asset_info_group)
        
        # Asset risk breakdown
        asset_risk_group = QGroupBox("Asset Risk Breakdown")
        asset_risk_layout = QVBoxLayout(asset_risk_group)
        
        self.asset_risk_figure = Figure(figsize=(4, 3), dpi=100)
        self.asset_risk_canvas = FigureCanvas(self.asset_risk_figure)
        asset_risk_layout.addWidget(self.asset_risk_canvas)
        
        asset_detail_layout.addWidget(asset_risk_group)
        
        # Add asset detail to lower section
        lower_layout.addWidget(self.asset_detail_widget)
        
        # Initially hide asset detail widget until an asset is selected
        self.asset_detail_widget.setVisible(False)
        
        self.main_splitter.addWidget(lower_widget)
        
        # Set initial splitter sizes
        self.main_splitter.setSizes([int(self.height() * 0.6), int(self.height() * 0.4)])
        
        layout.addWidget(self.main_splitter)
        
        # Initialize charts
        self._initialize_charts()
    
    def _initialize_charts(self):
        """Initialize empty charts"""
        # Map figure
        self.map_figure.clear()
        map_ax = self.map_figure.add_subplot(111)
        map_ax.set_title("Asset Risk Map")
        map_ax.set_xlabel("Longitude")
        map_ax.set_ylabel("Latitude")
        map_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=map_ax.transAxes)
        self.map_figure.tight_layout()
        self.map_canvas.draw()
        
        # Risk breakdown figure
        self.breakdown_figure.clear()
        breakdown_ax = self.breakdown_figure.add_subplot(111)
        breakdown_ax.set_title("Risk Breakdown by Type")
        breakdown_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=breakdown_ax.transAxes)
        self.breakdown_figure.tight_layout()
        self.breakdown_canvas.draw()
        
        # Risk trend figure
        self.trend_figure.clear()
        trend_ax = self.trend_figure.add_subplot(111)
        trend_ax.set_title("Risk Trend Over Time")
        trend_ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=trend_ax.transAxes)
        self.trend_figure.tight_layout()
        self.trend_canvas.draw()
        
        # Asset risk figure
        self.asset_risk_figure.clear()
        asset_ax = self.asset_risk_figure.add_subplot(111)
        asset_ax.set_title("Asset Risk by Hazard Type")
        asset_ax.text(0.5, 0.5, "No asset selected", ha='center', va='center', transform=asset_ax.transAxes)
        self.asset_risk_figure.tight_layout()
        self.asset_risk_canvas.draw()
    
    def set_company(self, company_id: str):
        """Set the company to display"""
        self.company_id = company_id
        
        # Clear previous results
        self.risk_results = None
        self._clear_ui()
        
        self.logger.info(f"Set company in physical risk tab: {company_id}")
    
    def set_scenario(self, scenario: str):
        """Set the scenario to display"""
        self.scenario = scenario
        
        self.logger.info(f"Set scenario in physical risk tab: {scenario}")
        
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
        
        # Clear assets table
        self.assets_table.setRowCount(0)
        
        # Clear asset details
        self.asset_name_label.setText("--")
        self.asset_type_label.setText("--")
        self.asset_location_label.setText("--")
        self.asset_status_label.setText("--")
        
        # Hide asset detail section
        self.asset_detail_widget.setVisible(False)
        
        # Reset charts
        self._initialize_charts()
    
    def _update_ui_with_results(self):
        """Update UI with current risk results"""
        # Check if we have physical risk results
        if not self.risk_results or "physical_risks" not in self.risk_results:
            return
        
        physical_risks = self.risk_results["physical_risks"]
        
        # If specific scenario selected, show only that one
        if self.scenario and self.scenario in physical_risks:
            self._show_scenario_results(self.scenario, physical_risks[self.scenario])
        else:
            # Find the first available scenario
            if physical_risks:
                scenario = next(iter(physical_risks))
                self._show_scenario_results(scenario, physical_risks[scenario])
    
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
        
        # Update asset table
        self._update_asset_table(risk_data)
        
        # Update charts
        self._update_map_chart(risk_data)
        self._update_breakdown_chart(risk_data)
        self._update_trend_chart(risk_data)
    
    def _update_asset_table(self, risk_data: Dict[str, Any]):
        """Update the asset risk table"""
        asset_risks = risk_data.get("asset_risk_scores", [])
        
        # Clear table
        self.assets_table.setRowCount(0)
        
        if not asset_risks:
            return
        
        # Sort assets by risk score (descending)
        sorted_assets = sorted(asset_risks, key=lambda x: x.get("overall_score", 0), reverse=True)
        
        # Add rows for each asset
        self.assets_table.setRowCount(len(sorted_assets))
        
        for i, asset in enumerate(sorted_assets):
            # Asset name
            name_item = QTableWidgetItem(asset.get("asset_name", "Unknown"))
            self.assets_table.setItem(i, 0, name_item)
            
            # Asset type
            type_item = QTableWidgetItem(asset.get("asset_type", "Unknown").title())
            self.assets_table.setItem(i, 1, type_item)
            
            # Location
            lat = asset.get("latitude", 0)
            lon = asset.get("longitude", 0)
            location_item = QTableWidgetItem(f"{lat:.2f}, {lon:.2f}")
            self.assets_table.setItem(i, 2, location_item)
            
            # Risk score
            score = asset.get("overall_score", 0)
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
            
            self.assets_table.setItem(i, 3, score_item)
            
            # Highest risk type
            risk_types = asset.get("risk_types", {})
            highest_risk = max(risk_types.items(), key=lambda x: x[1], default=("None", 0))
            highest_risk_item = QTableWidgetItem(highest_risk[0].title())
            self.assets_table.setItem(i, 4, highest_risk_item)
            
            # Value at risk (placeholder - would come from real data)
            asset_type = asset.get("asset_type", "")
            if asset_type == "refinery":
                var_value = score * 50e6  # $50M * risk score
            elif asset_type == "pipeline":
                var_value = score * 20e6  # $20M * risk score
            elif asset_type == "well":
                var_value = score * 5e6   # $5M * risk score
            else:
                var_value = score * 10e6  # $10M * risk score
            
            if var_value >= 1e9:
                var_text = f"${var_value/1e9:.1f}B"
            elif var_value >= 1e6:
                var_text = f"${var_value/1e6:.1f}M"
            else:
                var_text = f"${var_value:.0f}"
            
            var_item = QTableWidgetItem(var_text)
            self.assets_table.setItem(i, 5, var_item)
        
        # Resize rows to content
        self.assets_table.resizeRowsToContents()
    
    def _update_map_chart(self, risk_data: Dict[str, Any]):
        """Update the asset risk map"""
        asset_risks = risk_data.get("asset_risk_scores", [])
        
        if not asset_risks:
            return
        
        # Clear figure
        self.map_figure.clear()
        map_ax = self.map_figure.add_subplot(111)
        
        # Extract locations and risk scores
        lats = []
        lons = []
        sizes = []
        colors = []
        names = []
        
        for asset in asset_risks:
            lats.append(asset.get("latitude", 0))
            lons.append(asset.get("longitude", 0))
            score = asset.get("overall_score", 0)
            sizes.append(50 + score * 20)  # Scale size by risk score
            names.append(asset.get("asset_name", "Unknown"))
            
            # Color based on risk score
            if score >= 7:
                colors.append("#d9534f")  # Red
            elif score >= 5:
                colors.append("#f0ad4e")  # Orange
            elif score >= 3:
                colors.append("#5bc0de")  # Blue
            else:
                colors.append("#5cb85c")  # Green
        
        # Plot scatter
        scatter = map_ax.scatter(lons, lats, s=sizes, c=colors, alpha=0.7)
        
        # Add some basic map features (a simplified approach)
        map_ax.set_xlabel("Longitude")
        map_ax.set_ylabel("Latitude")
        map_ax.set_title("Asset Risk Map")
        map_ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#d9534f", label='High Risk (7-10)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#f0ad4e", label='Medium-High Risk (5-7)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#5bc0de", label='Medium-Low Risk (3-5)', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#5cb85c", label='Low Risk (0-3)', markersize=10),
        ]
        map_ax.legend(handles=legend_elements, loc='lower right')
        
        # Set reasonable axis limits with some padding
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        
        lon_range = max_lon - min_lon
        lat_range = max_lat - min_lat
        
        # Add 10% padding
        map_ax.set_xlim(min_lon - lon_range * 0.1, max_lon + lon_range * 0.1)
        map_ax.set_ylim(min_lat - lat_range * 0.1, max_lat + lat_range * 0.1)
        
        # Add equal aspect ratio to prevent distortion
        map_ax.set_aspect('equal', adjustable='datalim')
        
        # Redraw
        self.map_figure.tight_layout()
        self.map_canvas.draw()
    
    def _update_breakdown_chart(self, risk_data: Dict[str, Any]):
        """Update the risk breakdown chart"""
        risk_breakdown = risk_data.get("risk_breakdown", {})
        
        if not risk_breakdown:
            return
        
        # Clear figure
        self.breakdown_figure.clear()
        breakdown_ax = self.breakdown_figure.add_subplot(111)
        
        # Extract risk types and scores
        risk_types = []
        scores = []
        colors = []
        
        for risk_type, risk_info in risk_breakdown.items():
            # Pretty risk type names
            type_names = {
                "flood": "Flooding",
                "hurricane": "Tropical Cyclones",
                "fire": "Wildfires",
                "drought": "Drought",
                "extreme_heat": "Extreme Heat"
            }
            
            risk_name = type_names.get(risk_type, risk_type.title())
            risk_types.append(risk_name)
            
            score = risk_info.get("score", 0)
            scores.append(score)
            
            # Color based on risk score
            if score >= 7:
                colors.append("#d9534f")  # Red
            elif score >= 5:
                colors.append("#f0ad4e")  # Orange
            elif score >= 3:
                colors.append("#5bc0de")  # Blue
            else:
                colors.append("#5cb85c")  # Green
        
        # Plot horizontal bar chart
        y_pos = np.arange(len(risk_types))
        bars = breakdown_ax.barh(y_pos, scores, color=colors)
        
        # Add labels and styling
        breakdown_ax.set_yticks(y_pos)
        breakdown_ax.set_yticklabels(risk_types)
        breakdown_ax.set_xlabel("Risk Score (0-10)")
        breakdown_ax.set_title("Risk Breakdown by Hazard Type")
        breakdown_ax.set_xlim(0, 10)
        
        # Add score labels to bars
        for i, bar in enumerate(bars):
            score = scores[i]
            breakdown_ax.text(score + 0.1, bar.get_y() + bar.get_height()/2,
                           f"{score:.1f}", va='center')
        
        # Redraw
        self.breakdown_figure.tight_layout()
        self.breakdown_canvas.draw()
    
    def _update_trend_chart(self, risk_data: Dict[str, Any]):
        """Update the risk trend chart"""
        # This would normally use real projection data
        # For demo, we'll generate synthetic trend data
        
        # Clear figure
        self.trend_figure.clear()
        trend_ax = self.trend_figure.add_subplot(111)
        
        # Create synthetic data based on risk breakdown
        risk_breakdown = risk_data.get("risk_breakdown", {})
        
        if not risk_breakdown:
            trend_ax.text(0.5, 0.5, "No trend data available", ha='center', va='center', transform=trend_ax.transAxes)
            self.trend_figure.tight_layout()
            self.trend_canvas.draw()
            return
        
        # Years for projection
        years = [2020, 2030, 2050, 2080]
        
        # Generate trend lines for top 3 risks
        top_risks = sorted(risk_breakdown.items(), key=lambda x: x[1].get("score", 0), reverse=True)[:3]
        
        for risk_type, risk_info in top_risks:
            # Pretty risk type names
            type_names = {
                "flood": "Flooding",
                "hurricane": "Tropical Cyclones",
                "fire": "Wildfires",
                "drought": "Drought",
                "extreme_heat": "Extreme Heat"
            }
            
            risk_name = type_names.get(risk_type, risk_type.title())
            
            # Base score
            base_score = risk_info.get("score", 0)
            
            # Generate trend based on risk type and base score
            # Different risk types have different growth trajectories
            if risk_type == "flood" or risk_type == "hurricane":
                # Faster growth for precipitation-related risks
                scores = [
                    max(0, base_score * 0.8),  # 2020
                    base_score,                # 2030
                    min(10, base_score * 1.3), # 2050
                    min(10, base_score * 1.6)  # 2080
                ]
            elif risk_type == "extreme_heat" or risk_type == "fire":
                # Steady growth for temperature-related risks
                scores = [
                    max(0, base_score * 0.85),  # 2020
                    base_score,                 # 2030
                    min(10, base_score * 1.25), # 2050
                    min(10, base_score * 1.5)   # 2080
                ]
            else:
                # Moderate growth for other risks
                scores = [
                    max(0, base_score * 0.9),  # 2020
                    base_score,                # 2030
                    min(10, base_score * 1.2), # 2050
                    min(10, base_score * 1.4)  # 2080
                ]
            
            # Plot line
            trend_ax.plot(years, scores, marker='o', label=risk_name)
        
        # Add labels and styling
        trend_ax.set_xlabel("Year")
        trend_ax.set_ylabel("Risk Score (0-10)")
        trend_ax.set_title("Risk Projection Over Time")
        trend_ax.set_ylim(0, 10)
        trend_ax.grid(True, linestyle='--', alpha=0.6)
        trend_ax.legend()
        
        # Redraw
        self.trend_figure.tight_layout()
        self.trend_canvas.draw()
    
    def _update_asset_risk_chart(self, asset_data: Dict[str, Any]):
        """Update the asset risk breakdown chart"""
        risk_types = asset_data.get("risk_types", {})
        
        if not risk_types:
            return
        
        # Clear figure
        self.asset_risk_figure.clear()
        asset_ax = self.asset_risk_figure.add_subplot(111)
        
        # Extract risk types and scores
        types = []
        scores = []
        colors = []
        
        for risk_type, score in risk_types.items():
            # Pretty risk type names
            type_names = {
                "flood": "Flooding",
                "hurricane": "Tropical Cyclones",
                "fire": "Wildfires",
                "drought": "Drought",
                "extreme_heat": "Extreme Heat"
            }
            
            risk_name = type_names.get(risk_type, risk_type.title())
            types.append(risk_name)
            scores.append(score)
            
            # Color based on risk score
            if score >= 7:
                colors.append("#d9534f")  # Red
            elif score >= 5:
                colors.append("#f0ad4e")  # Orange
            elif score >= 3:
                colors.append("#5bc0de")  # Blue
            else:
                colors.append("#5cb85c")  # Green
        
        # Plot bar chart
        x_pos = np.arange(len(types))
        bars = asset_ax.bar(x_pos, scores, color=colors)
        
        # Add labels and styling
        asset_ax.set_xticks(x_pos)
        asset_ax.set_xticklabels(types, rotation=45, ha='right')
        asset_ax.set_ylabel("Risk Score (0-10)")
        asset_ax.set_title(f"Risk Breakdown for {asset_data.get('asset_name', 'Asset')}")
        asset_ax.set_ylim(0, 10)
        
        # Add score labels to bars
        for i, bar in enumerate(bars):
            score = scores[i]
            asset_ax.text(bar.get_x() + bar.get_width()/2, score + 0.1,
                       f"{score:.1f}", ha='center')
        
        # Redraw
        self.asset_risk_figure.tight_layout()
        self.asset_risk_canvas.draw()
    
    @pyqtSlot()
    def _asset_selected(self):
        """Handle asset selection in the table"""
        selected_items = self.assets_table.selectedItems()
        
        if not selected_items:
            self.asset_detail_widget.setVisible(False)
            return
        
        # Get the selected row
        selected_row = selected_items[0].row()
        
        # Get asset name from the first column
        asset_name = self.assets_table.item(selected_row, 0).text()
        
        # Find the corresponding asset in our data
        if not self.risk_results or "physical_risks" not in self.risk_results:
            return
        
        # Determine which scenario data to use
        physical_risks = self.risk_results["physical_risks"]
        scenario_data = None
        
        if self.scenario and self.scenario in physical_risks:
            scenario_data = physical_risks[self.scenario]
        elif physical_risks:
            # Use the first scenario
            scenario_data = next(iter(physical_risks.values()))
        
        if not scenario_data:
            return
        
        # Find the asset
        asset_risks = scenario_data.get("asset_risk_scores", [])
        selected_asset = None
        
        for asset in asset_risks:
            if asset.get("asset_name") == asset_name:
                selected_asset = asset
                break
        
        if not selected_asset:
            return
        
        # Update asset details
        self.asset_name_label.setText(selected_asset.get("asset_name", "--"))
        self.asset_type_label.setText(selected_asset.get("asset_type", "--").title())
        
        lat = selected_asset.get("latitude", 0)
        lon = selected_asset.get("longitude", 0)
        self.asset_location_label.setText(f"{lat:.4f}, {lon:.4f}")
        
        # Status is synthetic
        risk_score = selected_asset.get("overall_score", 0)
        if risk_score >= 7:
            status = "High Risk - Immediate Action Required"
        elif risk_score >= 5:
            status = "Medium Risk - Monitoring Required"
        else:
            status = "Low Risk - Normal Operations"
        
        self.asset_status_label.setText(status)
        
        # Update asset risk chart
        self._update_asset_risk_chart(selected_asset)
        
        # Show asset detail section
        self.asset_detail_widget.setVisible(True)
    
    @pyqtSlot(int)
    def _risk_type_changed(self, index):
        """Handle risk type selection"""
        risk_type = self.risk_type_combo.currentText()
        self.logger.info(f"Selected risk type: {risk_type}")
        
        # Update visualization based on selected risk type
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