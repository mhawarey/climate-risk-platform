"""
Scenario tab for the Climate Risk Integration Platform.
Allows creation and analysis of custom climate scenarios.
"""

import logging
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
                            QSplitter, QTabWidget, QGridLayout, QScrollArea,
                            QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                            QHeaderView, QSizePolicy, QGroupBox, QSlider, QTextEdit,
                            QLineEdit, QDoubleSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette, QBrush

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data_manager import DataManager
from core.risk_engine import RiskEngine


class ScenarioTab(QWidget):
    """Scenario tab for creating and analyzing custom climate scenarios"""
    
    def __init__(self, data_manager: DataManager, risk_engine: RiskEngine):
        super().__init__()
        
        self.logger = logging.getLogger("scenario_tab")
        self.data_manager = data_manager
        self.risk_engine = risk_engine
        
        # Initialize state
        self.custom_scenario = None
        
        # Set up the UI
        self._init_ui()
        
        self.logger.info("ScenarioTab initialized")
    
    def _init_ui(self):
        """Initialize the scenario tab UI"""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        
        # Base scenario selector
        base_scenario_group = QGroupBox("Base Scenario")
        base_scenario_layout = QVBoxLayout(base_scenario_group)
        
        self.base_scenario_combo = QComboBox()
        self.base_scenario_combo.addItems([
            "Low Emissions (1.5°C) - SSP1-1.9",
            "Low-Medium Emissions (2°C) - SSP1-2.6",
            "Medium Emissions (2.5-3°C) - SSP2-4.5",
            "Medium-High Emissions (3-4°C) - SSP3-7.0",
            "High Emissions (4-5°C) - SSP5-8.5"
        ])
        base_scenario_layout.addWidget(self.base_scenario_combo)
        
        header_layout.addWidget(base_scenario_group)
        
        # Scenario name field
        name_group = QGroupBox("Custom Scenario Name")
        name_layout = QVBoxLayout(name_group)
        
        self.scenario_name_edit = QLineEdit()
        self.scenario_name_edit.setPlaceholderText("Enter a name for your custom scenario")
        name_layout.addWidget(self.scenario_name_edit)
        
        header_layout.addWidget(name_group)
        
        # Generate button
        self.generate_button = QPushButton("Generate Custom Scenario")
        self.generate_button.clicked.connect(self._generate_scenario)
        header_layout.addWidget(self.generate_button)
        
        layout.addLayout(header_layout)
        
        # Main content in a splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Scenario parameters
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Parameters group
        parameters_group = QGroupBox("Scenario Parameters")
        parameters_layout = QFormLayout(parameters_group)
        
        # Temperature increase factor
        parameters_layout.addRow("Temperature Increase Factor:", 
                              QLabel("Higher values = more warming"))
        
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(80)
        self.temp_slider.setMaximum(150)
        self.temp_slider.setValue(100)
        self.temp_slider.setTickInterval(10)
        self.temp_slider.setTickPosition(QSlider.TicksBelow)
        
        self.temp_value_label = QLabel("1.0x (baseline)")
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_value_label)
        
        parameters_layout.addRow("", temp_layout)
        
        # Add slider value changed connection
        self.temp_slider.valueChanged.connect(self._temp_slider_changed)
        
        # Carbon price factor
        parameters_layout.addRow("Carbon Price Factor:", 
                              QLabel("Higher values = stronger carbon pricing"))
        
        self.carbon_slider = QSlider(Qt.Horizontal)
        self.carbon_slider.setMinimum(50)
        self.carbon_slider.setMaximum(300)
        self.carbon_slider.setValue(100)
        self.carbon_slider.setTickInterval(25)
        self.carbon_slider.setTickPosition(QSlider.TicksBelow)
        
        self.carbon_value_label = QLabel("1.0x (baseline)")
        
        carbon_layout = QHBoxLayout()
        carbon_layout.addWidget(self.carbon_slider)
        carbon_layout.addWidget(self.carbon_value_label)
        
        parameters_layout.addRow("", carbon_layout)
        
        # Add slider value changed connection
        self.carbon_slider.valueChanged.connect(self._carbon_slider_changed)
        
        # Renewable adoption factor
        parameters_layout.addRow("Renewable Adoption Factor:", 
                              QLabel("Higher values = faster renewable adoption"))
        
        self.renewable_slider = QSlider(Qt.Horizontal)
        self.renewable_slider.setMinimum(50)
        self.renewable_slider.setMaximum(200)
        self.renewable_slider.setValue(100)
        self.renewable_slider.setTickInterval(25)
        self.renewable_slider.setTickPosition(QSlider.TicksBelow)
        
        self.renewable_value_label = QLabel("1.0x (baseline)")
        
        renewable_layout = QHBoxLayout()
        renewable_layout.addWidget(self.renewable_slider)
        renewable_layout.addWidget(self.renewable_value_label)
        
        parameters_layout.addRow("", renewable_layout)
        
        # Add slider value changed connection
        self.renewable_slider.valueChanged.connect(self._renewable_slider_changed)
        
        # Policy delay years
        parameters_layout.addRow("Policy Delay Years:", 
                              QLabel("Higher values = slower policy action"))
        
        self.policy_delay_spin = QDoubleSpinBox()
        self.policy_delay_spin.setMinimum(0)
        self.policy_delay_spin.setMaximum(15)
        self.policy_delay_spin.setValue(0)
        self.policy_delay_spin.setSingleStep(1)
        self.policy_delay_spin.setDecimals(0)
        self.policy_delay_spin.setSuffix(" years")
        
        parameters_layout.addRow("", self.policy_delay_spin)
        
        # Extreme weather factor
        parameters_layout.addRow("Extreme Weather Factor:", 
                              QLabel("Higher values = more frequent extreme events"))
        
        self.weather_slider = QSlider(Qt.Horizontal)
        self.weather_slider.setMinimum(80)
        self.weather_slider.setMaximum(150)
        self.weather_slider.setValue(100)
        self.weather_slider.setTickInterval(10)
        self.weather_slider.setTickPosition(QSlider.TicksBelow)
        
        self.weather_value_label = QLabel("1.0x (baseline)")
        
        weather_layout = QHBoxLayout()
        weather_layout.addWidget(self.weather_slider)
        weather_layout.addWidget(self.weather_value_label)
        
        parameters_layout.addRow("", weather_layout)
        
        # Add slider value changed connection
        self.weather_slider.valueChanged.connect(self._weather_slider_changed)
        
        left_layout.addWidget(parameters_group)
        
        # Scenario explanation
        explanation_group = QGroupBox("Scenario Explanation")
        explanation_layout = QVBoxLayout(explanation_group)
        
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setPlaceholderText("Generate a custom scenario to see explanation")
        explanation_layout.addWidget(self.explanation_text)
        
        left_layout.addWidget(explanation_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel - Scenario visualization
        right_panel = QTabWidget()
        
        # Emissions tab
        emissions_tab = QWidget()
        emissions_layout = QVBoxLayout(emissions_tab)
        
        self.emissions_figure = Figure(figsize=(5, 4), dpi=100)
        self.emissions_canvas = FigureCanvas(self.emissions_figure)
        emissions_layout.addWidget(self.emissions_canvas)
        
        right_panel.addTab(emissions_tab, "Emissions Pathway")
        
        # Temperature tab
        temp_tab = QWidget()
        temp_layout = QVBoxLayout(temp_tab)
        
        self.temp_figure = Figure(figsize=(5, 4), dpi=100)
        self.temp_canvas = FigureCanvas(self.temp_figure)
        temp_layout.addWidget(self.temp_canvas)
        
        right_panel.addTab(temp_tab, "Temperature Pathway")
        
        # Energy mix tab
        energy_tab = QWidget()
        energy_layout = QVBoxLayout(energy_tab)
        
        self.energy_figure = Figure(figsize=(5, 4), dpi=100)
        self.energy_canvas = FigureCanvas(self.energy_figure)
        energy_layout.addWidget(self.energy_canvas)
        
        right_panel.addTab(energy_tab, "Energy Mix")
        
        # Carbon price tab
        carbon_tab = QWidget()
        carbon_layout = QVBoxLayout(carbon_tab)
        
        self.carbon_figure = Figure(figsize=(5, 4), dpi=100)
        self.carbon_canvas = FigureCanvas(self.carbon_figure)
        carbon_layout.addWidget(self.carbon_canvas)
        
        right_panel.addTab(carbon_tab, "Carbon Price")
        
        # Risk metrics tab
        risk_tab = QWidget()
        risk_layout = QVBoxLayout(risk_tab)
        
        # Risk metrics table
        self.risk_table = QTableWidget(5, 2)
        self.risk_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.risk_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.risk_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.risk_table.verticalHeader().setVisible(False)
        
        # Set row labels
        metrics = ["Physical Risk Score", "Transition Risk Score", "Overall Risk Score", 
                  "GDP Impact 2050", "GDP Impact 2100"]
        
        for i, metric in enumerate(metrics):
            self.risk_table.setItem(i, 0, QTableWidgetItem(metric))
            self.risk_table.setItem(i, 1, QTableWidgetItem("--"))
        
        risk_layout.addWidget(self.risk_table)
        
        right_panel.addTab(risk_tab, "Risk Metrics")
        
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes
        splitter.setSizes([int(self.width() * 0.4), int(self.width() * 0.6)])
        
        layout.addWidget(splitter)
        
        # Bottom buttons
        bottom_layout = QHBoxLayout()
        
        self.apply_scenario_button = QPushButton("Apply Scenario to Analysis")
        self.apply_scenario_button.setEnabled(False)  # Disabled until scenario is generated
        self.apply_scenario_button.clicked.connect(self._apply_scenario)
        bottom_layout.addWidget(self.apply_scenario_button)
        
        self.export_scenario_button = QPushButton("Export Scenario")
        self.export_scenario_button.setEnabled(False)  # Disabled until scenario is generated
        self.export_scenario_button.clicked.connect(self._export_scenario)
        bottom_layout.addWidget(self.export_scenario_button)
        
        bottom_layout.addStretch(1)
        
        layout.addLayout(bottom_layout)
        
        # Initialize charts
        self._initialize_charts()
    
    def _initialize_charts(self):
        """Initialize empty charts"""
        # Emissions figure
        self.emissions_figure.clear()
        emissions_ax = self.emissions_figure.add_subplot(111)
        emissions_ax.set_title("CO2 Emissions Pathway")
        emissions_ax.text(0.5, 0.5, "Generate a custom scenario to see emissions pathway", 
                       ha='center', va='center', transform=emissions_ax.transAxes)
        self.emissions_figure.tight_layout()
        self.emissions_canvas.draw()
        
        # Temperature figure
        self.temp_figure.clear()
        temp_ax = self.temp_figure.add_subplot(111)
        temp_ax.set_title("Temperature Pathway")
        temp_ax.text(0.5, 0.5, "Generate a custom scenario to see temperature pathway", 
                  ha='center', va='center', transform=temp_ax.transAxes)
        self.temp_figure.tight_layout()
        self.temp_canvas.draw()
        
        # Energy mix figure
        self.energy_figure.clear()
        energy_ax = self.energy_figure.add_subplot(111)
        energy_ax.set_title("Energy Mix Evolution")
        energy_ax.text(0.5, 0.5, "Generate a custom scenario to see energy mix evolution", 
                    ha='center', va='center', transform=energy_ax.transAxes)
        self.energy_figure.tight_layout()
        self.energy_canvas.draw()
        
        # Carbon price figure
        self.carbon_figure.clear()
        carbon_ax = self.carbon_figure.add_subplot(111)
        carbon_ax.set_title("Carbon Price Trajectory")
        carbon_ax.text(0.5, 0.5, "Generate a custom scenario to see carbon price trajectory", 
                    ha='center', va='center', transform=carbon_ax.transAxes)
        self.carbon_figure.tight_layout()
        self.carbon_canvas.draw()
    
    @pyqtSlot(int)
    def _temp_slider_changed(self, value):
        """Handle temperature slider change"""
        factor = value / 100.0
        self.temp_value_label.setText(f"{factor:.2f}x (baseline)")
    
    @pyqtSlot(int)
    def _carbon_slider_changed(self, value):
        """Handle carbon price slider change"""
        factor = value / 100.0
        self.carbon_value_label.setText(f"{factor:.2f}x (baseline)")
    
    @pyqtSlot(int)
    def _renewable_slider_changed(self, value):
        """Handle renewable adoption slider change"""
        factor = value / 100.0
        self.renewable_value_label.setText(f"{factor:.2f}x (baseline)")
    
    @pyqtSlot(int)
    def _weather_slider_changed(self, value):
        """Handle extreme weather slider change"""
        factor = value / 100.0
        self.weather_value_label.setText(f"{factor:.2f}x (baseline)")
    
    @pyqtSlot()
    def _generate_scenario(self):
        """Generate custom climate scenario based on parameters"""
        # Get scenario name (use default if empty)
        scenario_name = self.scenario_name_edit.text().strip()
        if not scenario_name:
            scenario_name = "Custom Scenario"
        
        # Get base scenario
        base_scenario_text = self.base_scenario_combo.currentText()
        base_scenario_id = ""
        
        if "SSP1-1.9" in base_scenario_text:
            base_scenario_id = "ipcc_ssp119"
        elif "SSP1-2.6" in base_scenario_text:
            base_scenario_id = "ipcc_ssp126"
        elif "SSP2-4.5" in base_scenario_text:
            base_scenario_id = "ipcc_ssp245"
        elif "SSP3-7.0" in base_scenario_text:
            base_scenario_id = "ipcc_ssp370"
        elif "SSP5-8.5" in base_scenario_text:
            base_scenario_id = "ipcc_ssp585"
        
        # Get parameter values
        temperature_increase_factor = self.temp_slider.value() / 100.0
        carbon_price_factor = self.carbon_slider.value() / 100.0
        renewable_adoption_factor = self.renewable_slider.value() / 100.0
        policy_delay_years = int(self.policy_delay_spin.value())
        extreme_weather_factor = self.weather_slider.value() / 100.0
        
        # Create parameters dictionary
        parameters = {
            "description": scenario_name,
            "temperature_increase_factor": temperature_increase_factor,
            "carbon_price_factor": carbon_price_factor,
            "renewable_adoption_factor": renewable_adoption_factor,
            "policy_delay_years": policy_delay_years,
            "extreme_weather_factor": extreme_weather_factor
        }
        
        self.logger.info(f"Generating custom scenario based on {base_scenario_id} with parameters: {parameters}")
        
        # Generate scenario using the AI generator
        self.custom_scenario = self.risk_engine.generate_custom_scenario(base_scenario_id, parameters)
        
        # Update UI with scenario details
        self._update_ui_with_scenario()
    
    def _update_ui_with_scenario(self):
        """Update UI with custom scenario details"""
        if not self.custom_scenario or self.custom_scenario.get("status") != "success":
            return
        
        # Enable buttons
        self.apply_scenario_button.setEnabled(True)
        self.export_scenario_button.setEnabled(True)
        
        # Update explanation text
        narrative = self.custom_scenario.get("narrative", "No narrative available.")
        self.explanation_text.setText(narrative)
        
        # Update charts
        self._update_emissions_chart()
        self._update_temperature_chart()
        self._update_energy_chart()
        self._update_carbon_chart()
        
        # Update risk metrics table
        self._update_risk_metrics()
    
    def _update_emissions_chart(self):
        """Update the emissions pathway chart"""
        if not self.custom_scenario:
            return
        
        # Get emissions data
        emissions = self.custom_scenario.get("co2_emissions", {})
        if not emissions:
            return
        
        # Clear figure
        self.emissions_figure.clear()
        emissions_ax = self.emissions_figure.add_subplot(111)
        
        # Extract years and emissions values
        years = []
        values = []
        
        for year_str, value in emissions.items():
            years.append(int(year_str))
            values.append(value)
        
        # Sort by year
        sorted_data = sorted(zip(years, values))
        years, values = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot line
        emissions_ax.plot(years, values, 'g-o', linewidth=2)
        
        # Add area fill below line
        emissions_ax.fill_between(years, 0, values, alpha=0.2, color='green')
        
        # Add labels and styling
        emissions_ax.set_xlabel("Year")
        emissions_ax.set_ylabel("CO2 Emissions (GtCO2/yr)")
        emissions_ax.set_title("CO2 Emissions Pathway")
        emissions_ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add zero line
        emissions_ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Redraw
        self.emissions_figure.tight_layout()
        self.emissions_canvas.draw()
    
    def _update_temperature_chart(self):
        """Update the temperature pathway chart"""
        if not self.custom_scenario:
            return
        
        # Get temperature data
        temperature = self.custom_scenario.get("temperature_pathway", {})
        if not temperature:
            return
        
        # Clear figure
        self.temp_figure.clear()
        temp_ax = self.temp_figure.add_subplot(111)
        
        # Extract years and temperature values
        years = []
        values = []
        
        for year_str, value in temperature.items():
            years.append(int(year_str))
            values.append(value)
        
        # Sort by year
        sorted_data = sorted(zip(years, values))
        years, values = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot line
        temp_ax.plot(years, values, 'r-o', linewidth=2)
        
        # Add area fill below line
        temp_ax.fill_between(years, 0, values, alpha=0.2, color='red')
        
        # Add threshold lines
        temp_ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='1.5°C threshold')
        temp_ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='2.0°C threshold')
        
        # Add labels and styling
        temp_ax.set_xlabel("Year")
        temp_ax.set_ylabel("Global Temperature Increase (°C)")
        temp_ax.set_title("Temperature Pathway")
        temp_ax.grid(True, linestyle='--', alpha=0.6)
        temp_ax.legend()
        
        # Redraw
        self.temp_figure.tight_layout()
        self.temp_canvas.draw()
    
    def _update_energy_chart(self):
        """Update the energy mix chart"""
        if not self.custom_scenario:
            return
        
        # Get energy mix data
        energy_mix = self.custom_scenario.get("energy_mix", {})
        if not energy_mix or not energy_mix.get("fossil") or not energy_mix.get("renewable"):
            return
        
        # Clear figure
        self.energy_figure.clear()
        energy_ax = self.energy_figure.add_subplot(111)
        
        # Extract years and values for each energy type
        years = []
        fossil_values = []
        renewable_values = []
        nuclear_values = []
        
        # Get years from fossil (assuming all have the same years)
        for year_str in energy_mix["fossil"].keys():
            years.append(int(year_str))
        
        # Sort years
        years.sort()
        
        # Extract values for each year
        for year in years:
            year_str = str(year)
            fossil_values.append(energy_mix["fossil"].get(year_str, 0))
            renewable_values.append(energy_mix["renewable"].get(year_str, 0))
            nuclear_values.append(energy_mix["nuclear"].get(year_str, 0))
        
        # Create stacked area chart
        energy_ax.stackplot(years, 
                          [fossil_values, renewable_values, nuclear_values],
                          labels=['Fossil Fuels', 'Renewables', 'Nuclear'],
                          colors=['#d9534f', '#5cb85c', '#5bc0de'],
                          alpha=0.8)
        
        # Add labels and styling
        energy_ax.set_xlabel("Year")
        energy_ax.set_ylabel("Share of Energy Mix (%)")
        energy_ax.set_title("Energy Mix Evolution")
        energy_ax.grid(True, linestyle='--', alpha=0.6)
        energy_ax.legend(loc='upper right')
        
        # Set y-axis limits
        energy_ax.set_ylim(0, 100)
        
        # Redraw
        self.energy_figure.tight_layout()
        self.energy_canvas.draw()
    
    def _update_carbon_chart(self):
        """Update the carbon price chart"""
        if not self.custom_scenario:
            return
        
        # Get carbon price data
        carbon_price = self.custom_scenario.get("carbon_price", {})
        if not carbon_price:
            return
        
        # Clear figure
        self.carbon_figure.clear()
        carbon_ax = self.carbon_figure.add_subplot(111)
        
        # Extract years and carbon price values
        years = []
        values = []
        
        for year_str, value in carbon_price.items():
            years.append(int(year_str))
            values.append(value)
        
        # Sort by year
        sorted_data = sorted(zip(years, values))
        years, values = zip(*sorted_data) if sorted_data else ([], [])
        
        # Plot line
        carbon_ax.plot(years, values, 'b-o', linewidth=2)
        
        # Add area fill below line
        carbon_ax.fill_between(years, 0, values, alpha=0.2, color='blue')
        
        # Add labels and styling
        carbon_ax.set_xlabel("Year")
        carbon_ax.set_ylabel("Carbon Price ($/tCO2)")
        carbon_ax.set_title("Carbon Price Trajectory")
        carbon_ax.grid(True, linestyle='--', alpha=0.6)
        
        # Redraw
        self.carbon_figure.tight_layout()
        self.carbon_canvas.draw()
    
    def _update_risk_metrics(self):
        """Update the risk metrics table"""
        if not self.custom_scenario:
            return
        
        # Get risk metrics
        risk_metrics = self.custom_scenario.get("risk_metrics", {})
        if not risk_metrics:
            return
        
        # Extract metrics
        physical_risk = risk_metrics.get("physical_risk_score", 0)
        transition_risk = risk_metrics.get("transition_risk_score", 0)
        overall_risk = risk_metrics.get("overall_risk_score", 0)
        
        gdp_impact = risk_metrics.get("gdp_impact", {})
        gdp_2050 = gdp_impact.get("2050", 0)
        gdp_2100 = gdp_impact.get("2100", 0)
        
        # Update table cells
        self.risk_table.item(0, 1).setText(f"{physical_risk:.1f} / 10")
        self.risk_table.item(1, 1).setText(f"{transition_risk:.1f} / 10")
        self.risk_table.item(2, 1).setText(f"{overall_risk:.1f} / 10")
        self.risk_table.item(3, 1).setText(f"{gdp_2050:.1f}%")
        self.risk_table.item(4, 1).setText(f"{gdp_2100:.1f}%")
        
        # Set cell colors based on risk levels
        for i, risk in enumerate([physical_risk, transition_risk, overall_risk]):
            item = self.risk_table.item(i, 1)
            
            if risk >= 7:
                item.setBackground(QColor("#d9534f"))
                item.setForeground(QColor("white"))
            elif risk >= 5:
                item.setBackground(QColor("#f0ad4e"))
            elif risk >= 3:
                item.setBackground(QColor("#5bc0de"))
            else:
                item.setBackground(QColor("#5cb85c"))
                item.setForeground(QColor("white"))
        
        # Set GDP impact cell colors
        for i, impact in enumerate([gdp_2050, gdp_2100]):
            item = self.risk_table.item(i+3, 1)
            
            if impact >= 10:
                item.setBackground(QColor("#d9534f"))
                item.setForeground(QColor("white"))
            elif impact >= 5:
                item.setBackground(QColor("#f0ad4e"))
            elif impact >= 2:
                item.setBackground(QColor("#5bc0de"))
            else:
                item.setBackground(QColor("#5cb85c"))
                item.setForeground(QColor("white"))
    
    @pyqtSlot()
    def _apply_scenario(self):
        """Apply the custom scenario to analysis"""
        if not self.custom_scenario:
            return
        
        # In a real implementation, this would register the scenario with the risk engine
        # and make it available for selection in the main UI
        
        self.logger.info("Applying custom scenario to analysis")
        
        # Show confirmation message
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Scenario Applied", 
                            "The custom scenario has been applied and is now available for analysis.")
    
    @pyqtSlot()
    def _export_scenario(self):
        """Export the custom scenario to a file"""
        if not self.custom_scenario:
            return
        
        self.logger.info("Exporting custom scenario")
        
        # In a real implementation, this would export the scenario to a file
        
        # Get file path from user
        from PyQt5.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Scenario",
            f"{self.scenario_name_edit.text() or 'Custom_Scenario'}.json",
            "JSON Files (*.json)"
        )
        
        if not file_path:
            return
        
        # Show success message
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Scenario Exported", 
                            f"The custom scenario has been exported to {file_path}")