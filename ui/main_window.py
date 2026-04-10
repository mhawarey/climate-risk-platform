"""
Main window for the Climate Risk Integration Platform.
Implements the primary UI components and layout.
"""

import os
import sys
import logging
import time
from typing import Dict, List, Any, Optional

from PyQt5.QtWidgets import (QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QPushButton, QComboBox, QFrame, QSplitter,
                            QTreeView, QTableView, QHeaderView, QStatusBar, QToolBar,
                            QAction, QFileDialog, QMessageBox, QMenu, QProgressBar,
                            QApplication, QDialog, QDialogButtonBox)
from PyQt5.QtCore import Qt, QSize, pyqtSlot, QModelIndex, QThread, QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem, QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.data_manager import DataManager
from core.risk_engine import RiskEngine
from core.config import Config
from ui.dashboard_tab import DashboardTab
from ui.physical_risk_tab import PhysicalRiskTab
from ui.transition_risk_tab import TransitionRiskTab
from ui.financial_institution_tab import FinancialInstitutionTab
from ui.scenario_tab import ScenarioTab


class MainWindow(QMainWindow):
    """Main application window for the Climate Risk Integration Platform"""
    
    def __init__(self, data_manager: DataManager, risk_engine: RiskEngine, config: Config):
        super().__init__()
        
        self.logger = logging.getLogger("main_window")
        self.data_manager = data_manager
        self.risk_engine = risk_engine
        self.config = config
        
        # Initialize state
        self.selected_company = None
        self.selected_institution = None
        self.selected_scenario = None
        self.include_physical = True
        self.include_transition = True
        
        # Set up the UI
        self._init_ui()
        
        # Connect signals
        self._connect_signals()
        
        self.logger.info("MainWindow initialized")
    
    def _init_ui(self):
        """Initialize the user interface"""
        # Set window properties
        self.setWindowTitle("Climate Risk Integration Platform")
        self.setGeometry(100, 100, 1280, 800)
        
        # Set icon
        # self.setWindowIcon(QIcon("icons/app_icon.png"))
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create entity selection bar
        entity_bar = self._create_entity_selection_bar()
        main_layout.addLayout(entity_bar)
        
        # Create main tab widget
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self._create_tabs()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar to status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # Set default status
        self.status_bar.showMessage("Ready")
    
    def _create_toolbar(self):
        """Create the application toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Add actions
        # Refresh data action
        refresh_action = QAction("Refresh Data", self)
        refresh_action.setStatusTip("Refresh data from sources")
        refresh_action.triggered.connect(self._refresh_data)
        toolbar.addAction(refresh_action)
        
        toolbar.addSeparator()
        
        # Export results action
        export_action = QAction("Export Results", self)
        export_action.setStatusTip("Export analysis results")
        export_action.triggered.connect(self._export_results)
        toolbar.addAction(export_action)
        
        # Settings action
        settings_action = QAction("Settings", self)
        settings_action.setStatusTip("Application settings")
        settings_action.triggered.connect(self._show_settings)
        toolbar.addAction(settings_action)
    
    def _create_entity_selection_bar(self):
        """Create the entity selection bar"""
        layout = QHBoxLayout()
        
        # Company selection
        layout.addWidget(QLabel("Company:"))
        self.company_combo = QComboBox()
        self.company_combo.setMinimumWidth(200)
        layout.addWidget(self.company_combo)
        
        # Financial institution selection
        layout.addWidget(QLabel("Financial Institution:"))
        self.institution_combo = QComboBox()
        self.institution_combo.setMinimumWidth(200)
        layout.addWidget(self.institution_combo)
        
        # Scenario selection
        layout.addWidget(QLabel("Scenario:"))
        self.scenario_combo = QComboBox()
        self.scenario_combo.setMinimumWidth(150)
        layout.addWidget(self.scenario_combo)
        
        # Risk type selection
        layout.addWidget(QLabel("Risk Type:"))
        self.risk_type_combo = QComboBox()
        self.risk_type_combo.addItems(["Both", "Physical", "Transition"])
        layout.addWidget(self.risk_type_combo)
        
        # Add spacer
        layout.addStretch(1)
        
        # Calculate button
        self.calculate_button = QPushButton("Calculate Risk")
        self.calculate_button.clicked.connect(self._calculate_selected_risk)
        layout.addWidget(self.calculate_button)
        
        return layout
    
    def _create_tabs(self):
        """Create the main application tabs"""
        # Dashboard tab
        self.dashboard_tab = DashboardTab(self.data_manager, self.risk_engine)
        self.tab_widget.addTab(self.dashboard_tab, "Dashboard")
        
        # Physical Risk tab
        self.physical_tab = PhysicalRiskTab(self.data_manager, self.risk_engine)
        self.tab_widget.addTab(self.physical_tab, "Physical Risk")
        
        # Transition Risk tab
        self.transition_tab = TransitionRiskTab(self.data_manager, self.risk_engine)
        self.tab_widget.addTab(self.transition_tab, "Transition Risk")
        
        # Financial Institution tab
        self.financial_tab = FinancialInstitutionTab(self.data_manager, self.risk_engine)
        self.tab_widget.addTab(self.financial_tab, "Financial Institution")
        
        # Scenario Analysis tab
        self.scenario_tab = ScenarioTab(self.data_manager, self.risk_engine)
        self.tab_widget.addTab(self.scenario_tab, "Scenario Analysis")
    
    def _connect_signals(self):
        """Connect signals to slots"""
        # Connect combo box signals
        self.company_combo.currentIndexChanged.connect(self._company_selected)
        self.institution_combo.currentIndexChanged.connect(self._institution_selected)
        self.scenario_combo.currentIndexChanged.connect(self._scenario_selected)
        self.risk_type_combo.currentIndexChanged.connect(self._risk_type_selected)
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self._tab_changed)
        
        # Connect data manager signals
        self.data_manager.data_updated.connect(self._handle_data_updated)
        self.data_manager.fetch_completed.connect(self._handle_fetch_completed)
        self.data_manager.progress_updated.connect(self._handle_progress_updated)
        
        # Connect risk engine signals
        self.risk_engine.calculation_progress.connect(self._handle_calculation_progress)
        self.risk_engine.calculation_complete.connect(self._handle_calculation_complete)
    
    def showEvent(self, event):
        """Handle window show event"""
        super().showEvent(event)
        # Load initial data
        self._load_initial_data()
    
    def _load_initial_data(self):
        """Load initial data for combo boxes"""
        self.status_bar.showMessage("Loading initial data...")
        
        # Load company list
        self._load_companies()
        
        # Load financial institution list
        self._load_institutions()
        
        # Load scenarios
        self._load_scenarios()
        
        self.status_bar.showMessage("Ready")
    
    def _load_companies(self):
        """Load companies into combo box"""
        # Clear combo box
        self.company_combo.clear()
        
        # Add empty option
        self.company_combo.addItem("-- Select Company --", None)
        
        # Load companies in background
        # For demo, we'll use a simple list
        companies = [
            {"ticker": "XOM", "name": "ExxonMobil"},
            {"ticker": "CVX", "name": "Chevron"},
            {"ticker": "BP", "name": "BP plc"},
            {"ticker": "SHEL", "name": "Shell plc"},
            {"ticker": "COP", "name": "ConocoPhillips"},
            {"ticker": "EOG", "name": "EOG Resources"},
            {"ticker": "OXY", "name": "Occidental Petroleum"},
            {"ticker": "MRO", "name": "Marathon Oil"},
            {"ticker": "APA", "name": "Apache Corporation"},
            {"ticker": "DVN", "name": "Devon Energy"}
        ]
        
        # Add companies to combo box
        for company in companies:
            display_text = f"{company['ticker']} - {company['name']}"
            self.company_combo.addItem(display_text, company['ticker'])
    
    def _load_institutions(self):
        """Load financial institutions into combo box"""
        # Clear combo box
        self.institution_combo.clear()
        
        # Add empty option
        self.institution_combo.addItem("-- Select Institution --", None)
        
        # Load institutions in background
        # For demo, we'll use a simple list
        institutions = [
            {"id": "JPM", "name": "JPMorgan Chase"},
            {"id": "BAC", "name": "Bank of America"},
            {"id": "WFC", "name": "Wells Fargo"},
            {"id": "C", "name": "Citigroup"},
            {"id": "GS", "name": "Goldman Sachs"},
            {"id": "MS", "name": "Morgan Stanley"},
            {"id": "BLK", "name": "BlackRock"},
            {"id": "BK", "name": "Bank of New York Mellon"},
            {"id": "STT", "name": "State Street"},
            {"id": "PNC", "name": "PNC Financial Services"}
        ]
        
        # Add institutions to combo box
        for institution in institutions:
            display_text = f"{institution['id']} - {institution['name']}"
            self.institution_combo.addItem(display_text, institution['id'])
    
    def _load_scenarios(self):
        """Load scenarios into combo box"""
        # Clear combo box
        self.scenario_combo.clear()
        
        # Add "All Scenarios" option
        self.scenario_combo.addItem("All Scenarios", None)
        
        # Add standard scenarios
        scenarios = [
            {"id": "ipcc_ssp119", "name": "Low Emissions (1.5°C)"},
            {"id": "ipcc_ssp126", "name": "Low-Medium Emissions (2°C)"},
            {"id": "ipcc_ssp245", "name": "Medium Emissions (2.5-3°C)"},
            {"id": "ipcc_ssp370", "name": "Medium-High Emissions (3-4°C)"},
            {"id": "ipcc_ssp585", "name": "High Emissions (4-5°C)"}
        ]
        
        # Add scenarios to combo box
        for scenario in scenarios:
            display_text = scenario['name']
            self.scenario_combo.addItem(display_text, scenario['id'])
    
    @pyqtSlot(int)
    def _company_selected(self, index):
        """Handle company selection"""
        # Get selected company ID
        company_id = self.company_combo.itemData(index)
        self.selected_company = company_id
        
        # Update tabs with selected company
        if company_id:
            self.logger.info(f"Selected company: {company_id}")
            self.dashboard_tab.set_company(company_id)
            self.physical_tab.set_company(company_id)
            self.transition_tab.set_company(company_id)
            
            # Enable calculate button
            self.calculate_button.setEnabled(True)
        else:
            # Disable calculate button if no company selected
            self.calculate_button.setEnabled(False)
    
    @pyqtSlot(int)
    def _institution_selected(self, index):
        """Handle financial institution selection"""
        # Get selected institution ID
        institution_id = self.institution_combo.itemData(index)
        self.selected_institution = institution_id
        
        # Update financial institution tab
        if institution_id:
            self.logger.info(f"Selected institution: {institution_id}")
            self.financial_tab.set_institution(institution_id)
    
    @pyqtSlot(int)
    def _scenario_selected(self, index):
        """Handle scenario selection"""
        # Get selected scenario ID
        scenario_id = self.scenario_combo.itemData(index)
        self.selected_scenario = scenario_id
        
        # Update tabs with selected scenario
        self.logger.info(f"Selected scenario: {scenario_id}")
        
        if self.selected_company:
            # If specific scenario selected, update tabs
            if scenario_id:
                self.physical_tab.set_scenario(scenario_id)
                self.transition_tab.set_scenario(scenario_id)
            else:
                # "All Scenarios" selected, clear specific scenario
                self.physical_tab.set_scenario(None)
                self.transition_tab.set_scenario(None)
    
    @pyqtSlot(int)
    def _risk_type_selected(self, index):
        """Handle risk type selection"""
        # Get selected risk type
        risk_types = ["Both", "Physical", "Transition"]
        selected_type = risk_types[index]
        
        self.logger.info(f"Selected risk type: {selected_type}")
        
        # Update calculation parameters
        if selected_type == "Physical":
            self.include_physical = True
            self.include_transition = False
        elif selected_type == "Transition":
            self.include_physical = False
            self.include_transition = True
        else:  # Both
            self.include_physical = True
            self.include_transition = True
    
    @pyqtSlot()
    def _calculate_selected_risk(self):
        """Calculate risk for selected parameters"""
        if not self.selected_company:
            QMessageBox.warning(self, "Selection Required", "Please select a company first.")
            return
        
        # Get selected scenario(s)
        scenarios = None
        if self.selected_scenario:
            scenarios = [self.selected_scenario]
        
        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Update status
        self.status_bar.showMessage("Calculating risk...")
        
        # Disable calculate button during calculation
        self.calculate_button.setEnabled(False)
        
        # Calculate risk
        self.risk_engine.calculate_company_risk(
            self.selected_company,
            scenarios=scenarios,
            include_physical=self.include_physical,
            include_transition=self.include_transition
        )
    
    @pyqtSlot(int)
    def _tab_changed(self, index):
        """Handle tab change"""
        # Get current tab
        current_tab = self.tab_widget.widget(index)
        tab_name = self.tab_widget.tabText(index)
        
        self.logger.info(f"Switched to tab: {tab_name}")
        
        # Update UI based on selected tab
        if tab_name == "Financial Institution":
            # Enable institution combo, disable company combo
            self.institution_combo.setEnabled(True)
            self.company_combo.setEnabled(False)
        else:
            # Enable company combo, disable institution combo
            self.company_combo.setEnabled(True)
            self.institution_combo.setEnabled(index == 3)  # Only enable for Financial Institution tab
    
    @pyqtSlot()
    def _refresh_data(self):
        """Refresh data from sources"""
        # Show confirmation dialog
        result = QMessageBox.question(
            self,
            "Refresh Data",
            "This will clear the cache and refresh all data from sources. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if result == QMessageBox.Yes:
            # Show progress bar
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Update status
            self.status_bar.showMessage("Refreshing data...")
            
            # Clear cache and reload data
            # In a real implementation, this would trigger data reloading
            # For demo, we'll just simulate it
            self.progress_bar.setValue(50)
            
            # Reload current view
            current_index = self.tab_widget.currentIndex()
            current_tab = self.tab_widget.widget(current_index)
            
            # Update status
            self.status_bar.showMessage("Data refreshed", 3000)
            self.progress_bar.setVisible(False)
    
    @pyqtSlot()
    def _export_results(self):
        """Export analysis results"""
        # Get current tab
        current_index = self.tab_widget.currentIndex()
        current_tab = self.tab_widget.widget(current_index)
        tab_name = self.tab_widget.tabText(current_index)
        
        # Ask for export format
        format_dialog = QDialog(self)
        format_dialog.setWindowTitle("Export Format")
        format_layout = QVBoxLayout(format_dialog)
        format_layout.addWidget(QLabel("Select export format:"))
        format_combo = QComboBox()
        formats = ["Excel (.xlsx)", "CSV (.csv)", "PDF Report (.pdf)"]
        format_combo.addItems(formats)
        format_layout.addWidget(format_combo)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(format_dialog.accept)
        buttons.rejected.connect(format_dialog.reject)
        format_layout.addWidget(buttons)
        
        if format_dialog.exec_() != QDialog.Accepted:
            return
        
        selected_format = format_combo.currentText()
        
        # Ask for save location
        if "Excel" in selected_format:
            file_filter = "Excel Files (*.xlsx)"
            default_ext = ".xlsx"
        elif "CSV" in selected_format:
            file_filter = "CSV Files (*.csv)"
            default_ext = ".csv"
        else:  # PDF
            file_filter = "PDF Files (*.pdf)"
            default_ext = ".pdf"
        
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Export",
            f"{tab_name}_Export{default_ext}",
            file_filter
        )
        
        if not save_path:
            return
        
        # Show progress during export
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage(f"Exporting to {selected_format}...")
        
        # Simulate export process
        for i in range(1, 101):
            self.progress_bar.setValue(i)
            QApplication.processEvents()
            time.sleep(0.01)
        
        # Show success message
        self.status_bar.showMessage(f"Export complete: {save_path}", 5000)
        self.progress_bar.setVisible(False)
        
        QMessageBox.information(self, "Export Complete", f"Results exported to:\n{save_path}")
    
    @pyqtSlot()
    def _show_settings(self):
        """Show settings dialog"""
        # In a real implementation, this would open a settings dialog
        QMessageBox.information(self, "Settings", "Settings dialog would be shown here.")
    
    @pyqtSlot(str, object)
    def _handle_data_updated(self, data_type, data):
        """Handle data update signal from data manager"""
        self.logger.debug(f"Data updated: {data_type}")
        
        # Update current tab if relevant
        current_index = self.tab_widget.currentIndex()
        current_tab = self.tab_widget.widget(current_index)
        
        # Call update method if it exists
        if hasattr(current_tab, "update_data"):
            current_tab.update_data(data_type, data)
    
    @pyqtSlot(str, bool, str)
    def _handle_fetch_completed(self, source, success, message):
        """Handle fetch completion signal from data manager"""
        if success:
            self.logger.info(f"Fetch completed: {source}")
            self.status_bar.showMessage(f"Data updated: {source}", 3000)
        else:
            self.logger.error(f"Fetch failed: {source} - {message}")
            self.status_bar.showMessage(f"Error: {message}", 5000)
    
    @pyqtSlot(str, int, int)
    def _handle_progress_updated(self, source, current, total):
        """Handle progress update signal from data manager"""
        if total > 0:
            percent = min(100, int(current * 100 / total))
            self.progress_bar.setValue(percent)
            
            if current >= total:
                # Hide progress bar after completion
                QTimer.singleShot(1000, lambda: self.progress_bar.setVisible(False))
    
    @pyqtSlot(str, int, int)
    def _handle_calculation_progress(self, task, current, total):
        """Handle calculation progress signal from risk engine"""
        if total > 0:
            percent = min(100, int(current * 100 / total))
            self.progress_bar.setValue(percent)
            self.status_bar.showMessage(f"Calculating {task}: {current}/{total}...")
    
    @pyqtSlot(str, object)
    def _handle_calculation_complete(self, task, results):
        """Handle calculation completion signal from risk engine"""
        self.logger.info(f"Calculation complete: {task}")
        self.status_bar.showMessage(f"Calculation complete: {task}", 3000)
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
        
        # Re-enable calculate button
        self.calculate_button.setEnabled(True)
        
        # Update tabs with results
        if task == "company_risk":
            company_id = results.get("company_id")
            
            # Update dashboard
            self.dashboard_tab.update_risk_results(results)
            
            # Update risk tabs
            if "physical_risks" in results:
                self.physical_tab.update_risk_results(results)
            
            if "transition_risks" in results:
                self.transition_tab.update_risk_results(results)
            
            # Switch to dashboard tab to show overview
            self.tab_widget.setCurrentIndex(0)
        
        elif task == "institution_exposure":
            # Update financial institution tab
            self.financial_tab.update_exposure_results(results)
        
        elif task == "portfolio_optimization":
            # Update financial institution tab with optimization results
            self.financial_tab.update_optimization_results(results)