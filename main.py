#!/usr/bin/env python3
"""
Climate Risk Integration Platform
Main application entry point

This platform provides comprehensive climate risk assessment for the oil and gas sector,
integrating physical and transition risks with financial institution exposure analysis.
"""

import sys
import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, QObject

# Load environment variables from .env file if present
load_dotenv()

# Add the project root to the Python path to allow for absolute imports
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)

# Import local modules after setting up the Python path
from ui.main_window import MainWindow
from core.config import Config
from core.data_manager import DataManager
from core.risk_engine import RiskEngine
from utils.logger import setup_logger


class AppController(QObject):
    """Main application controller that coordinates data, models, and views"""
    
    def __init__(self):
        super().__init__()
        # Set up logging
        self.logger = setup_logger("app_controller")
        self.logger.info("Initializing Climate Risk Integration Platform")
        
        # Initialize configuration
        self.config = Config()
        
        # Create core components
        self.data_manager = DataManager(self.config)
        self.risk_engine = RiskEngine(self.data_manager)
        
        # Thread for background tasks
        self.worker_thread = QThread()
        self.data_manager.moveToThread(self.worker_thread)
        self.worker_thread.start()
        
        # Initialize UI last so it can connect to backend components
        self.main_window = None
    
    def start(self):
        """Initialize and show the main UI window"""
        self.logger.info("Starting application UI")
        self.main_window = MainWindow(self.data_manager, self.risk_engine, self.config)
        self.main_window.show()
    
    def shutdown(self):
        """Clean shutdown of all components"""
        self.logger.info("Shutting down application")
        if self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.data_manager.shutdown()
        self.risk_engine.shutdown()


def main():
    """Main application entry point"""
    # Create the QApplication instance
    app = QApplication(sys.argv)
    app.setApplicationName("Climate Risk Integration Platform")
    app.setOrganizationName("QuantClimate")
    
    # Create and start the application controller
    controller = AppController()
    controller.start()
    
    # Exit cleanly on app quit
    exit_code = app.exec_()
    controller.shutdown()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()