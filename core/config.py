"""
Configuration management for the Climate Risk Integration Platform.
Handles loading settings, API keys, and user preferences.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("config")
        
        # If no config path provided, use default locations
        if config_path is None:
            # Look for config in the standard locations
            self.config_path = self._find_config_file()
        else:
            self.config_path = Path(config_path)
        
        # Load configuration
        self.settings = self._load_config()
        
        # Cache for API credentials
        self._api_credentials = {}

    def _find_config_file(self) -> Path:
        """Search for configuration file in standard locations"""
        # Check for config specified by environment variable
        if "CLIMATE_RISK_CONFIG" in os.environ:
            config_path = Path(os.environ["CLIMATE_RISK_CONFIG"])
            if config_path.exists():
                self.logger.info(f"Using config from environment: {config_path}")
                return config_path
        
        # Check in current directory
        local_config = Path("config.yaml")
        if local_config.exists():
            self.logger.info(f"Using local config: {local_config}")
            return local_config
            
        # Check in user config directory
        user_config_dir = Path.home() / ".config" / "climate_risk"
        user_config = user_config_dir / "config.yaml"
        if user_config.exists():
            self.logger.info(f"Using user config: {user_config}")
            return user_config
            
        # If no config found, use default config in package
        package_dir = Path(__file__).parent.parent
        default_config = package_dir / "config" / "default_config.yaml"
        self.logger.info(f"Using default config: {default_config}")
        return default_config

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            self.logger.warning("Falling back to minimal default configuration")
            # Return minimal working configuration
            return {
                "data_sources": {
                    "climate": {
                        "nasa_power": {"enabled": True, "api_key": None},
                        "noaa_cdo": {"enabled": True, "api_key": None},
                        "copernicus": {"enabled": False, "api_key": None}
                    },
                    "financial": {
                        "yahoo_finance": {"enabled": True},
                        "fred": {"enabled": True, "api_key": None},
                        "alpha_vantage": {"enabled": False, "api_key": None}
                    },
                    "energy": {
                        "eia": {"enabled": True, "api_key": None},
                        "global_energy_monitor": {"enabled": False}
                    }
                },
                "ui": {
                    "theme": "system",
                    "default_view": "dashboard",
                    "map_provider": "open_street_map"
                },
                "risk_engine": {
                    "default_scenarios": ["ipcc_ssp245", "ipcc_ssp370", "ipcc_ssp585"],
                    "monte_carlo_iterations": 1000,
                    "confidence_interval": 0.95
                },
                "cache": {
                    "enabled": True,
                    "max_age_hours": 24,
                    "location": "~/.cache/climate_risk"
                }
            }

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation
        Example: config.get("data_sources.climate.nasa_power.enabled")
        """
        keys = key_path.split(".")
        value = self.settings
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            self.logger.debug(f"Config key not found: {key_path}, using default: {default}")
            return default
            
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value using dot notation and save changes
        Example: config.set("ui.theme", "dark")
        """
        keys = key_path.split(".")
        config_section = self.settings
        
        # Navigate to the correct nested dictionary
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the value
        config_section[keys[-1]] = value
        
        # Save the updated configuration
        self._save_config()
    
    def _save_config(self) -> None:
        """Save current configuration to disk"""
        # Make sure parent directories exist
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.config_path, "w") as f:
                yaml.dump(self.settings, f, default_flow_style=False)
            self.logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def get_api_credentials(self, service_name: str) -> Dict[str, str]:
        """
        Get API credentials for a service, looking in environment variables first,
        then in the configuration, and finally prompting the user if needed
        """
        if service_name in self._api_credentials:
            return self._api_credentials[service_name]
            
        # First check for credentials in environment variables
        env_var_name = f"{service_name.upper()}_API_KEY"
        if env_var_name in os.environ:
            self._api_credentials[service_name] = {"api_key": os.environ[env_var_name]}
            return self._api_credentials[service_name]
            
        # Then check in configuration
        service_path = self._find_service_in_config(service_name)
        if service_path and self.get(f"{service_path}.api_key"):
            self._api_credentials[service_name] = {"api_key": self.get(f"{service_path}.api_key")}
            return self._api_credentials[service_name]
            
        # Return empty dict if not found (caller will need to handle missing credentials)
        return {}
    
    def _find_service_in_config(self, service_name: str) -> Optional[str]:
        """Find the path to a service in the configuration"""
        # Common service locations in config
        sections = ["data_sources.climate", "data_sources.financial", "data_sources.energy"]
        
        for section in sections:
            try:
                if service_name in self.get(section, {}):
                    return f"{section}.{service_name}"
            except (TypeError, KeyError):
                continue
                
        return None