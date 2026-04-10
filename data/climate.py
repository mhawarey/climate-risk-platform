"""
Climate data sources for the Climate Risk Integration Platform.
Handles acquisition of climate data from various public sources.
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import random

import pandas as pd
import numpy as np
import requests

from core.config import Config
from utils.cache import Cache


class ClimateDataSource(ABC):
    """
    Abstract base class for climate data sources
    
    This defines the interface that all climate data sources must implement.
    """
    
    def __init__(self, name: str, config: Config, cache: Cache):
        self.name = name
        self.config = config
        self.cache = cache
        self.logger = logging.getLogger(f"climate.{name}")
        self.api_credentials = self._get_credentials()
        
        self.logger.info(f"Initialized climate data source: {name}")
    
    @staticmethod
    def create(source_name: str, config: Config, cache: Cache) -> 'ClimateDataSource':
        """Factory method to create climate data sources"""
        if source_name == "nasa_power":
            return NASAPowerDataSource(config, cache)
        elif source_name == "noaa_cdo":
            return NOAAClimateDataSource(config, cache)
        elif source_name == "copernicus":
            return CopernicusClimateDataSource(config, cache)
        else:
            raise ValueError(f"Unknown climate data source: {source_name}")
    
    def _get_credentials(self) -> Dict[str, str]:
        """Get API credentials for this data source"""
        return self.config.get_api_credentials(self.name)
    
    @abstractmethod
    def get_data(self, 
                data_type: str,
                location: Union[str, Dict[str, float]], 
                start_date: datetime,
                end_date: datetime,
                scenario: Optional[str] = None,
                variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get climate data for the specified parameters
        
        Args:
            data_type: Type of climate data to retrieve
            location: Location identifier or coordinates
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            scenario: Climate scenario (for projection data)
            variables: Specific variables to retrieve
            
        Returns:
            DataFrame with the requested climate data
        """
        pass
    
    def _location_to_str(self, location: Union[str, Dict[str, float]]) -> str:
        """Convert location to string for cache keys"""
        if isinstance(location, str):
            return location
        elif isinstance(location, dict):
            return f"{location.get('lat', 0)}_{location.get('lon', 0)}"
        return str(location)


class NASAPowerDataSource(ClimateDataSource):
    """
    Climate data source using NASA POWER API
    
    POWER (Prediction of Worldwide Energy Resources) provides solar and meteorological data
    """
    
    def __init__(self, config: Config, cache: Cache):
        super().__init__("nasa_power", config, cache)
        self.base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def get_data(self, 
                data_type: str,
                location: Union[str, Dict[str, float]], 
                start_date: datetime,
                end_date: datetime,
                scenario: Optional[str] = None,
                variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Get climate data from NASA POWER"""
        # Convert location to coordinates
        if isinstance(location, str):
            # In a real implementation, this would geocode the location
            # For demo, we'll use placeholder coordinates
            coords = {"lat": 40.7128, "lon": -74.0060}  # New York City
        else:
            coords = location
        
        # Map data_type to NASA POWER parameters
        # In a real implementation, this would be more comprehensive
        param_mapping = {
            "temperature": ["T2M", "T2M_MAX", "T2M_MIN"],
            "precipitation": ["PRECTOT"],
            "solar": ["ALLSKY_SFC_SW_DWN"],
            "humidity": ["RH2M"],
            "wind": ["WS10M"],
            "extreme_heat": ["T2M_MAX"]
        }
        
        # If specific variables provided, use those instead
        if not variables:
            variables = param_mapping.get(data_type, ["T2M"])
        
        # For demo purposes, we'll generate synthetic data for NASA POWER
        # In a real implementation, this would call the actual API
        return self._generate_synthetic_data(data_type, coords, start_date, end_date, variables)
    
    def _call_api(self, coords: Dict[str, float], start_date: datetime, end_date: datetime, 
                 parameters: List[str]) -> Dict[str, Any]:
        """
        Call the NASA POWER API
        
        In a real implementation, this would make an actual API request.
        For the demo, we'll return a placeholder response.
        """
        # In a real implementation, this would be:
        # params = {
        #     "start": start_date.strftime("%Y%m%d"),
        #     "end": end_date.strftime("%Y%m%d"),
        #     "latitude": coords["lat"],
        #     "longitude": coords["lon"],
        #     "parameters": ",".join(parameters),
        #     "community": "RE",
        #     "format": "JSON"
        # }
        # response = requests.get(self.base_url, params=params)
        # return response.json()
        
        self.logger.info(f"Simulating NASA POWER API call for {parameters}")
        # For demo, simulate API latency
        time.sleep(0.5)
        
        # Return placeholder response
        return {
            "properties": {
                "parameter": {param: {"values": []} for param in parameters}
            }
        }
    
    def _generate_synthetic_data(self, data_type: str, coords: Dict[str, float], 
                               start_date: datetime, end_date: datetime, 
                               variables: List[str]) -> pd.DataFrame:
        """Generate synthetic climate data for demonstration purposes"""
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with dates
        df = pd.DataFrame(index=dates)
        df.index.name = 'date'
        
        # Generate data based on data_type
        if data_type == "temperature":
            # Base temperature based on latitude (crude approximation)
            lat = coords["lat"]
            base_temp = 25 - abs(lat) * 0.5  # Higher latitudes = colder
            
            # Add yearly cycle
            for date in dates:
                # Day of year normalized to -1 to 1
                day_factor = np.sin((date.dayofyear / 365) * 2 * np.pi - np.pi/2)
                yearly_cycle = day_factor * 10  # +/- 10 degrees seasonal variation
                
                # Add some random variation
                np.random.seed(int(date.timestamp()) % 10000)
                daily_variation = np.random.normal(0, 3)
                
                # Calculate temperatures
                mean_temp = base_temp + yearly_cycle + daily_variation
                max_temp = mean_temp + np.random.uniform(2, 5)
                min_temp = mean_temp - np.random.uniform(2, 5)
                
                # Add to dataframe
                if "T2M" in variables:
                    df.loc[date, "T2M"] = mean_temp
                if "T2M_MAX" in variables:
                    df.loc[date, "T2M_MAX"] = max_temp
                if "T2M_MIN" in variables:
                    df.loc[date, "T2M_MIN"] = min_temp
        
        elif data_type == "precipitation":
            # Base precipitation based on latitude (crude approximation)
            lat = coords["lat"]
            if abs(lat) < 10:
                # Tropical
                base_precip = 6  # mm/day
                variance = 8
            elif abs(lat) < 40:
                # Temperate
                base_precip = 3
                variance = 4
            else:
                # Polar
                base_precip = 1
                variance = 2
            
            # Add yearly cycle
            for date in dates:
                # Seasonality factor
                if lat >= 0:  # Northern hemisphere
                    season_factor = np.sin((date.dayofyear / 365) * 2 * np.pi)
                else:  # Southern hemisphere
                    season_factor = np.sin((date.dayofyear / 365) * 2 * np.pi + np.pi)
                
                # More rain in summer
                seasonal_precip = base_precip + season_factor * base_precip * 0.5
                
                # Add randomness - precipitation is often 0 with occasional higher values
                np.random.seed(int(date.timestamp()) % 10000)
                random_factor = np.random.exponential(1)
                
                # Some days have no rain
                if np.random.random() < 0.7:
                    daily_precip = seasonal_precip * random_factor
                else:
                    daily_precip = 0
                
                # Add to dataframe
                if "PRECTOT" in variables:
                    df.loc[date, "PRECTOT"] = daily_precip
        
        elif data_type == "solar":
            # Base solar radiation based on latitude (crude approximation)
            lat = coords["lat"]
            base_solar = 20 - abs(lat) * 0.2  # Higher latitudes = less solar
            
            # Add yearly cycle
            for date in dates:
                # Day of year normalized to -1 to 1
                day_factor = np.sin((date.dayofyear / 365) * 2 * np.pi - np.pi/2)
                yearly_cycle = day_factor * 10  # +/- 10 units seasonal variation
                
                # Add some random variation for weather
                np.random.seed(int(date.timestamp()) % 10000)
                daily_variation = np.random.normal(0, 3)
                
                # Calculate solar radiation
                solar_rad = max(0, base_solar + yearly_cycle + daily_variation)
                
                # Add to dataframe
                if "ALLSKY_SFC_SW_DWN" in variables:
                    df.loc[date, "ALLSKY_SFC_SW_DWN"] = solar_rad
        
        # Add metadata columns
        df["latitude"] = coords["lat"]
        df["longitude"] = coords["lon"]
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Convert date column to string for JSON serialization
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        
        # Rename columns to be more user-friendly
        column_names = {
            "T2M": "temperature_avg",
            "T2M_MAX": "temperature_max",
            "T2M_MIN": "temperature_min",
            "PRECTOT": "precipitation",
            "ALLSKY_SFC_SW_DWN": "solar_radiation"
        }
        
        df = df.rename(columns={col: column_names.get(col, col) for col in df.columns})
        
        return df


class NOAAClimateDataSource(ClimateDataSource):
    """
    Climate data source using NOAA Climate Data Online API
    
    Provides access to historical climate and weather data
    """
    
    def __init__(self, config: Config, cache: Cache):
        super().__init__("noaa_cdo", config, cache)
        self.base_url = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
    
    def get_data(self, 
                data_type: str,
                location: Union[str, Dict[str, float]], 
                start_date: datetime,
                end_date: datetime,
                scenario: Optional[str] = None,
                variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Get climate data from NOAA Climate Data Online"""
        # Convert location to coordinates
        if isinstance(location, str):
            # In a real implementation, this would geocode the location
            # For demo, we'll use placeholder coordinates
            coords = {"lat": 40.7128, "lon": -74.0060}  # New York City
        else:
            coords = location
        
        # Map data_type to NOAA CDO dataset IDs and variable names
        dataset_mapping = {
            "temperature": {
                "dataset": "GHCND",
                "variables": ["TMAX", "TMIN", "TAVG"]
            },
            "precipitation": {
                "dataset": "GHCND",
                "variables": ["PRCP"]
            },
            "extreme_events": {
                "dataset": "GHCND",
                "variables": ["TMAX", "PRCP"]
            },
            "drought": {
                "dataset": "GHCND",
                "variables": ["PRCP", "TMAX"]
            },
            "flood": {
                "dataset": "GHCND",
                "variables": ["PRCP"]
            },
            "hurricane": {
                "dataset": "GHCND",
                "variables": ["PRCP", "AWND"]
            }
        }
        
        # Get dataset and variables for data_type
        dataset_info = dataset_mapping.get(data_type, {"dataset": "GHCND", "variables": ["TAVG"]})
        dataset = dataset_info["dataset"]
        
        # If specific variables provided, use those instead
        if not variables:
            variables = dataset_info["variables"]
        
        # For demo purposes, we'll generate synthetic data for NOAA CDO
        # In a real implementation, this would call the actual API
        return self._generate_synthetic_data(data_type, coords, start_date, end_date, variables)
    
    def _call_api(self, dataset: str, location: Dict[str, float], start_date: datetime, 
                end_date: datetime, variables: List[str]) -> Dict[str, Any]:
        """
        Call the NOAA CDO API
        
        In a real implementation, this would make an actual API request.
        For the demo, we'll return a placeholder response.
        """
        # Check if we have API token
        api_key = self.api_credentials.get("api_key")
        if not api_key:
            self.logger.warning("No API key found for NOAA CDO")
            return {"results": []}
        
        # In a real implementation, this would be:
        # headers = {
        #     "token": api_key
        # }
        # params = {
        #     "datasetid": dataset,
        #     "startdate": start_date.strftime("%Y-%m-%d"),
        #     "enddate": end_date.strftime("%Y-%m-%d"),
        #     "latitude": location["lat"],
        #     "longitude": location["lon"],
        #     "datatypeid": ",".join(variables),
        #     "units": "metric"
        # }
        # response = requests.get(self.base_url, headers=headers, params=params)
        # return response.json()
        
        self.logger.info(f"Simulating NOAA CDO API call for {variables}")
        # For demo, simulate API latency
        time.sleep(0.5)
        
        # Return placeholder response
        return {"results": []}
    
    def _generate_synthetic_data(self, data_type: str, coords: Dict[str, float], 
                               start_date: datetime, end_date: datetime, 
                               variables: List[str]) -> pd.DataFrame:
        """Generate synthetic climate data for demonstration purposes"""
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with dates
        df = pd.DataFrame(index=dates)
        df.index.name = 'date'
        
        # Generate data based on data_type
        if data_type == "temperature":
            # Base temperature based on latitude (crude approximation)
            lat = coords["lat"]
            base_temp = 25 - abs(lat) * 0.5  # Higher latitudes = colder
            
            # Add yearly cycle
            for date in dates:
                # Day of year normalized to -1 to 1
                day_factor = np.sin((date.dayofyear / 365) * 2 * np.pi - np.pi/2)
                yearly_cycle = day_factor * 10  # +/- 10 degrees seasonal variation
                
                # Add some random variation
                np.random.seed(int(date.timestamp()) % 10000)
                daily_variation = np.random.normal(0, 3)
                
                # Calculate temperatures
                mean_temp = base_temp + yearly_cycle + daily_variation
                max_temp = mean_temp + np.random.uniform(2, 5)
                min_temp = mean_temp - np.random.uniform(2, 5)
                
                # Add to dataframe
                if "TAVG" in variables:
                    df.loc[date, "TAVG"] = mean_temp
                if "TMAX" in variables:
                    df.loc[date, "TMAX"] = max_temp
                if "TMIN" in variables:
                    df.loc[date, "TMIN"] = min_temp
        
        elif data_type in ["precipitation", "flood"]:
            # Base precipitation based on latitude (crude approximation)
            lat = coords["lat"]
            if abs(lat) < 10:
                # Tropical
                base_precip = 6  # mm/day
                variance = 8
            elif abs(lat) < 40:
                # Temperate
                base_precip = 3
                variance = 4
            else:
                # Polar
                base_precip = 1
                variance = 2
            
            # Add yearly cycle
            for date in dates:
                # Seasonality factor
                if lat >= 0:  # Northern hemisphere
                    season_factor = np.sin((date.dayofyear / 365) * 2 * np.pi)
                else:  # Southern hemisphere
                    season_factor = np.sin((date.dayofyear / 365) * 2 * np.pi + np.pi)
                
                # More rain in summer
                seasonal_precip = base_precip + season_factor * base_precip * 0.5
                
                # Add randomness - precipitation is often 0 with occasional higher values
                np.random.seed(int(date.timestamp()) % 10000)
                random_factor = np.random.exponential(1)
                
                # Some days have no rain
                if np.random.random() < 0.7:
                    daily_precip = seasonal_precip * random_factor
                else:
                    daily_precip = 0
                
                # Add to dataframe
                if "PRCP" in variables:
                    df.loc[date, "PRCP"] = daily_precip
        
        elif data_type == "hurricane":
            # Add extreme wind events
            for date in dates:
                # Basic wind speed
                basic_wind = np.random.normal(15, 5)
                
                # Add hurricane season effect (higher in summer/fall)
                month = date.month
                if 6 <= month <= 11:  # Hurricane season
                    # Higher probability of extreme winds
                    if np.random.random() < 0.05:  # 5% chance during season
                        # Hurricane-force winds
                        wind_speed = np.random.normal(50, 15)
                    else:
                        wind_speed = basic_wind
                else:
                    wind_speed = basic_wind
                
                if "AWND" in variables:
                    df.loc[date, "AWND"] = wind_speed
        
        # Add metadata columns
        df["latitude"] = coords["lat"]
        df["longitude"] = coords["lon"]
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Convert date column to string for JSON serialization
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        
        # Rename columns to be more user-friendly
        column_names = {
            "TAVG": "temperature_avg",
            "TMAX": "temperature_max",
            "TMIN": "temperature_min",
            "PRCP": "precipitation",
            "AWND": "wind_speed"
        }
        
        df = df.rename(columns={col: column_names.get(col, col) for col in df.columns})
        
        return df


class CopernicusClimateDataSource(ClimateDataSource):
    """
    Climate data source using Copernicus Climate Data Store API
    
    Provides access to climate reanalysis data and climate projections
    """
    
    def __init__(self, config: Config, cache: Cache):
        super().__init__("copernicus", config, cache)
        self.base_url = "https://cds.climate.copernicus.eu/api/v2"
    
    def get_data(self, 
                data_type: str,
                location: Union[str, Dict[str, float]], 
                start_date: datetime,
                end_date: datetime,
                scenario: Optional[str] = None,
                variables: Optional[List[str]] = None) -> pd.DataFrame:
        """Get climate data from Copernicus Climate Data Store"""
        # Convert location to coordinates
        if isinstance(location, str):
            # In a real implementation, this would geocode the location
            # For demo, we'll use placeholder coordinates
            coords = {"lat": 40.7128, "lon": -74.0060}  # New York City
        else:
            coords = location
        
        # Map data_type to Copernicus dataset IDs and variable names
        dataset_mapping = {
            "temperature": {
                "dataset": "reanalysis-era5-single-levels",
                "variables": ["2m_temperature"]
            },
            "precipitation": {
                "dataset": "reanalysis-era5-single-levels",
                "variables": ["total_precipitation"]
            },
            "projections": {
                "dataset": "projections-cmip6",
                "variables": ["near_surface_air_temperature", "precipitation"]
            },
            "extreme_events": {
                "dataset": "reanalysis-era5-single-levels",
                "variables": ["2m_temperature", "total_precipitation", "10m_wind_speed"]
            }
        }
        
        # Get dataset and variables for data_type
        dataset_info = dataset_mapping.get(data_type, {"dataset": "reanalysis-era5-single-levels", "variables": ["2m_temperature"]})
        dataset = dataset_info["dataset"]
        
        # If specific variables provided, use those instead
        if not variables:
            variables = dataset_info["variables"]
        
        # Map scenario to CMIP6 scenario name
        scenario_mapping = {
            "ipcc_ssp119": "ssp1_1_9",
            "ipcc_ssp126": "ssp1_2_6",
            "ipcc_ssp245": "ssp2_4_5",
            "ipcc_ssp370": "ssp3_7_0",
            "ipcc_ssp585": "ssp5_8_5"
        }
        
        cmip_scenario = scenario_mapping.get(scenario, "ssp2_4_5") if scenario else "historical"
        
        # For demo purposes, we'll generate synthetic data for Copernicus
        # In a real implementation, this would call the actual API
        return self._generate_synthetic_data(data_type, coords, start_date, end_date, variables, cmip_scenario)
    
    def _call_api(self, dataset: str, location: Dict[str, float], start_date: datetime, 
                end_date: datetime, variables: List[str], scenario: str = "historical") -> Dict[str, Any]:
        """
        Call the Copernicus Climate Data Store API
        
        In a real implementation, this would make an actual API request.
        For the demo, we'll return a placeholder response.
        """
        # Check if we have API credentials
        api_key = self.api_credentials.get("api_key")
        api_url = self.api_credentials.get("api_url")
        
        if not api_key:
            self.logger.warning("No API key found for Copernicus CDS")
            return {"results": []}
        
        # In a real implementation, this would be:
        # headers = {
        #     "Authorization": f"Bearer {api_key}"
        # }
        # params = {
        #     "dataset": dataset,
        #     "start_date": start_date.strftime("%Y-%m-%d"),
        #     "end_date": end_date.strftime("%Y-%m-%d"),
        #     "latitude": location["lat"],
        #     "longitude": location["lon"],
        #     "variables": variables,
        #     "format": "json"
        # }
        # if dataset == "projections-cmip6":
        #     params["experiment"] = scenario
        #
        # response = requests.get(self.base_url, headers=headers, params=params)
        # return response.json()
        
        self.logger.info(f"Simulating Copernicus CDS API call for {variables} with scenario {scenario}")
        # For demo, simulate API latency
        time.sleep(0.5)
        
        # Return placeholder response
        return {"results": []}
    
    def _generate_synthetic_data(self, data_type: str, coords: Dict[str, float], 
                               start_date: datetime, end_date: datetime, 
                               variables: List[str], scenario: str) -> pd.DataFrame:
        """Generate synthetic climate data for demonstration purposes"""
        # Create date range
        # For projections, use monthly or yearly frequency
        if data_type == "projections":
            # For very long ranges (climate projections), use yearly data
            if (end_date - start_date).days > 3650:  # More than 10 years
                dates = pd.date_range(start=start_date, end=end_date, freq='YS')  # Year start
            else:
                dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start
        else:
            # For historical data, use daily frequency
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame with dates
        df = pd.DataFrame(index=dates)
        df.index.name = 'date'
        
        # Generate data based on scenario
        # Extract scenario intensity factor
        scenario_factors = {
            "historical": 1.0,
            "ssp1_1_9": 0.8,  # Low emissions
            "ssp1_2_6": 0.9,  # Low-medium emissions
            "ssp2_4_5": 1.0,  # Medium emissions
            "ssp3_7_0": 1.2,  # Medium-high emissions
            "ssp5_8_5": 1.5   # High emissions
        }
        
        scenario_factor = scenario_factors.get(scenario, 1.0)
        
        # Base values depending on data type
        if "temperature" in data_type or "2m_temperature" in variables or "near_surface_air_temperature" in variables:
            # Base temperature trends
            # For projections, add warming trend based on scenario
            for date in dates:
                # Base temperature based on latitude and season
                lat = coords["lat"]
                base_temp = 25 - abs(lat) * 0.5  # Higher latitudes = colder
                
                # Day of year normalized to -1 to 1 for seasonality
                day_factor = np.sin((date.dayofyear / 365) * 2 * np.pi - np.pi/2)
                yearly_cycle = day_factor * 10  # +/- 10 degrees seasonal variation
                
                # Add some random variation
                np.random.seed(int(date.timestamp()) % 10000)
                variation = np.random.normal(0, 3)
                
                # For projections, add warming trend
                if data_type == "projections":
                    # Years since start
                    years_offset = (date.year - start_date.year)
                    # Warming rate depends on scenario (°C per decade)
                    warming_rate = 0.3 * scenario_factor  # degrees per decade
                    warming_trend = years_offset / 10 * warming_rate
                else:
                    warming_trend = 0
                
                # Calculate temperature
                temperature = base_temp + yearly_cycle + variation + warming_trend
                
                # Add to dataframe with appropriate variable name
                if "2m_temperature" in variables:
                    df.loc[date, "2m_temperature"] = temperature
                elif "near_surface_air_temperature" in variables:
                    df.loc[date, "near_surface_air_temperature"] = temperature
                elif "temperature" in variables:
                    df.loc[date, "temperature"] = temperature
        
        if "precipitation" in data_type or "total_precipitation" in variables:
            # Base precipitation based on latitude
            lat = coords["lat"]
            if abs(lat) < 10:  # Tropical
                base_precip = 6  # mm/day
            elif abs(lat) < 40:  # Temperate
                base_precip = 3
            else:  # Polar
                base_precip = 1
            
            for date in dates:
                # Seasonality factor
                if lat >= 0:  # Northern hemisphere
                    season_factor = np.sin((date.dayofyear / 365) * 2 * np.pi)
                else:  # Southern hemisphere
                    season_factor = np.sin((date.dayofyear / 365) * 2 * np.pi + np.pi)
                
                # More rain in summer
                seasonal_precip = base_precip + season_factor * base_precip * 0.5
                
                # Add randomness
                np.random.seed(int(date.timestamp()) % 10000)
                random_factor = np.random.exponential(1)
                
                # For projections, modify precipitation based on scenario
                if data_type == "projections":
                    # Years since start
                    years_offset = (date.year - start_date.year)
                    
                    # Different scenarios have different precipitation changes
                    if scenario in ["ssp5_8_5", "ssp3_7_0"]:
                        # Higher emissions scenarios: more extreme (both wetter and drier)
                        if lat < 0:  # Southern hemisphere
                            # Drier tropics in southern hemisphere
                            precip_change = -0.05 * years_offset / 10  # -5% per decade
                        elif lat > 60:  # High northern latitudes
                            # Wetter high latitudes
                            precip_change = 0.1 * years_offset / 10  # +10% per decade
                        else:
                            # Mixed in mid-latitudes
                            precip_change = 0.02 * years_offset / 10  # +2% per decade
                    else:
                        # Lower emissions scenarios: less extreme changes
                        if lat < 0:
                            precip_change = -0.02 * years_offset / 10
                        elif lat > 60:
                            precip_change = 0.05 * years_offset / 10
                        else:
                            precip_change = 0.01 * years_offset / 10
                    
                    # Apply change factor
                    precip_factor = 1 + precip_change
                else:
                    precip_factor = 1
                
                # Some days have no rain
                if np.random.random() < 0.7:
                    precipitation = seasonal_precip * random_factor * precip_factor
                else:
                    precipitation = 0
                
                # Add to dataframe with appropriate variable name
                if "total_precipitation" in variables:
                    df.loc[date, "total_precipitation"] = precipitation
                elif "precipitation" in variables:
                    df.loc[date, "precipitation"] = precipitation
        
        if "wind" in data_type or "10m_wind_speed" in variables:
            # Base wind speed based on latitude
            lat = coords["lat"]
            
            # Higher winds at higher latitudes
            base_wind = 5 + abs(lat) * 0.1
            
            for date in dates:
                # Seasonal variation
                season_factor = np.sin((date.dayofyear / 365) * 2 * np.pi)
                seasonal_wind = base_wind + season_factor * 2  # More wind in winter
                
                # Add randomness
                np.random.seed(int(date.timestamp()) % 10000)
                random_factor = np.random.normal(1, 0.3)
                
                # For projections, some scenarios show increased storminess
                if data_type == "projections":
                    years_offset = (date.year - start_date.year)
                    
                    # Higher emissions scenarios may have more extreme winds
                    if scenario in ["ssp5_8_5", "ssp3_7_0"]:
                        wind_change = 0.03 * years_offset / 10  # +3% per decade
                    else:
                        wind_change = 0.01 * years_offset / 10  # +1% per decade
                    
                    wind_factor = 1 + wind_change
                else:
                    wind_factor = 1
                
                wind_speed = seasonal_wind * random_factor * wind_factor
                
                # Add to dataframe
                if "10m_wind_speed" in variables:
                    df.loc[date, "10m_wind_speed"] = wind_speed
                elif "wind_speed" in variables:
                    df.loc[date, "wind_speed"] = wind_speed
        
        # Add metadata columns
        df["latitude"] = coords["lat"]
        df["longitude"] = coords["lon"]
        df["scenario"] = scenario
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Convert date column to string for JSON serialization
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        
        # Rename columns to be more user-friendly
        column_names = {
            "2m_temperature": "temperature",
            "near_surface_air_temperature": "temperature",
            "total_precipitation": "precipitation",
            "10m_wind_speed": "wind_speed"
        }
        
        df = df.rename(columns={col: column_names.get(col, col) for col in df.columns})
        
        return df