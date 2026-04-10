"""
Energy infrastructure data sources for the Climate Risk Integration Platform.
Handles acquisition of energy asset and infrastructure data.
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


class EnergyDataSource(ABC):
    """
    Abstract base class for energy infrastructure data sources
    
    This defines the interface that all energy data sources must implement.
    """
    
    def __init__(self, name: str, config: Config, cache: Cache):
        self.name = name
        self.config = config
        self.cache = cache
        self.logger = logging.getLogger(f"energy.{name}")
        self.api_credentials = self._get_credentials()
        
        self.logger.info(f"Initialized energy data source: {name}")
    
    @staticmethod
    def create(source_name: str, config: Config, cache: Cache) -> 'EnergyDataSource':
        """Factory method to create energy data sources"""
        if source_name == "eia":
            return EIADataSource(config, cache)
        elif source_name == "global_energy_monitor":
            return GlobalEnergyMonitorDataSource(config, cache)
        else:
            raise ValueError(f"Unknown energy data source: {source_name}")
    
    def _get_credentials(self) -> Dict[str, str]:
        """Get API credentials for this data source"""
        return self.config.get_api_credentials(self.name)
    
    @abstractmethod
    def get_data(self,
                data_type: str,
                filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Get energy infrastructure data
        
        Args:
            data_type: Type of infrastructure data (e.g., "oil_wells", "refineries")
            filters: Optional filters to apply to the data
            
        Returns:
            DataFrame with the requested infrastructure data
        """
        pass


class EIADataSource(EnergyDataSource):
    """
    Energy data source using U.S. Energy Information Administration (EIA) API
    
    Provides data on energy production, consumption, and infrastructure
    """
    
    def __init__(self, config: Config, cache: Cache):
        super().__init__("eia", config, cache)
        self.base_url = "https://api.eia.gov/v2"
    
    def get_data(self,
                data_type: str,
                filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Get energy infrastructure data from EIA"""
        # Check if we have an API key
        api_key = self.api_credentials.get("api_key")
        if not api_key:
            self.logger.warning("No API key found for EIA")
            return pd.DataFrame()
        
        # Apply default filters if none provided
        if filters is None:
            filters = {}
        
        # Map data_type to EIA endpoints
        endpoint_mapping = {
            "oil_wells": "petroleum/crude-oil/wells",
            "refineries": "petroleum/refineries",
            "pipelines": "natural-gas/pipelines",
            "storage_facilities": "petroleum/storage",
            "terminals": "petroleum/terminals",
            "production": "petroleum/production",
            "consumption": "petroleum/consumption",
            "reserves": "petroleum/reserves"
        }
        
        endpoint = endpoint_mapping.get(data_type, "petroleum/crude-oil/wells")
        
        # For demo purposes, we'll generate synthetic data
        # In a real implementation, this would call the actual API
        return self._generate_synthetic_data(data_type, filters)
    
    def _call_api(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call the EIA API
        
        In a real implementation, this would make an actual API request.
        For the demo, we'll return a placeholder response.
        """
        api_key = self.api_credentials.get("api_key")
        
        # In a real implementation, this would be:
        # params["api_key"] = api_key
        # response = requests.get(f"{self.base_url}/{endpoint}", params=params)
        # return response.json()
        
        self.logger.info(f"Simulating EIA API call for {endpoint}")
        # For demo, simulate API latency
        time.sleep(0.1)
        
        # Return placeholder response
        return {"response": {"data": []}}
    
    def _generate_synthetic_data(self, 
                               data_type: str, 
                               filters: Dict[str, Any]) -> pd.DataFrame:
        """Generate synthetic energy infrastructure data for demonstration purposes"""
        # Extract filters
        company_id = filters.get("company_id")
        
        # If company ID is provided, use it as seed for reproducible randomness
        if company_id:
            seed = sum(ord(c) for c in company_id)
            np.random.seed(seed)
        
        # Generate different data based on data_type
        if data_type == "oil_wells":
            return self._generate_oil_wells(company_id)
        elif data_type == "refineries":
            return self._generate_refineries(company_id)
        elif data_type == "pipelines":
            return self._generate_pipelines(company_id)
        elif data_type == "storage_facilities":
            return self._generate_storage_facilities(company_id)
        elif data_type == "terminals":
            return self._generate_terminals(company_id)
        elif data_type == "production":
            return self._generate_production(company_id)
        elif data_type == "consumption":
            return self._generate_consumption()
        elif data_type == "reserves":
            return self._generate_reserves(company_id)
        else:
            # Default to empty DataFrame
            return pd.DataFrame()
    
    def _generate_oil_wells(self, company_id: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic oil well data"""
        # Number of wells to generate
        if company_id:
            n_wells = np.random.randint(50, 500)
        else:
            n_wells = np.random.randint(1000, 5000)
        
        # Generate data
        wells = []
        
        # US oil producing regions
        regions = [
            {"name": "Permian Basin", "state": "TX", "lat_range": (30.5, 32.5), "lon_range": (-103.0, -101.0)},
            {"name": "Eagle Ford", "state": "TX", "lat_range": (28.0, 29.5), "lon_range": (-100.0, -97.0)},
            {"name": "Bakken", "state": "ND", "lat_range": (47.5, 48.5), "lon_range": (-104.0, -102.0)},
            {"name": "Marcellus", "state": "PA", "lat_range": (40.5, 42.0), "lon_range": (-80.0, -76.0)},
            {"name": "Niobrara", "state": "CO", "lat_range": (39.5, 41.0), "lon_range": (-105.0, -103.0)}
        ]
        
        companies = ["XOM", "CVX", "BP", "SHEL", "COP", "EOG", "OXY", "MRO", "APA", "DVN"]
        
        for i in range(n_wells):
            # Select a region
            region = random.choice(regions)
            
            # Generate location within region
            latitude = np.random.uniform(region["lat_range"][0], region["lat_range"][1])
            longitude = np.random.uniform(region["lon_range"][0], region["lon_range"][1])
            
            # Select a company
            if company_id:
                well_company = company_id
            else:
                well_company = random.choice(companies)
            
            # Generate well data
            well = {
                "asset_id": f"WELL_{well_company}_{i+1:05d}",
                "name": f"Well {i+1}",
                "type": "well",
                "company_id": well_company,
                "latitude": latitude,
                "longitude": longitude,
                "state": region["state"],
                "region": region["name"],
                "status": np.random.choice(["producing", "idle", "abandoned"], p=[0.8, 0.15, 0.05]),
                "production_bpd": np.random.lognormal(4, 1) if np.random.random() < 0.8 else 0,  # Barrels per day
                "depth_ft": np.random.uniform(5000, 15000),
                "start_year": np.random.randint(1980, 2023)
            }
            
            wells.append(well)
        
        return pd.DataFrame(wells)
    
    def _generate_refineries(self, company_id: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic refinery data"""
        # Number of refineries to generate
        if company_id:
            n_refineries = np.random.randint(3, 10)
        else:
            n_refineries = np.random.randint(20, 50)
        
        # Generate data
        refineries = []
        
        # US regions with major refineries
        regions = [
            {"name": "Gulf Coast", "state": "TX", "lat_range": (29.0, 30.5), "lon_range": (-95.5, -93.5)},
            {"name": "Gulf Coast", "state": "LA", "lat_range": (29.5, 30.5), "lon_range": (-91.5, -89.5)},
            {"name": "West Coast", "state": "CA", "lat_range": (33.5, 34.5), "lon_range": (-118.5, -117.5)},
            {"name": "Mid-Continent", "state": "OK", "lat_range": (35.5, 36.5), "lon_range": (-97.5, -96.5)},
            {"name": "East Coast", "state": "NJ", "lat_range": (40.5, 41.0), "lon_range": (-74.5, -74.0)}
        ]
        
        companies = ["XOM", "CVX", "BP", "SHEL", "COP", "MPC", "PSX", "VLO", "HFC", "PBF"]
        
        for i in range(n_refineries):
            # Select a region
            region = random.choice(regions)
            
            # Generate location within region
            latitude = np.random.uniform(region["lat_range"][0], region["lat_range"][1])
            longitude = np.random.uniform(region["lon_range"][0], region["lon_range"][1])
            
            # Select a company
            if company_id:
                refinery_company = company_id
            else:
                refinery_company = random.choice(companies)
            
            # Generate refinery data
            refinery = {
                "asset_id": f"REF_{refinery_company}_{i+1:02d}",
                "name": f"{region['name']} Refinery {i+1}",
                "type": "refinery",
                "company_id": refinery_company,
                "latitude": latitude,
                "longitude": longitude,
                "state": region["state"],
                "region": region["name"],
                "status": np.random.choice(["operating", "maintenance", "shutdown"], p=[0.9, 0.08, 0.02]),
                "capacity_bpd": np.random.uniform(50000, 600000),  # Barrels per day
                "complexity": np.random.uniform(5, 15),  # Nelson complexity index
                "products": np.random.choice(["gasoline, diesel, jet fuel", "full slate", "specialized"], p=[0.6, 0.3, 0.1]),
                "start_year": np.random.randint(1950, 2010)
            }
            
            refineries.append(refinery)
        
        return pd.DataFrame(refineries)
    
    def _generate_pipelines(self, company_id: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic pipeline data"""
        # Number of pipelines to generate
        if company_id:
            n_pipelines = np.random.randint(5, 20)
        else:
            n_pipelines = np.random.randint(30, 100)
        
        # Generate data
        pipelines = []
        
        # Major US pipeline corridors
        corridors = [
            {"name": "Gulf Coast to Midwest", "start_state": "TX", "end_state": "IL", 
             "start_lat": 29.7, "start_lon": -95.4, "end_lat": 41.9, "end_lon": -87.6},
            {"name": "Permian to Gulf Coast", "start_state": "TX", "end_state": "TX", 
             "start_lat": 31.5, "start_lon": -102.0, "end_lat": 29.7, "end_lon": -95.4},
            {"name": "Bakken to Midwest", "start_state": "ND", "end_state": "IL", 
             "start_lat": 48.0, "start_lon": -103.0, "end_lat": 41.9, "end_lon": -87.6},
            {"name": "Gulf Coast to East Coast", "start_state": "TX", "end_state": "NJ", 
             "start_lat": 29.7, "start_lon": -95.4, "end_lat": 40.7, "end_lon": -74.0},
            {"name": "Rockies to West Coast", "start_state": "CO", "end_state": "CA", 
             "start_lat": 39.7, "start_lon": -104.9, "end_lat": 34.0, "end_lon": -118.2}
        ]
        
        companies = ["EPD", "KMI", "ET", "MMP", "PAA", "WMB", "MPLX", "OKE", "ENB", "TRP"]
        
        for i in range(n_pipelines):
            # Select a corridor
            corridor = random.choice(corridors)
            
            # Generate midpoints along the route
            n_points = np.random.randint(5, 15)
            lat_step = (corridor["end_lat"] - corridor["start_lat"]) / (n_points - 1)
            lon_step = (corridor["end_lon"] - corridor["start_lon"]) / (n_points - 1)
            
            route_points = []
            for j in range(n_points):
                # Add some randomness to the route
                lat_jitter = np.random.normal(0, 0.1)
                lon_jitter = np.random.normal(0, 0.1)
                
                lat = corridor["start_lat"] + j * lat_step + lat_jitter
                lon = corridor["start_lon"] + j * lon_step + lon_jitter
                
                route_points.append({"lat": lat, "lon": lon})
            
            # Calculate center point for mapping
            center_lat = (corridor["start_lat"] + corridor["end_lat"]) / 2
            center_lon = (corridor["start_lon"] + corridor["end_lon"]) / 2
            
            # Select a company
            if company_id:
                pipeline_company = company_id
            else:
                pipeline_company = random.choice(companies)
            
            # Generate pipeline data
            pipeline = {
                "asset_id": f"PIPE_{pipeline_company}_{i+1:02d}",
                "name": f"{corridor['name']} Pipeline {i+1}",
                "type": "pipeline",
                "company_id": pipeline_company,
                "latitude": center_lat,  # Center point for mapping
                "longitude": center_lon,
                "start_state": corridor["start_state"],
                "end_state": corridor["end_state"],
                "route": route_points,
                "length_miles": np.random.uniform(100, 1500),
                "capacity_bpd": np.random.uniform(100000, 1000000),  # Barrels per day
                "product_type": np.random.choice(["crude oil", "refined products", "natural gas", "NGLs"]),
                "status": np.random.choice(["operating", "maintenance", "proposed"], p=[0.85, 0.1, 0.05]),
                "start_year": np.random.randint(1950, 2020)
            }
            
            pipelines.append(pipeline)
        
        return pd.DataFrame(pipelines)
    
    def _generate_storage_facilities(self, company_id: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic storage facility data"""
        # Number of facilities to generate
        if company_id:
            n_facilities = np.random.randint(5, 15)
        else:
            n_facilities = np.random.randint(20, 50)
        
        # Generate data
        facilities = []
        
        # US regions with major storage facilities
        regions = [
            {"name": "Cushing", "state": "OK", "lat_range": (35.9, 36.1), "lon_range": (-96.8, -96.7)},
            {"name": "Gulf Coast", "state": "TX", "lat_range": (29.0, 30.5), "lon_range": (-95.5, -93.5)},
            {"name": "Gulf Coast", "state": "LA", "lat_range": (29.5, 30.5), "lon_range": (-91.5, -89.5)},
            {"name": "West Coast", "state": "CA", "lat_range": (33.5, 34.5), "lon_range": (-118.5, -117.5)},
            {"name": "Mid-Atlantic", "state": "NJ", "lat_range": (40.5, 41.0), "lon_range": (-74.5, -74.0)}
        ]
        
        companies = ["EPD", "KMI", "ET", "MMP", "PAA", "XOM", "CVX", "BP", "SHEL", "PSX"]
        
        for i in range(n_facilities):
            # Select a region
            region = random.choice(regions)
            
            # Generate location within region
            latitude = np.random.uniform(region["lat_range"][0], region["lat_range"][1])
            longitude = np.random.uniform(region["lon_range"][0], region["lon_range"][1])
            
            # Select a company
            if company_id:
                facility_company = company_id
            else:
                facility_company = random.choice(companies)
            
            # Generate storage facility data
            facility = {
                "asset_id": f"STOR_{facility_company}_{i+1:02d}",
                "name": f"{region['name']} Storage {i+1}",
                "type": "storage",
                "company_id": facility_company,
                "latitude": latitude,
                "longitude": longitude,
                "state": region["state"],
                "region": region["name"],
                "status": np.random.choice(["operating", "maintenance", "expansion"], p=[0.9, 0.05, 0.05]),
                "capacity_bbl": np.random.uniform(1e6, 5e7),  # Barrels
                "tank_count": np.random.randint(10, 100),
                "product_type": np.random.choice(["crude oil", "refined products", "mixed"]),
                "start_year": np.random.randint(1960, 2015)
            }
            
            facilities.append(facility)
        
        return pd.DataFrame(facilities)
    
    def _generate_terminals(self, company_id: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic terminal data"""
        # Number of terminals to generate
        if company_id:
            n_terminals = np.random.randint(3, 10)
        else:
            n_terminals = np.random.randint(15, 40)
        
        # Generate data
        terminals = []
        
        # US coastal regions with major terminals
        regions = [
            {"name": "Houston Ship Channel", "state": "TX", "lat_range": (29.6, 29.8), "lon_range": (-95.1, -94.9)},
            {"name": "Louisiana Gulf", "state": "LA", "lat_range": (29.0, 29.2), "lon_range": (-90.0, -89.8)},
            {"name": "Los Angeles", "state": "CA", "lat_range": (33.7, 33.9), "lon_range": (-118.3, -118.1)},
            {"name": "New York Harbor", "state": "NY", "lat_range": (40.6, 40.8), "lon_range": (-74.1, -73.9)},
            {"name": "Puget Sound", "state": "WA", "lat_range": (47.5, 47.7), "lon_range": (-122.4, -122.2)}
        ]
        
        companies = ["XOM", "CVX", "BP", "SHEL", "PSX", "MPC", "VLO", "EPD", "KMI", "ET"]
        
        for i in range(n_terminals):
            # Select a region
            region = random.choice(regions)
            
            # Generate location within region
            latitude = np.random.uniform(region["lat_range"][0], region["lat_range"][1])
            longitude = np.random.uniform(region["lon_range"][0], region["lon_range"][1])
            
            # Select a company
            if company_id:
                terminal_company = company_id
            else:
                terminal_company = random.choice(companies)
            
            # Generate terminal data
            terminal = {
                "asset_id": f"TERM_{terminal_company}_{i+1:02d}",
                "name": f"{region['name']} Terminal {i+1}",
                "type": "terminal",
                "company_id": terminal_company,
                "latitude": latitude,
                "longitude": longitude,
                "state": region["state"],
                "region": region["name"],
                "status": np.random.choice(["operating", "maintenance", "expansion"], p=[0.9, 0.05, 0.05]),
                "capacity_bpd": np.random.uniform(50000, 500000),  # Barrels per day
                "berth_count": np.random.randint(1, 6),
                "max_vessel_size": np.random.choice(["Handysize", "Panamax", "Aframax", "Suezmax", "VLCC"]),
                "product_type": np.random.choice(["crude oil", "refined products", "LNG", "mixed"]),
                "start_year": np.random.randint(1960, 2015)
            }
            
            terminals.append(terminal)
        
        return pd.DataFrame(terminals)
    
    def _generate_production(self, company_id: Optional[str] = None) -> pd.DataFrame:
        """Generate synthetic production data"""
        # Generate monthly data for the past 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 5)
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start
        
        # Generate data
        production_data = []
        
        # Major US oil producing regions
        regions = ["Permian Basin", "Eagle Ford", "Bakken", "Gulf of Mexico", "Marcellus"]
        
        companies = ["XOM", "CVX", "BP", "SHEL", "COP", "EOG", "OXY", "MRO", "APA", "DVN"]
        
        # Filter to specific company if provided
        if company_id:
            company_list = [company_id]
        else:
            company_list = companies
        
        # For each company, generate production data
        for company in company_list:
            # Use company name as seed for reproducible randomness
            seed = sum(ord(c) for c in company)
            np.random.seed(seed)
            
            # Base production level - different for each company
            base_oil = np.random.uniform(100, 1000) * 1000  # Thousands of barrels per day
            base_gas = np.random.uniform(500, 5000) * 1000000  # Thousands of cubic feet per day
            
            # Growth trend - different for each company
            trend_oil = np.random.normal(0.02, 0.05)  # Annual growth rate
            trend_gas = np.random.normal(0.03, 0.05)  # Annual growth rate
            
            # Seasonal factors
            seasonal_amp_oil = base_oil * 0.05  # 5% seasonal amplitude
            seasonal_amp_gas = base_gas * 0.1  # 10% seasonal amplitude
            
            # For each date, generate production
            for date in dates:
                # Calculate time factor (years since start)
                years_elapsed = (date - start_date).days / 365
                
                # Apply growth trend
                trend_factor_oil = (1 + trend_oil) ** years_elapsed
                trend_factor_gas = (1 + trend_gas) ** years_elapsed
                
                # Apply seasonality
                month = date.month
                # More production in summer
                season_factor = np.sin((month / 12) * 2 * np.pi)
                seasonal_oil = seasonal_amp_oil * season_factor
                seasonal_gas = seasonal_amp_gas * season_factor
                
                # Add random variation
                random_oil = np.random.normal(0, base_oil * 0.02)  # 2% random variation
                random_gas = np.random.normal(0, base_gas * 0.02)  # 2% random variation
                
                # Calculate final production
                oil_production = base_oil * trend_factor_oil + seasonal_oil + random_oil
                gas_production = base_gas * trend_factor_gas + seasonal_gas + random_gas
                
                # Ensure non-negative values
                oil_production = max(0, oil_production)
                gas_production = max(0, gas_production)
                
                # Add regional breakdown
                for region in regions:
                    # Regional allocation - different for each company and region
                    region_seed = seed + sum(ord(c) for c in region)
                    np.random.seed(region_seed)
                    
                    # Allocate different percentages to each region
                    region_pct = np.random.uniform(0.05, 0.4)
                    
                    # Calculate regional production
                    region_oil = oil_production * region_pct
                    region_gas = gas_production * region_pct
                    
                    # Add to data
                    production_data.append({
                        "company_id": company,
                        "date": date,
                        "region": region,
                        "oil_production_bpd": region_oil,
                        "gas_production_mcfd": region_gas
                    })
        
        return pd.DataFrame(production_data)
    
    def _generate_consumption(self) -> pd.DataFrame:
        """Generate synthetic consumption data"""
        # Generate monthly data for the past 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 5)
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='MS')  # Month start
        
        # Generate data
        consumption_data = []
        
        # US regions
        regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
        
        # Product types
        products = ["Gasoline", "Diesel", "Jet Fuel", "Residual Fuel Oil", "Propane"]
        
        # For each region and product, generate consumption data
        for region in regions:
            for product in products:
                # Use region and product as seed for reproducible randomness
                seed = sum(ord(c) for c in region) + sum(ord(c) for c in product)
                np.random.seed(seed)
                
                # Base consumption level - different for each region and product
                base_consumption = np.random.uniform(100, 1000) * 1000  # Thousands of barrels per day
                
                # Growth trend - different for each product
                if product == "Gasoline":
                    trend = np.random.normal(-0.005, 0.01)  # Slight decline for gasoline
                elif product == "Diesel":
                    trend = np.random.normal(0.01, 0.01)  # Slight growth for diesel
                elif product == "Jet Fuel":
                    trend = np.random.normal(0.02, 0.02)  # Stronger growth for jet fuel
                else:
                    trend = np.random.normal(0, 0.01)  # Flat for others
                
                # Seasonal factors - stronger for some products
                if product in ["Gasoline", "Jet Fuel"]:
                    seasonal_amp = base_consumption * 0.15  # 15% seasonal amplitude
                elif product == "Propane":
                    seasonal_amp = base_consumption * 0.3  # 30% seasonal amplitude for propane (winter heating)
                else:
                    seasonal_amp = base_consumption * 0.05  # 5% seasonal amplitude
                
                # For each date, generate consumption
                for date in dates:
                    # Calculate time factor (years since start)
                    years_elapsed = (date - start_date).days / 365
                    
                    # Apply growth trend
                    trend_factor = (1 + trend) ** years_elapsed
                    
                    # Apply seasonality
                    month = date.month
                    
                    # Different seasonal patterns
                    if product in ["Gasoline", "Jet Fuel"]:
                        # More consumption in summer
                        season_factor = np.sin((month / 12) * 2 * np.pi)
                    elif product == "Propane":
                        # More consumption in winter
                        season_factor = -np.sin((month / 12) * 2 * np.pi)
                    else:
                        # Milder seasonality
                        season_factor = np.sin((month / 12) * 2 * np.pi) * 0.5
                    
                    # Calculate seasonal effect
                    seasonal_effect = seasonal_amp * season_factor
                    
                    # Add random variation
                    random_effect = np.random.normal(0, base_consumption * 0.03)  # 3% random variation
                    
                    # Calculate final consumption
                    consumption = base_consumption * trend_factor + seasonal_effect + random_effect
                    
                    # Ensure non-negative values
                    consumption = max(0, consumption)
                    
                    # Add to data
                    consumption_data.append({
                        "date": date,
                        "region": region,
                        "product": product,
                        "consumption_bpd": consumption,
                        "trend_factor": trend_factor,
                        "seasonal_effect": seasonal_effect
                    })
        
        return pd.DataFrame(consumption_data)