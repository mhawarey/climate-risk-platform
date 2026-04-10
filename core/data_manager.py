"""
Data management system for the Climate Risk Integration Platform.
Handles data acquisition, storage, and retrieval from various sources.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures

import pandas as pd
import numpy as np
import requests
from PyQt5.QtCore import QObject, pyqtSignal

from core.config import Config
from data.climate import ClimateDataSource
from data.financial import FinancialDataSource
from data.energy import EnergyDataSource
from utils.cache import Cache


class DataManager(QObject):
    """
    Manages data acquisition, storage, and retrieval from various sources
    Provides unified access to climate, financial, and energy infrastructure data
    """
    
    # Define signals for asynchronous operations
    data_updated = pyqtSignal(str, object)  # Signal emitted when data is updated
    fetch_completed = pyqtSignal(str, bool, str)  # Signal emitted when fetch is complete (source, success, message)
    progress_updated = pyqtSignal(str, int, int)  # Signal for progress updates (source, current, total)
    
    def __init__(self, config: Config):
        super().__init__()
        self.logger = logging.getLogger("data_manager")
        self.config = config
        
        # Initialize cache
        cache_enabled = config.get("cache.enabled", True)
        cache_location = os.path.expanduser(config.get("cache.location", "~/.cache/climate_risk"))
        cache_max_age = config.get("cache.max_age_hours", 24)
        self.cache = Cache(cache_location, max_age_hours=cache_max_age, enabled=cache_enabled)
        
        # Initialize data sources
        self._init_data_sources()
        
        # In-memory data store for frequently accessed data
        self.data_store = {}
        
        self.logger.info("DataManager initialized")
    
    def _init_data_sources(self):
        """Initialize all data sources based on configuration"""
        self.climate_sources = {}
        self.financial_sources = {}
        self.energy_sources = {}
        
        # Initialize climate data sources
        climate_config = self.config.get("data_sources.climate", {})
        for source_name, source_config in climate_config.items():
            if source_config.get("enabled", False):
                try:
                    source = ClimateDataSource.create(source_name, self.config, self.cache)
                    self.climate_sources[source_name] = source
                    self.logger.info(f"Initialized climate data source: {source_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize climate source {source_name}: {e}")
        
        # Initialize financial data sources
        financial_config = self.config.get("data_sources.financial", {})
        for source_name, source_config in financial_config.items():
            if source_config.get("enabled", False):
                try:
                    source = FinancialDataSource.create(source_name, self.config, self.cache)
                    self.financial_sources[source_name] = source
                    self.logger.info(f"Initialized financial data source: {source_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize financial source {source_name}: {e}")
        
        # Initialize energy infrastructure data sources
        energy_config = self.config.get("data_sources.energy", {})
        for source_name, source_config in energy_config.items():
            if source_config.get("enabled", False):
                try:
                    source = EnergyDataSource.create(source_name, self.config, self.cache)
                    self.energy_sources[source_name] = source
                    self.logger.info(f"Initialized energy data source: {source_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize energy source {source_name}: {e}")
    
    def get_climate_data(self, 
                         data_type: str,
                         location: Union[str, Dict[str, float]], 
                         start_date: datetime,
                         end_date: datetime,
                         scenario: Optional[str] = None,
                         variables: Optional[List[str]] = None,
                         force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieve climate data for a specific location and time period
        
        Args:
            data_type: Type of climate data (e.g., 'temperature', 'precipitation', 'extreme_events')
            location: Location identifier or coordinates {lat: float, lon: float}
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            scenario: Climate scenario (for projection data)
            variables: Specific climate variables to retrieve
            force_refresh: Force refresh from source instead of using cache
            
        Returns:
            DataFrame containing the requested climate data
        """
        # Create cache key
        cache_key = f"climate_{data_type}_{self._location_to_str(location)}_{start_date.isoformat()}_{end_date.isoformat()}"
        if scenario:
            cache_key += f"_{scenario}"
        if variables:
            cache_key += f"_{','.join(variables)}"
            
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Retrieved climate data from cache: {cache_key}")
                return pd.DataFrame(cached_data)
        
        # Determine best source for this data type
        source = self._get_best_climate_source(data_type)
        if not source:
            self.logger.error(f"No suitable climate data source found for: {data_type}")
            return pd.DataFrame()
        
        # Fetch data from source
        try:
            self.logger.info(f"Fetching climate data from {source.name}: {data_type}")
            data = source.get_data(data_type, location, start_date, end_date, scenario, variables)
            
            # Cache the result
            self.cache.set(cache_key, data.to_dict())
            
            # Emit signal that data has been updated
            self.data_updated.emit(f"climate_{data_type}", data)
            
            return data
        except Exception as e:
            self.logger.error(f"Error fetching climate data: {str(e)}")
            self.fetch_completed.emit(f"climate_{data_type}", False, str(e))
            return pd.DataFrame()
    
    def _get_best_climate_source(self, data_type: str) -> Optional[ClimateDataSource]:
        """Determine the best climate data source for a given data type"""
        # Priority order for different data types
        source_priority = {
            "temperature": ["nasa_power", "noaa_cdo", "copernicus"],
            "precipitation": ["noaa_cdo", "nasa_power", "copernicus"],
            "extreme_events": ["noaa_cdo", "copernicus"],
            "sea_level": ["noaa_cdo", "copernicus"],
            "drought": ["noaa_cdo", "nasa_power"],
            "projections": ["copernicus", "nasa_power"]
        }
        
        # Use default priority if data type not specifically mapped
        priority_list = source_priority.get(data_type, list(self.climate_sources.keys()))
        
        # Find the first available source in priority order
        for source_name in priority_list:
            if source_name in self.climate_sources:
                return self.climate_sources[source_name]
        
        # If no specific priority source is available, return any available source
        if self.climate_sources:
            return next(iter(self.climate_sources.values()))
            
        return None
    
    def get_financial_data(self,
                           data_type: str,
                           identifiers: List[str],
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           frequency: str = "daily",
                           force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieve financial data for companies or market indices
        
        Args:
            data_type: Type of financial data (e.g., 'price', 'fundamentals', 'volatility')
            identifiers: List of ticker symbols or company identifiers
            start_date: Start date for time series data
            end_date: End date for time series data
            frequency: Data frequency (daily, weekly, monthly)
            force_refresh: Force refresh from source instead of using cache
            
        Returns:
            DataFrame containing the requested financial data
        """
        # Create cache key
        identifiers_str = ','.join(sorted(identifiers))
        cache_key = f"financial_{data_type}_{identifiers_str}_{frequency}"
        if start_date:
            cache_key += f"_{start_date.isoformat()}"
        if end_date:
            cache_key += f"_{end_date.isoformat()}"
            
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Retrieved financial data from cache: {cache_key}")
                return pd.DataFrame(cached_data)
        
        # Determine best source for this data type
        source = self._get_best_financial_source(data_type)
        if not source:
            self.logger.error(f"No suitable financial data source found for: {data_type}")
            return pd.DataFrame()
        
        # Fetch data from source
        try:
            self.logger.info(f"Fetching financial data from {source.name}: {data_type}")
            data = source.get_data(data_type, identifiers, start_date, end_date, frequency)
            
            # Cache the result
            self.cache.set(cache_key, data.to_dict())
            
            # Emit signal that data has been updated
            self.data_updated.emit(f"financial_{data_type}", data)
            
            return data
        except Exception as e:
            self.logger.error(f"Error fetching financial data: {str(e)}")
            self.fetch_completed.emit(f"financial_{data_type}", False, str(e))
            return pd.DataFrame()
    
    def _get_best_financial_source(self, data_type: str) -> Optional[FinancialDataSource]:
        """Determine the best financial data source for a given data type"""
        # Priority order for different data types
        source_priority = {
            "price": ["yahoo_finance", "alpha_vantage"],
            "fundamentals": ["alpha_vantage", "yahoo_finance"],
            "macro": ["fred", "alpha_vantage"],
            "volatility": ["yahoo_finance", "alpha_vantage"]
        }
        
        # Use default priority if data type not specifically mapped
        priority_list = source_priority.get(data_type, list(self.financial_sources.keys()))
        
        # Find the first available source in priority order
        for source_name in priority_list:
            if source_name in self.financial_sources:
                return self.financial_sources[source_name]
        
        # If no specific priority source is available, return any available source
        if self.financial_sources:
            return next(iter(self.financial_sources.values()))
            
        return None
    
    def get_energy_infrastructure_data(self,
                                       data_type: str,
                                       filters: Optional[Dict[str, Any]] = None,
                                       force_refresh: bool = False) -> pd.DataFrame:
        """
        Retrieve energy infrastructure data
        
        Args:
            data_type: Type of infrastructure data (e.g., 'oil_wells', 'refineries', 'pipelines')
            filters: Optional filters to apply to the data
            force_refresh: Force refresh from source instead of using cache
            
        Returns:
            DataFrame containing the requested infrastructure data
        """
        # Create cache key
        cache_key = f"energy_{data_type}"
        if filters:
            # Convert filters to a stable string representation
            filters_str = json.dumps(filters, sort_keys=True)
            cache_key += f"_{hash(filters_str)}"
            
        # Check cache if not forcing refresh
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Retrieved energy infrastructure data from cache: {cache_key}")
                return pd.DataFrame(cached_data)
        
        # Determine best source for this data type
        source = self._get_best_energy_source(data_type)
        if not source:
            self.logger.error(f"No suitable energy data source found for: {data_type}")
            return pd.DataFrame()
        
        # Fetch data from source
        try:
            self.logger.info(f"Fetching energy infrastructure data from {source.name}: {data_type}")
            data = source.get_data(data_type, filters)
            
            # Cache the result
            self.cache.set(cache_key, data.to_dict())
            
            # Emit signal that data has been updated
            self.data_updated.emit(f"energy_{data_type}", data)
            
            return data
        except Exception as e:
            self.logger.error(f"Error fetching energy infrastructure data: {str(e)}")
            self.fetch_completed.emit(f"energy_{data_type}", False, str(e))
            return pd.DataFrame()
    
    def _get_best_energy_source(self, data_type: str) -> Optional[EnergyDataSource]:
        """Determine the best energy data source for a given data type"""
        # Priority order for different data types
        source_priority = {
            "oil_wells": ["eia", "global_energy_monitor"],
            "refineries": ["eia", "global_energy_monitor"],
            "pipelines": ["eia", "global_energy_monitor"],
            "production": ["eia"],
            "consumption": ["eia"],
            "reserves": ["eia", "global_energy_monitor"]
        }
        
        # Use default priority if data type not specifically mapped
        priority_list = source_priority.get(data_type, list(self.energy_sources.keys()))
        
        # Find the first available source in priority order
        for source_name in priority_list:
            if source_name in self.energy_sources:
                return self.energy_sources[source_name]
        
        # If no specific priority source is available, return any available source
        if self.energy_sources:
            return next(iter(self.energy_sources.values()))
            
        return None
    
    def get_company_list(self, sector: str = "oil_and_gas", country: Optional[str] = None) -> pd.DataFrame:
        """
        Get a list of companies in the specified sector
        
        Args:
            sector: Industry sector (default: "oil_and_gas")
            country: Optional country filter
            
        Returns:
            DataFrame with company information
        """
        cache_key = f"company_list_{sector}"
        if country:
            cache_key += f"_{country}"
        
        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)
        
        # If no cached data, compile from financial sources
        companies = []
        for source in self.financial_sources.values():
            try:
                sector_companies = source.get_companies_by_sector(sector, country)
                if not sector_companies.empty:
                    companies.append(sector_companies)
            except Exception as e:
                self.logger.warning(f"Error getting company list from {source.name}: {str(e)}")
        
        if companies:
            # Combine and remove duplicates
            result = pd.concat(companies).drop_duplicates(subset=["ticker"])
            # Cache the result
            self.cache.set(cache_key, result.to_dict())
            return result
        else:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=["ticker", "name", "sector", "industry", "country", "market_cap"])
    
    def get_bank_exposure(self, bank_id: str, sector: str = "oil_and_gas") -> pd.DataFrame:
        """
        Get a bank's exposure to a specific sector
        
        Args:
            bank_id: Bank identifier (ticker or name)
            sector: Industry sector (default: "oil_and_gas")
            
        Returns:
            DataFrame with exposure information
        """
        cache_key = f"bank_exposure_{bank_id}_{sector}"
        
        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)
        
        # This is a more complex query that might require combining data from multiple sources
        # For demo purposes, we'll generate synthetic data
        # In a real implementation, this would query actual financial disclosure data
        
        # Get companies in the sector
        sector_companies = self.get_company_list(sector)
        if sector_companies.empty:
            return pd.DataFrame()
        
        # Generate synthetic exposure data
        np.random.seed(hash(bank_id) % 10000)  # Deterministic but different for each bank
        
        exposure_data = []
        for _, company in sector_companies.iterrows():
            # Only include some companies (random selection)
            if np.random.random() < 0.7:  # 70% chance of inclusion
                exposure_type = np.random.choice(["loan", "equity", "bond"], p=[0.6, 0.3, 0.1])
                
                # Scale exposure based on market cap
                market_cap = company.get("market_cap", 1e9)
                base_amount = np.random.lognormal(mean=np.log(market_cap * 0.01))
                
                exposure_data.append({
                    "bank_id": bank_id,
                    "company_ticker": company["ticker"],
                    "company_name": company["name"],
                    "exposure_type": exposure_type,
                    "amount": base_amount,
                    "currency": "USD",
                    "report_date": datetime.now().replace(day=1) - timedelta(days=30)  # Last month
                })
        
        result = pd.DataFrame(exposure_data)
        
        # Cache the result
        self.cache.set(cache_key, result.to_dict())
        
        return result
    
    def get_asset_physical_risk(self, 
                                company_id: str, 
                                asset_types: Optional[List[str]] = None,
                                risk_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get physical risk data for company assets
        
        Args:
            company_id: Company identifier
            asset_types: Types of assets to include (e.g., "refinery", "well", "pipeline")
            risk_types: Types of risks to include (e.g., "flood", "hurricane", "fire")
            
        Returns:
            DataFrame with asset risk information
        """
        # Build cache key
        cache_key = f"physical_risk_{company_id}"
        if asset_types:
            cache_key += f"_{'_'.join(sorted(asset_types))}"
        if risk_types:
            cache_key += f"_{'_'.join(sorted(risk_types))}"
            
        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)
        
        # Get company assets
        assets = self._get_company_assets(company_id, asset_types)
        if assets.empty:
            return pd.DataFrame()
        
        # For each asset, calculate physical risks
        risk_data = []
        
        for _, asset in assets.iterrows():
            asset_location = {"lat": asset["latitude"], "lon": asset["longitude"]}
            
            # Default to all risk types if none specified
            if not risk_types:
                risk_types = ["flood", "hurricane", "fire", "drought", "extreme_heat"]
            
            # Get climate data for this location
            for risk_type in risk_types:
                try:
                    # Get historical data
                    historical_end = datetime.now()
                    historical_start = historical_end - timedelta(days=365*10)  # 10 years
                    historical_data = self.get_climate_data(
                        risk_type, 
                        asset_location,
                        historical_start,
                        historical_end
                    )
                    
                    # Get projection data for different scenarios
                    scenarios = self.config.get("risk_engine.default_scenarios", ["ipcc_ssp245"])
                    projection_start = datetime.now()
                    projection_end = projection_start + timedelta(days=365*30)  # 30 years
                    
                    for scenario in scenarios:
                        projection_data = self.get_climate_data(
                            risk_type,
                            asset_location,
                            projection_start,
                            projection_end,
                            scenario=scenario
                        )
                        
                        # Calculate risk scores based on historical and projection data
                        # This is a simplification - real implementation would use more sophisticated models
                        if not historical_data.empty and not projection_data.empty:
                            # Calculate baseline from historical data
                            if 'value' in historical_data.columns:
                                baseline = historical_data['value'].mean()
                                baseline_std = historical_data['value'].std()
                                
                                # Calculate projected change
                                if 'value' in projection_data.columns:
                                    projected = projection_data['value'].mean()
                                    
                                    # Calculate risk score (simplified)
                                    # Higher score = higher risk
                                    change_ratio = projected / baseline if baseline > 0 else 1
                                    volatility = baseline_std / baseline if baseline > 0 else 1
                                    
                                    risk_score = min(10, max(1, change_ratio * (1 + volatility) * 5))
                                    
                                    risk_data.append({
                                        "company_id": company_id,
                                        "asset_id": asset["asset_id"],
                                        "asset_name": asset["name"],
                                        "asset_type": asset["type"],
                                        "risk_type": risk_type,
                                        "risk_score": risk_score,
                                        "scenario": scenario,
                                        "baseline": baseline,
                                        "projected": projected,
                                        "change_ratio": change_ratio,
                                        "latitude": asset["latitude"],
                                        "longitude": asset["longitude"]
                                    })
                except Exception as e:
                    self.logger.warning(f"Error calculating {risk_type} risk for asset {asset['asset_id']}: {str(e)}")
        
        result = pd.DataFrame(risk_data)
        
        # Cache the result
        self.cache.set(cache_key, result.to_dict())
        
        return result
    
    def _get_company_assets(self, company_id: str, asset_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Get assets for a specific company"""
        # Build cache key
        cache_key = f"company_assets_{company_id}"
        if asset_types:
            cache_key += f"_{'_'.join(sorted(asset_types))}"
            
        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)
        
        # Query energy infrastructure data sources
        assets = []
        asset_type_mapping = {
            "refinery": "refineries",
            "well": "oil_wells",
            "pipeline": "pipelines",
            "storage": "storage_facilities",
            "terminal": "terminals"
        }
        
        # Default to all asset types if none specified
        query_types = []
        if asset_types:
            for asset_type in asset_types:
                if asset_type in asset_type_mapping:
                    query_types.append(asset_type_mapping[asset_type])
        else:
            query_types = list(asset_type_mapping.values())
        
        # Query each asset type
        for data_type in query_types:
            try:
                # Query with company filter
                df = self.get_energy_infrastructure_data(
                    data_type,
                    filters={"company_id": company_id}
                )
                
                if not df.empty:
                    # Add asset type column if not present
                    if "type" not in df.columns:
                        # Reverse map to get the simple type name
                        simple_type = next((k for k, v in asset_type_mapping.items() if v == data_type), data_type)
                        df["type"] = simple_type
                    
                    assets.append(df)
            except Exception as e:
                self.logger.warning(f"Error fetching {data_type} for company {company_id}: {str(e)}")
        
        if assets:
            # Combine all asset types
            result = pd.concat(assets, ignore_index=True)
            
            # Ensure required columns exist
            required_cols = ["asset_id", "name", "type", "latitude", "longitude"]
            for col in required_cols:
                if col not in result.columns:
                    if col == "asset_id" and "id" in result.columns:
                        result["asset_id"] = result["id"]
                    else:
                        result[col] = None
            
            # Cache the result
            self.cache.set(cache_key, result.to_dict())
            
            return result
        else:
            # For demo purposes with no data, generate synthetic assets
            return self._generate_synthetic_assets(company_id, asset_types)
    
    def _generate_synthetic_assets(self, company_id: str, asset_types: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate synthetic assets for demonstration purposes"""
        self.logger.info(f"Generating synthetic assets for company {company_id}")
        
        # Use company ID as seed for reproducible randomness
        np.random.seed(hash(company_id) % 10000)
        
        # Default asset types if none specified
        if not asset_types:
            asset_types = ["refinery", "well", "pipeline", "storage", "terminal"]
        
        # Company HQ location (randomly selected US energy hub)
        hq_locations = {
            # Houston area
            "houston": {"lat": 29.7604, "lon": -95.3698},
            # Dallas area
            "dallas": {"lat": 32.7767, "lon": -96.7970},
            # Oklahoma City
            "oklahoma": {"lat": 35.4676, "lon": -97.5164},
            # New Orleans
            "neworleans": {"lat": 29.9511, "lon": -90.0715},
            # Los Angeles
            "losangeles": {"lat": 34.0522, "lon": -118.2437}
        }
        
        hq_location = hq_locations[np.random.choice(list(hq_locations.keys()))]
        
        assets = []
        
        # Generate a reasonable number of assets based on asset type
        counts = {
            "refinery": np.random.randint(1, 5),
            "well": np.random.randint(10, 100),
            "pipeline": np.random.randint(3, 15),
            "storage": np.random.randint(3, 12),
            "terminal": np.random.randint(2, 8)
        }
        
        for asset_type in asset_types:
            count = counts.get(asset_type, 5)
            
            for i in range(count):
                # Generate location near HQ with some dispersion
                # More dispersion for wells, less for refineries
                dispersion = {
                    "refinery": 1.0,
                    "well": 5.0,
                    "pipeline": 3.0,
                    "storage": 2.0,
                    "terminal": 2.5
                }.get(asset_type, 2.0)
                
                lat = hq_location["lat"] + np.random.normal(0, dispersion * 0.1)
                lon = hq_location["lon"] + np.random.normal(0, dispersion * 0.1)
                
                # Ensure coordinates are valid
                lat = max(-90, min(90, lat))
                lon = max(-180, min(180, lon))
                
                asset_id = f"{company_id}_{asset_type}_{i+1}"
                
                assets.append({
                    "asset_id": asset_id,
                    "name": f"{asset_type.title()} {i+1}",
                    "type": asset_type,
                    "latitude": lat,
                    "longitude": lon,
                    "capacity": np.random.lognormal(mean=np.log(1000), sigma=1.0),
                    "status": np.random.choice(["operational", "maintenance", "planned"], p=[0.8, 0.15, 0.05]),
                    "construction_year": np.random.randint(1960, 2020)
                })
        
        return pd.DataFrame(assets)
    
    def get_transition_risk(self,
                            company_id: str,
                            scenarios: Optional[List[str]] = None,
                            time_horizon: str = "2050") -> pd.DataFrame:
        """
        Get transition risk data for a company
        
        Args:
            company_id: Company identifier
            scenarios: Climate policy scenarios to analyze
            time_horizon: Time horizon for analysis
            
        Returns:
            DataFrame with transition risk information
        """
        # Build cache key
        cache_key = f"transition_risk_{company_id}_{time_horizon}"
        if scenarios:
            cache_key += f"_{'_'.join(sorted(scenarios))}"
            
        # Check cache
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)
        
        # Default scenarios if none provided
        if not scenarios:
            scenarios = self.config.get("risk_engine.default_scenarios", ["ipcc_ssp245", "ipcc_ssp370"])
        
        # Get company financial data
        financials = self.get_financial_data(
            "fundamentals",
            [company_id],
            start_date=datetime.now() - timedelta(days=365*3),  # 3 years
            end_date=datetime.now()
        )
        
        # Calculate transition risks for each scenario
        risk_data = []
        
        for scenario in scenarios:
            try:
                # Get carbon price projections for scenario
                carbon_prices = self._get_carbon_price_projection(scenario, time_horizon)
                
                # Get energy demand projections for scenario
                energy_demand = self._get_energy_demand_projection(scenario, time_horizon)
                
                # Get company production and emissions data (or estimate)
                production = self._get_company_production(company_id)
                emissions = self._get_company_emissions(company_id)
                
                # Calculate risks
                if not carbon_prices.empty and not energy_demand.empty:
                    # Extract values for analytical years
                    years = [2025, 2030, 2040, 2050]
                    results = {}
                    
                    for year in years:
                        if str(year) in carbon_prices.columns and str(year) in energy_demand.columns:
                            carbon_price = carbon_prices[str(year)].iloc[0]
                            demand_factor = energy_demand[str(year)].iloc[0]
                            
                            # For demo purposes, rough calculations
                            # In a real model, these would be much more sophisticated
                            
                            # Calculate stranded asset risk
                            stranded_asset_risk = 0
                            revenue_impact = 0
                            cost_impact = 0
                            
                            # Basic calculations if we have financials
                            if not financials.empty and not emissions.empty and not production.empty:
                                # Stranded asset risk based on reserves and demand
                                reserves = production.get('reserves', 0)
                                stranded_percent = max(0, 1 - demand_factor)
                                stranded_asset_risk = reserves * stranded_percent
                                
                                # Revenue impact based on demand changes
                                revenue = financials['revenue'].mean() if 'revenue' in financials.columns else 1e9
                                revenue_impact = revenue * (demand_factor - 1)
                                
                                # Cost impact based on carbon pricing
                                emissions_value = emissions.get('scope1', 0) + emissions.get('scope2', 0)
                                cost_impact = emissions_value * carbon_price
                            
                            results[year] = {
                                "stranded_asset_risk": stranded_asset_risk,
                                "revenue_impact": revenue_impact,
                                "carbon_cost_impact": cost_impact,
                                "carbon_price": carbon_price,
                                "demand_factor": demand_factor
                            }
                    
                    # Overall risk score calculation
                    max_year = max(results.keys())
                    near_term = min(results.keys())
                    
                    # Calculate overall transition risk score (1-10 scale)
                    # Higher score = higher risk
                    # This is a simplified calculation - real models would be more sophisticated
                    
                    # Normalize financial impacts to company size
                    company_value = financials['market_cap'].mean() if 'market_cap' in financials.columns else 1e10
                    
                    stranded_pct = results[max_year]["stranded_asset_risk"] / company_value if company_value > 0 else 0
                    revenue_pct = abs(results[max_year]["revenue_impact"]) / company_value if company_value > 0 else 0
                    cost_pct = results[max_year]["carbon_cost_impact"] / company_value if company_value > 0 else 0
                    
                    # Combine different risk factors with weights
                    risk_score = min(10, (stranded_pct * 3 + revenue_pct * 2 + cost_pct * 2) * 10)
                    
                    # Add to results
                    risk_data.append({
                        "company_id": company_id,
                        "scenario": scenario,
                        "time_horizon": time_horizon,
                        "risk_score": risk_score,
                        "stranded_asset_risk": results[max_year]["stranded_asset_risk"],
                        "revenue_impact": results[max_year]["revenue_impact"],
                        "carbon_cost_impact": results[max_year]["carbon_cost_impact"], 
                        "carbon_price_2030": results.get(2030, {}).get("carbon_price", 0),
                        "carbon_price_2050": results.get(2050, {}).get("carbon_price", 0),
                        "demand_factor_2030": results.get(2030, {}).get("demand_factor", 1),
                        "demand_factor_2050": results.get(2050, {}).get("demand_factor", 1)
                    })
            except Exception as e:
                self.logger.warning(f"Error calculating transition risk for {company_id} in scenario {scenario}: {str(e)}")
        
        result = pd.DataFrame(risk_data)
        
        # Cache the result
        self.cache.set(cache_key, result.to_dict())
        
        return result
    
    def _get_carbon_price_projection(self, scenario: str, time_horizon: str) -> pd.DataFrame:
        """Get carbon price projections for a scenario"""
        # In a real implementation, this would query climate scenario databases
        # For demo, we'll use predetermined values based on IPCC scenarios
        
        # Common years for projections
        years = ["2025", "2030", "2040", "2050"]
        
        # Filter based on requested time horizon
        if time_horizon != "2050":
            max_year = time_horizon if time_horizon in years else "2050"
            years = [year for year in years if year <= max_year]
        
        # Carbon prices in USD per tCO2e for different scenarios
        prices = {
            "ipcc_ssp119": [45, 85, 140, 210],   # 1.5°C scenario with high carbon price
            "ipcc_ssp126": [30, 60, 100, 160],   # 2°C scenario
            "ipcc_ssp245": [15, 30, 75, 120],    # Moderate policy action
            "ipcc_ssp370": [5, 15, 35, 60],      # Limited policy action
            "ipcc_ssp585": [0, 5, 10, 20]        # Minimal policy action
        }
        
        # Default to moderate scenario if not found
        if scenario not in prices:
            scenario_prices = prices["ipcc_ssp245"]
        else:
            scenario_prices = prices[scenario]
        
        # Create DataFrame
        df = pd.DataFrame([scenario_prices], columns=years)
        df["scenario"] = scenario
        
        return df
    
    def _get_energy_demand_projection(self, scenario: str, time_horizon: str) -> pd.DataFrame:
        """Get energy demand projections for oil and gas under different scenarios"""
        # In a real implementation, this would query energy outlook databases
        # For demo, we'll use predetermined values based on IPCC and IEA projections
        
        # Common years for projections - values are demand factors relative to 2020 (1.0)
        years = ["2025", "2030", "2040", "2050"]
        
        # Filter based on requested time horizon
        if time_horizon != "2050":
            max_year = time_horizon if time_horizon in years else "2050"
            years = [year for year in years if year <= max_year]
        
        # Demand factors for different scenarios (values < 1 mean decreasing demand)
        # These are simplified and would be more detailed in a real model
        factors = {
            "ipcc_ssp119": [0.95, 0.80, 0.55, 0.30],  # Rapid transition away from fossil fuels
            "ipcc_ssp126": [0.98, 0.90, 0.70, 0.45],  # Strong but measured transition
            "ipcc_ssp245": [1.02, 0.95, 0.85, 0.75],  # Moderate transition
            "ipcc_ssp370": [1.05, 1.10, 1.05, 0.95],  # Slow transition
            "ipcc_ssp585": [1.08, 1.15, 1.20, 1.15]   # Continued growth of fossil fuels
        }
        
        # Default to moderate scenario if not found
        if scenario not in factors:
            scenario_factors = factors["ipcc_ssp245"]
        else:
            scenario_factors = factors[scenario]
        
        # Create DataFrame
        df = pd.DataFrame([scenario_factors], columns=years)
        df["scenario"] = scenario
        
        return df
    
    def _get_company_production(self, company_id: str) -> Dict[str, float]:
        """Get or estimate company production data"""
        # In a real implementation, this would query production databases
        # For demo, we'll generate synthetic data
        
        # Use company ID as seed for reproducible randomness
        np.random.seed(hash(company_id) % 10000)
        
        # Generate synthetic production data
        return {
            "oil_production": np.random.lognormal(mean=np.log(1e6), sigma=1.0),  # barrels per day
            "gas_production": np.random.lognormal(mean=np.log(1e8), sigma=1.0),  # cubic feet per day
            "reserves": np.random.lognormal(mean=np.log(1e9), sigma=1.0)         # barrels of oil equivalent
        }
    
    def _get_company_emissions(self, company_id: str) -> Dict[str, float]:
        """Get or estimate company emissions data"""
        # In a real implementation, this would query emissions databases
        # For demo, we'll generate synthetic data
        
        # Use company ID as seed for reproducible randomness
        np.random.seed(hash(company_id) % 10000)
        
        # Generate synthetic emissions data
        production = self._get_company_production(company_id)
        
        # Scale emissions based on production (simplified relationship)
        production_factor = (production["oil_production"] + production["gas_production"] / 6000) / 1e6
        
        return {
            "scope1": production_factor * np.random.uniform(0.8, 1.2) * 1e6,  # tCO2e (direct)
            "scope2": production_factor * np.random.uniform(0.3, 0.7) * 1e6,  # tCO2e (energy)
            "scope3": production_factor * np.random.uniform(3.0, 5.0) * 1e6   # tCO2e (value chain)
        }
    
    def _location_to_str(self, location: Union[str, Dict[str, float]]) -> str:
        """Convert location to string for cache keys"""
        if isinstance(location, str):
            return location
        elif isinstance(location, dict):
            return f"{location.get('lat', 0)}_{location.get('lon', 0)}"
        return str(location)
    
    def shutdown(self):
        """Clean shutdown of the data manager"""
        self.logger.info("Shutting down DataManager")
        self.cache.flush()  # Ensure cache is saved