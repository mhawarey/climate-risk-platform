"""
Financial data sources for the Climate Risk Integration Platform.
Handles acquisition of financial data from various public sources.
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


class FinancialDataSource(ABC):
    """
    Abstract base class for financial data sources
    
    This defines the interface that all financial data sources must implement.
    """
    
    def __init__(self, name: str, config: Config, cache: Cache):
        self.name = name
        self.config = config
        self.cache = cache
        self.logger = logging.getLogger(f"financial.{name}")
        self.api_credentials = self._get_credentials()
        
        self.logger.info(f"Initialized financial data source: {name}")
    
    @staticmethod
    def create(source_name: str, config: Config, cache: Cache) -> 'FinancialDataSource':
        """Factory method to create financial data sources"""
        if source_name == "yahoo_finance":
            return YahooFinanceDataSource(config, cache)
        elif source_name == "alpha_vantage":
            return AlphaVantageDataSource(config, cache)
        elif source_name == "fred":
            return FREDDataSource(config, cache)
        else:
            raise ValueError(f"Unknown financial data source: {source_name}")
    
    def _get_credentials(self) -> Dict[str, str]:
        """Get API credentials for this data source"""
        return self.config.get_api_credentials(self.name)
    
    @abstractmethod
    def get_data(self,
                data_type: str,
                identifiers: List[str],
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                frequency: str = "daily") -> pd.DataFrame:
        """
        Get financial data for the specified parameters
        
        Args:
            data_type: Type of financial data to retrieve
            identifiers: List of ticker symbols or other identifiers
            start_date: Start date for time series data
            end_date: End date for time series data
            frequency: Data frequency (daily, weekly, monthly)
            
        Returns:
            DataFrame with the requested financial data
        """
        pass
    
    @abstractmethod
    def get_companies_by_sector(self, sector: str, country: Optional[str] = None) -> pd.DataFrame:
        """
        Get a list of companies in the specified sector
        
        Args:
            sector: Industry sector (e.g., "oil_and_gas")
            country: Optional country filter
            
        Returns:
            DataFrame with company information
        """
        pass


class YahooFinanceDataSource(FinancialDataSource):
    """
    Financial data source using Yahoo Finance API
    
    Provides market data for stocks, bonds, and other securities
    """
    
    def __init__(self, config: Config, cache: Cache):
        super().__init__("yahoo_finance", config, cache)
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
    
    def get_data(self,
                data_type: str,
                identifiers: List[str],
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                frequency: str = "daily") -> pd.DataFrame:
        """Get financial data from Yahoo Finance"""
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # Map data_type to Yahoo Finance parameters
        # In a real implementation, this would be more comprehensive
        param_mapping = {
            "price": "chart",
            "fundamentals": "fundamentals",
            "volatility": "chart",
            "options": "options"
        }
        
        endpoint = param_mapping.get(data_type, "chart")
        
        # For demo purposes, we'll generate synthetic data for Yahoo Finance
        # In a real implementation, this would call the actual API
        return self._generate_synthetic_data(data_type, identifiers, start_date, end_date, frequency)
    
    def get_companies_by_sector(self, sector: str, country: Optional[str] = None) -> pd.DataFrame:
        """Get companies in the oil and gas sector"""
        # For demo purposes, we'll return a predefined list of companies
        # In a real implementation, this would query the actual API
        
        # Oil and gas companies
        if sector == "oil_and_gas":
            companies = [
                {"ticker": "XOM", "name": "ExxonMobil", "sector": "Energy", "industry": "Oil & Gas Integrated", "country": "US", "market_cap": 400e9},
                {"ticker": "CVX", "name": "Chevron", "sector": "Energy", "industry": "Oil & Gas Integrated", "country": "US", "market_cap": 300e9},
                {"ticker": "BP", "name": "BP plc", "sector": "Energy", "industry": "Oil & Gas Integrated", "country": "UK", "market_cap": 100e9},
                {"ticker": "SHEL", "name": "Shell plc", "sector": "Energy", "industry": "Oil & Gas Integrated", "country": "UK", "market_cap": 200e9},
                {"ticker": "COP", "name": "ConocoPhillips", "sector": "Energy", "industry": "Oil & Gas E&P", "country": "US", "market_cap": 150e9},
                {"ticker": "EOG", "name": "EOG Resources", "sector": "Energy", "industry": "Oil & Gas E&P", "country": "US", "market_cap": 70e9},
                {"ticker": "OXY", "name": "Occidental Petroleum", "sector": "Energy", "industry": "Oil & Gas E&P", "country": "US", "market_cap": 50e9},
                {"ticker": "MRO", "name": "Marathon Oil", "sector": "Energy", "industry": "Oil & Gas E&P", "country": "US", "market_cap": 15e9},
                {"ticker": "APA", "name": "Apache Corporation", "sector": "Energy", "industry": "Oil & Gas E&P", "country": "US", "market_cap": 12e9},
                {"ticker": "DVN", "name": "Devon Energy", "sector": "Energy", "industry": "Oil & Gas E&P", "country": "US", "market_cap": 30e9}
            ]
            
            if country:
                companies = [c for c in companies if c["country"] == country]
            
            return pd.DataFrame(companies)
        
        # Return empty DataFrame for other sectors
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry", "country", "market_cap"])
    
    def _call_api(self, 
                 ticker: str, 
                 start_date: datetime, 
                 end_date: datetime, 
                 interval: str = "1d") -> Dict[str, Any]:
        """
        Call the Yahoo Finance API
        
        In a real implementation, this would make an actual API request.
        For the demo, we'll return a placeholder response.
        """
        # In a real implementation, this would be:
        # params = {
        #     "period1": int(start_date.timestamp()),
        #     "period2": int(end_date.timestamp()),
        #     "interval": interval,
        #     "includePrePost": "false",
        #     "events": "div,split"
        # }
        # response = requests.get(f"{self.base_url}{ticker}", params=params)
        # return response.json()
        
        self.logger.info(f"Simulating Yahoo Finance API call for {ticker}")
        # For demo, simulate API latency
        time.sleep(0.1)
        
        # Return placeholder response
        return {
            "chart": {
                "result": [{
                    "meta": {"symbol": ticker},
                    "timestamp": [],
                    "indicators": {"quote": [{"close": []}]}
                }]
            }
        }
    
    def _generate_synthetic_data(self, 
                               data_type: str, 
                               identifiers: List[str], 
                               start_date: datetime, 
                               end_date: datetime, 
                               frequency: str) -> pd.DataFrame:
        """Generate synthetic financial data for demonstration purposes"""
        # Determine the frequency
        if frequency == "daily":
            freq = "B"  # Business day
        elif frequency == "weekly":
            freq = "W"
        elif frequency == "monthly":
            freq = "MS"  # Month start
        else:
            freq = "B"  # Default to business day
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Initialize result DataFrame
        result_data = []
        
        for ticker in identifiers:
            # Use ticker as seed for reproducible randomness
            seed = sum(ord(c) for c in ticker)
            np.random.seed(seed)
            
            # Generate price data
            if data_type == "price":
                # Base price
                base_price = np.random.uniform(20, 200)
                
                # Trend factor - different for each stock
                trend = np.random.uniform(-0.2, 0.2)  # Annual trend
                
                # Volatility factor
                volatility = np.random.uniform(0.01, 0.05)  # Daily volatility
                
                # Generate price series
                prices = []
                current_price = base_price
                
                for i, date in enumerate(dates):
                    # Add trend
                    days_elapsed = (date - start_date).days
                    trend_factor = 1 + trend * (days_elapsed / 365)
                    
                    # Add random walk
                    random_walk = np.random.normal(0, volatility)
                    
                    # Update price
                    current_price = current_price * (1 + random_walk) * (trend_factor / (1 + trend * ((date - start_date).days - 1) / 365))
                    
                    # Ensure price doesn't go negative
                    current_price = max(0.1, current_price)
                    
                    # Add to result
                    result_data.append({
                        "ticker": ticker,
                        "date": date,
                        "open": current_price * (1 + np.random.normal(0, 0.005)),
                        "high": current_price * (1 + np.random.uniform(0, 0.02)),
                        "low": current_price * (1 - np.random.uniform(0, 0.02)),
                        "close": current_price,
                        "volume": np.random.lognormal(15, 1),
                        "data_type": "price"
                    })
            
            # Generate fundamental data
            elif data_type == "fundamentals":
                # Create quarterly or annual data
                if len(dates) > 20:  # If we have enough dates
                    # Use quarterly frequency
                    quarterly_dates = pd.date_range(start=start_date, end=end_date, freq="Q")
                else:
                    # Use annual frequency
                    quarterly_dates = pd.date_range(start=start_date, end=end_date, freq="YE")
                
                # Metrics scale with random company size
                company_size = np.random.lognormal(23, 1)  # Random company size (revenue)
                
                for date in quarterly_dates:
                    # Calculate years since start
                    years_elapsed = (date - start_date).days / 365
                    
                    # Add some growth trend
                    growth_trend = np.random.uniform(0.01, 0.1)  # Annual growth
                    growth_factor = (1 + growth_trend) ** years_elapsed
                    
                    # Generate fundamental metrics
                    revenue = company_size * growth_factor * (1 + np.random.normal(0, 0.1))
                    
                    # Different margins for different companies
                    base_margin = np.random.uniform(0.1, 0.3)
                    ebitda_margin = base_margin * (1 + np.random.normal(0, 0.1))
                    net_margin = base_margin * 0.6 * (1 + np.random.normal(0, 0.1))
                    
                    ebitda = revenue * ebitda_margin
                    net_income = revenue * net_margin
                    
                    # Balance sheet items
                    assets = revenue * np.random.uniform(1.5, 3)
                    debt = assets * np.random.uniform(0.2, 0.5)
                    equity = assets - debt
                    
                    # Add to result
                    result_data.append({
                        "ticker": ticker,
                        "date": date,
                        "revenue": revenue,
                        "ebitda": ebitda,
                        "ebitda_margin": ebitda_margin,
                        "net_income": net_income,
                        "assets": assets,
                        "debt": debt,
                        "equity": equity,
                        "data_type": "fundamentals"
                    })
            
            # Generate volatility data
            elif data_type == "volatility":
                # Base volatility
                base_volatility = np.random.uniform(15, 35)  # Annual volatility in percentage
                
                for date in dates:
                    # Add seasonality - volatility tends to be higher in fall
                    month = date.month
                    if month in [9, 10, 11]:  # Fall months
                        seasonal_factor = 1.2
                    else:
                        seasonal_factor = 1.0
                    
                    # Add some random variation
                    random_factor = np.random.lognormal(0, 0.2)
                    
                    # Calculate volatility
                    volatility = base_volatility * seasonal_factor * random_factor
                    
                    # Add to result
                    result_data.append({
                        "ticker": ticker,
                        "date": date,
                        "volatility": volatility,
                        "data_type": "volatility"
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(result_data)
        
        return df


class AlphaVantageDataSource(FinancialDataSource):
    """
    Financial data source using Alpha Vantage API
    
    Provides market data and fundamental information for stocks
    """
    
    def __init__(self, config: Config, cache: Cache):
        super().__init__("alpha_vantage", config, cache)
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_data(self,
                data_type: str,
                identifiers: List[str],
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                frequency: str = "daily") -> pd.DataFrame:
        """Get financial data from Alpha Vantage"""
        # Check if we have an API key
        api_key = self.api_credentials.get("api_key")
        if not api_key:
            self.logger.warning("No API key found for Alpha Vantage")
            return pd.DataFrame()
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # Map data_type to Alpha Vantage functions
        function_mapping = {
            "price": "TIME_SERIES_DAILY_ADJUSTED",
            "fundamentals": "OVERVIEW",
            "volatility": "TIME_SERIES_DAILY_ADJUSTED"
        }
        
        function = function_mapping.get(data_type, "TIME_SERIES_DAILY_ADJUSTED")
        
        # For demo purposes, we'll generate synthetic data
        # In a real implementation, this would call the actual API
        return self._generate_synthetic_data(data_type, identifiers, start_date, end_date, frequency)
    
    def get_companies_by_sector(self, sector: str, country: Optional[str] = None) -> pd.DataFrame:
        """Get companies in the specified sector"""
        # Alpha Vantage doesn't have a direct endpoint for this
        # In a real implementation, this might use a combination of endpoints or a separate data source
        # For demo, return an empty DataFrame
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry", "country", "market_cap"])
    
    def _call_api(self, function: str, symbol: str) -> Dict[str, Any]:
        """
        Call the Alpha Vantage API
        
        In a real implementation, this would make an actual API request.
        For the demo, we'll return a placeholder response.
        """
        api_key = self.api_credentials.get("api_key")
        
        # In a real implementation, this would be:
        # params = {
        #     "function": function,
        #     "symbol": symbol,
        #     "apikey": api_key,
        #     "outputsize": "full"
        # }
        # response = requests.get(self.base_url, params=params)
        # return response.json()
        
        self.logger.info(f"Simulating Alpha Vantage API call for {symbol} with function {function}")
        # For demo, simulate API latency
        time.sleep(0.1)
        
        # Return placeholder response
        return {}
    
    def _generate_synthetic_data(self, 
                               data_type: str, 
                               identifiers: List[str], 
                               start_date: datetime, 
                               end_date: datetime, 
                               frequency: str) -> pd.DataFrame:
        """Generate synthetic financial data for demonstration purposes"""
        # This is similar to the Yahoo Finance data generation
        # For simplicity, we'll reuse that for basic data
        
        # Use Yahoo Finance data source with slight variations
        yahoo_source = YahooFinanceDataSource(self.config, self.cache)
        base_data = yahoo_source._generate_synthetic_data(data_type, identifiers, start_date, end_date, frequency)
        
        # Modify slightly to simulate different data source
        if not base_data.empty:
            if "close" in base_data.columns:
                # Add small random variation to prices
                base_data["close"] = base_data["close"] * (1 + np.random.normal(0, 0.01, len(base_data)))
            
            # Add Alpha Vantage specific metrics
            if data_type == "fundamentals":
                # Add some additional metrics
                for i, row in base_data.iterrows():
                    base_data.loc[i, "pe_ratio"] = np.random.uniform(10, 30)
                    base_data.loc[i, "dividend_yield"] = np.random.uniform(0, 0.05)
                    base_data.loc[i, "market_cap"] = row.get("revenue", 1e9) * np.random.uniform(1, 5)
        
        return base_data


class FREDDataSource(FinancialDataSource):
    """
    Financial data source using Federal Reserve Economic Data (FRED) API
    
    Provides macroeconomic indicators and financial data
    """
    
    def __init__(self, config: Config, cache: Cache):
        super().__init__("fred", config, cache)
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    def get_data(self,
                data_type: str,
                identifiers: List[str],
                start_date: Optional[datetime] = None,
                end_date: Optional[datetime] = None,
                frequency: str = "daily") -> pd.DataFrame:
        """Get financial data from FRED"""
        # Check if we have an API key
        api_key = self.api_credentials.get("api_key")
        if not api_key:
            self.logger.warning("No API key found for FRED")
            return pd.DataFrame()
        
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)  # Default to 1 year
        
        # FRED series IDs for common economic indicators
        if data_type == "macro":
            # Map to common macro indicators if not explicitly provided
            if not identifiers or identifiers == ["macro"]:
                identifiers = ["GDP", "UNRATE", "CPIAUCSL", "WTI", "FEDFUNDS"]
        
        # For demo purposes, we'll generate synthetic data
        # In a real implementation, this would call the actual API
        return self._generate_synthetic_data(data_type, identifiers, start_date, end_date, frequency)
    
    def get_companies_by_sector(self, sector: str, country: Optional[str] = None) -> pd.DataFrame:
        """FRED doesn't provide company-level data"""
        # Return empty DataFrame
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry", "country", "market_cap"])
    
    def _call_api(self, series_id: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Call the FRED API
        
        In a real implementation, this would make an actual API request.
        For the demo, we'll return a placeholder response.
        """
        api_key = self.api_credentials.get("api_key")
        
        # In a real implementation, this would be:
        # params = {
        #     "series_id": series_id,
        #     "api_key": api_key,
        #     "file_type": "json",
        #     "observation_start": start_date.strftime("%Y-%m-%d"),
        #     "observation_end": end_date.strftime("%Y-%m-%d")
        # }
        # response = requests.get(self.base_url, params=params)
        # return response.json()
        
        self.logger.info(f"Simulating FRED API call for {series_id}")
        # For demo, simulate API latency
        time.sleep(0.1)
        
        # Return placeholder response
        return {"observations": []}
    
    def _generate_synthetic_data(self, 
                               data_type: str, 
                               identifiers: List[str], 
                               start_date: datetime, 
                               end_date: datetime, 
                               frequency: str) -> pd.DataFrame:
        """Generate synthetic economic data for demonstration purposes"""
        # Determine the frequency for economic data
        if frequency == "daily":
            freq = "B"  # Business day
        elif frequency == "weekly":
            freq = "W"
        elif frequency == "monthly":
            freq = "MS"  # Month start
        elif frequency == "quarterly":
            freq = "QS"  # Quarter start
        else:
            freq = "MS"  # Default to monthly for economic data
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Initialize result DataFrame
        result_data = []
        
        # Generate data for each indicator
        for indicator in identifiers:
            # Use indicator as seed for reproducible randomness
            seed = sum(ord(c) for c in indicator)
            np.random.seed(seed)
            
            # Set base values and trends for common indicators
            if indicator == "GDP":
                # US GDP, ~24 trillion as of 2023
                base_value = 24e12
                annual_growth = np.random.normal(0.02, 0.005)  # ~2% annual growth
                volatility = 0.01
            elif indicator == "UNRATE":
                # US Unemployment rate, ~3.5% as of 2023
                base_value = 3.5
                annual_growth = np.random.normal(0, 0.2)  # Slight drift
                volatility = 0.1
            elif indicator == "CPIAUCSL":
                # US CPI, index value ~300 as of 2023
                base_value = 300
                annual_growth = np.random.normal(0.03, 0.01)  # ~3% inflation
                volatility = 0.002
            elif indicator == "WTI":
                # WTI Crude Oil price, ~$70 as of 2023
                base_value = 70
                annual_growth = np.random.normal(0.01, 0.1)  # Volatile with slight upward trend
                volatility = 0.03
            elif indicator == "FEDFUNDS":
                # Federal Funds Rate, ~4% as of 2023
                base_value = 4.0
                annual_growth = np.random.normal(0, 0.5)  # No clear trend
                volatility = 0.05
            else:
                # Generic values for other indicators
                base_value = 100
                annual_growth = np.random.normal(0.02, 0.01)
                volatility = 0.01
            
            # Generate time series
            current_value = base_value
            
            for date in dates:
                # Calculate time elapsed
                days_elapsed = (date - start_date).days
                years_elapsed = days_elapsed / 365
                
                # Add growth trend
                trend_factor = (1 + annual_growth) ** years_elapsed
                
                # Add random variation
                if indicator in ["UNRATE", "FEDFUNDS"]:
                    # These tend to be sticky (autocorrelated)
                    random_walk = np.random.normal(0, volatility) / 2
                    current_value = current_value * (1 + random_walk)
                else:
                    random_walk = np.random.normal(0, volatility)
                    current_value = base_value * trend_factor * (1 + random_walk)
                
                # Ensure reasonable bounds
                if indicator == "UNRATE":
                    current_value = max(2, min(15, current_value))
                elif indicator == "FEDFUNDS":
                    current_value = max(0, min(20, current_value))
                
                # Add to result
                result_data.append({
                    "indicator": indicator,
                    "date": date,
                    "value": current_value,
                    "data_type": "macro"
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(result_data)
        
        return df