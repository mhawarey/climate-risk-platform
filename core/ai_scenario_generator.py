"""
AI-powered climate scenario generator for the Climate Risk Integration Platform.
Generates customized climate scenarios based on user parameters.
"""

import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import random

import numpy as np
import pandas as pd
from scipy import stats


class AIScenarioGenerator:
    """
    AI-powered generator for custom climate scenarios
    Creates internally consistent climate pathways based on user inputs
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ai_scenario_generator")
        self.logger.info("Initializing AIScenarioGenerator")
        
        # Base scenario data from IPCC
        self.base_scenarios = self._load_base_scenarios()
        
        # Scenario parameter ranges
        self.parameter_ranges = {
            "temperature_increase": {
                "ipcc_ssp119": (1.0, 1.8),
                "ipcc_ssp126": (1.3, 2.4),
                "ipcc_ssp245": (2.1, 3.5),
                "ipcc_ssp370": (3.3, 4.7),
                "ipcc_ssp585": (4.4, 5.8)
            },
            "carbon_price_factor": {
                "min": 0.5,
                "max": 3.0
            },
            "renewable_adoption_factor": {
                "min": 0.5,
                "max": 2.0
            },
            "policy_delay_years": {
                "min": 0,
                "max": 15
            },
            "extreme_weather_factor": {
                "min": 0.8,
                "max": 1.5
            }
        }
    
    def _load_base_scenarios(self) -> Dict[str, Any]:
        """
        Load base IPCC scenario data
        In a real implementation, this would load from a database
        For demo purposes, we create synthetic data
        """
        base_scenarios = {}
        
        # Key years for projections
        years = [2025, 2030, 2040, 2050, 2075, 2100]
        
        # SSP1-1.9 (Low emissions scenario)
        base_scenarios["ipcc_ssp119"] = {
            "description": "Sustainability - Taking the Green Road (Low)",
            "temperature_pathway": {str(year): 1.0 + (year - 2020) * 0.008 for year in years},
            "co2_emissions": {
                # Emissions peak early and decline rapidly (GtCO2/yr)
                "2025": 35.0,
                "2030": 25.0,
                "2040": 10.0,
                "2050": 2.0,
                "2075": -5.0,  # Net negative
                "2100": -7.0   # Net negative
            },
            "carbon_price": {
                # Carbon price rises rapidly (USD/tCO2)
                "2025": 45,
                "2030": 85,
                "2040": 140,
                "2050": 210,
                "2075": 350,
                "2100": 450
            },
            "energy_mix": {
                # Share of energy sources (%)
                "fossil": {
                    "2025": 70,
                    "2030": 55,
                    "2040": 30,
                    "2050": 15,
                    "2075": 5,
                    "2100": 0
                },
                "renewable": {
                    "2025": 25,
                    "2030": 40,
                    "2040": 65,
                    "2050": 80,
                    "2075": 90,
                    "2100": 95
                },
                "nuclear": {
                    "2025": 5,
                    "2030": 5,
                    "2040": 5,
                    "2050": 5,
                    "2075": 5,
                    "2100": 5
                }
            },
            "extreme_weather": {
                # Frequency multiplier relative to 1980-2010 average
                "floods": {
                    "2025": 1.2,
                    "2030": 1.3,
                    "2040": 1.4,
                    "2050": 1.5,
                    "2075": 1.6,
                    "2100": 1.7
                },
                "droughts": {
                    "2025": 1.3,
                    "2030": 1.4,
                    "2040": 1.5,
                    "2050": 1.6,
                    "2075": 1.7,
                    "2100": 1.8
                },
                "tropical_cyclones": {
                    "2025": 1.1,
                    "2030": 1.2,
                    "2040": 1.2,
                    "2050": 1.3,
                    "2075": 1.3,
                    "2100": 1.4
                }
            }
        }
        
        # SSP2-4.5 (Intermediate emissions scenario)
        base_scenarios["ipcc_ssp245"] = {
            "description": "Middle of the Road (Intermediate)",
            "temperature_pathway": {str(year): 1.0 + (year - 2020) * 0.028 for year in years},
            "co2_emissions": {
                # Emissions peak around mid-century
                "2025": 40.0,
                "2030": 42.0,
                "2040": 43.0,
                "2050": 40.0,
                "2075": 25.0,
                "2100": 15.0
            },
            "carbon_price": {
                # Moderate carbon price growth
                "2025": 15,
                "2030": 30,
                "2040": 75,
                "2050": 120,
                "2075": 200,
                "2100": 300
            },
            "energy_mix": {
                "fossil": {
                    "2025": 80,
                    "2030": 75,
                    "2040": 65,
                    "2050": 55,
                    "2075": 35,
                    "2100": 25
                },
                "renewable": {
                    "2025": 15,
                    "2030": 20,
                    "2040": 30,
                    "2050": 40,
                    "2075": 60,
                    "2100": 70
                },
                "nuclear": {
                    "2025": 5,
                    "2030": 5,
                    "2040": 5,
                    "2050": 5,
                    "2075": 5,
                    "2100": 5
                }
            },
            "extreme_weather": {
                "floods": {
                    "2025": 1.3,
                    "2030": 1.5,
                    "2040": 1.8,
                    "2050": 2.0,
                    "2075": 2.5,
                    "2100": 3.0
                },
                "droughts": {
                    "2025": 1.4,
                    "2030": 1.6,
                    "2040": 2.0,
                    "2050": 2.3,
                    "2075": 2.8,
                    "2100": 3.3
                },
                "tropical_cyclones": {
                    "2025": 1.2,
                    "2030": 1.3,
                    "2040": 1.5,
                    "2050": 1.7,
                    "2075": 2.0,
                    "2100": 2.3
                }
            }
        }
        
        # SSP5-8.5 (High emissions scenario)
        base_scenarios["ipcc_ssp585"] = {
            "description": "Fossil-fueled Development (High)",
            "temperature_pathway": {str(year): 1.0 + (year - 2020) * 0.05 for year in years},
            "co2_emissions": {
                # Emissions continue to rise
                "2025": 45.0,
                "2030": 50.0,
                "2040": 65.0,
                "2050": 75.0,
                "2075": 90.0,
                "2100": 105.0
            },
            "carbon_price": {
                # Low carbon pricing
                "2025": 0,
                "2030": 5,
                "2040": 10,
                "2050": 20,
                "2075": 35,
                "2100": 50
            },
            "energy_mix": {
                "fossil": {
                    "2025": 85,
                    "2030": 84,
                    "2040": 82,
                    "2050": 80,
                    "2075": 75,
                    "2100": 70
                },
                "renewable": {
                    "2025": 10,
                    "2030": 11,
                    "2040": 13,
                    "2050": 15,
                    "2075": 20,
                    "2100": 25
                },
                "nuclear": {
                    "2025": 5,
                    "2030": 5,
                    "2040": 5,
                    "2050": 5,
                    "2075": 5,
                    "2100": 5
                }
            },
            "extreme_weather": {
                "floods": {
                    "2025": 1.4,
                    "2030": 1.7,
                    "2040": 2.2,
                    "2050": 2.8,
                    "2075": 4.0,
                    "2100": 5.0
                },
                "droughts": {
                    "2025": 1.5,
                    "2030": 1.8,
                    "2040": 2.4,
                    "2050": 3.0,
                    "2075": 4.2,
                    "2100": 5.5
                },
                "tropical_cyclones": {
                    "2025": 1.3,
                    "2030": 1.5,
                    "2040": 2.0,
                    "2050": 2.5,
                    "2075": 3.5,
                    "2100": 4.5
                }
            }
        }
        
        # Add SSP3-7.0 as well
        base_scenarios["ipcc_ssp370"] = {
            "description": "Regional Rivalry (High-Medium)",
            "temperature_pathway": {str(year): 1.0 + (year - 2020) * 0.04 for year in years},
            "co2_emissions": {
                # Emissions rise then plateau at high level
                "2025": 42.0,
                "2030": 46.0,
                "2040": 55.0,
                "2050": 60.0,
                "2075": 65.0,
                "2100": 70.0
            },
            "carbon_price": {
                # Very low carbon pricing
                "2025": 5,
                "2030": 15,
                "2040": 35,
                "2050": 60,
                "2075": 100,
                "2100": 150
            },
            "energy_mix": {
                "fossil": {
                    "2025": 83,
                    "2030": 80,
                    "2040": 75,
                    "2050": 70,
                    "2075": 65,
                    "2100": 60
                },
                "renewable": {
                    "2025": 12,
                    "2030": 15,
                    "2040": 20,
                    "2050": 25,
                    "2075": 30,
                    "2100": 35
                },
                "nuclear": {
                    "2025": 5,
                    "2030": 5,
                    "2040": 5,
                    "2050": 5,
                    "2075": 5,
                    "2100": 5
                }
            },
            "extreme_weather": {
                "floods": {
                    "2025": 1.3,
                    "2030": 1.6,
                    "2040": 2.0,
                    "2050": 2.5,
                    "2075": 3.5,
                    "2100": 4.0
                },
                "droughts": {
                    "2025": 1.4,
                    "2030": 1.7,
                    "2040": 2.2,
                    "2050": 2.7,
                    "2075": 3.7,
                    "2100": 4.5
                },
                "tropical_cyclones": {
                    "2025": 1.2,
                    "2030": 1.4,
                    "2040": 1.8,
                    "2050": 2.2,
                    "2075": 3.0,
                    "2100": 3.8
                }
            }
        }
        
        return base_scenarios
    
    def generate_scenario(self, base_scenario: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate customized climate scenario based on user parameters
        
        Args:
            base_scenario: Base IPCC scenario to customize
            parameters: Customization parameters
            
        Returns:
            Dictionary with scenario definition
        """
        self.logger.info(f"Generating custom scenario based on {base_scenario}")
        
        # Check if base scenario exists
        if base_scenario not in self.base_scenarios:
            self.logger.error(f"Base scenario {base_scenario} not found")
            return {
                "status": "error",
                "message": f"Base scenario {base_scenario} not found"
            }
        
        # Generate a unique ID for the custom scenario
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        scenario_id = f"custom_{base_scenario}_{timestamp}"
        
        # Copy base scenario
        base = self.base_scenarios[base_scenario]
        custom_scenario = {
            "id": scenario_id,
            "base_scenario": base_scenario,
            "description": parameters.get("description", f"Custom scenario based on {base['description']}"),
            "created_at": datetime.now().isoformat(),
            "parameters": parameters,
        }
        
        # Process customization parameters
        try:
            # Temperature pathway adjustment
            temp_increase_factor = parameters.get("temperature_increase_factor", 1.0)
            custom_scenario["temperature_pathway"] = self._adjust_temperature_pathway(
                base["temperature_pathway"],
                temp_increase_factor
            )
            
            # Carbon price adjustment
            carbon_price_factor = parameters.get("carbon_price_factor", 1.0)
            policy_delay_years = parameters.get("policy_delay_years", 0)
            custom_scenario["carbon_price"] = self._adjust_carbon_price(
                base["carbon_price"],
                carbon_price_factor,
                policy_delay_years
            )
            
            # Energy mix adjustment
            renewable_adoption_factor = parameters.get("renewable_adoption_factor", 1.0)
            custom_scenario["energy_mix"] = self._adjust_energy_mix(
                base["energy_mix"],
                renewable_adoption_factor
            )
            
            # CO2 emissions adjustment - derived from energy mix changes
            custom_scenario["co2_emissions"] = self._adjust_co2_emissions(
                base["co2_emissions"],
                custom_scenario["energy_mix"],
                base["energy_mix"]
            )
            
            # Extreme weather adjustment
            extreme_weather_factor = parameters.get("extreme_weather_factor", 1.0)
            # Use temperature pathway to inform extreme weather
            temp_pathway = custom_scenario["temperature_pathway"]
            custom_scenario["extreme_weather"] = self._adjust_extreme_weather(
                base["extreme_weather"],
                extreme_weather_factor,
                temp_pathway
            )
            
            # Generate narrative description
            custom_scenario["narrative"] = self._generate_scenario_narrative(custom_scenario, base)
            
            # Calculate scenario risk metrics
            custom_scenario["risk_metrics"] = self._calculate_scenario_risk_metrics(custom_scenario)
            
            # Status
            custom_scenario["status"] = "success"
            
            return custom_scenario
            
        except Exception as e:
            self.logger.error(f"Error generating custom scenario: {e}")
            return {
                "status": "error",
                "message": f"Error generating custom scenario: {str(e)}"
            }
    
    def _adjust_temperature_pathway(self, 
                                   base_pathway: Dict[str, float], 
                                   factor: float) -> Dict[str, float]:
        """Adjust temperature pathway based on factor"""
        adjusted = {}
        
        # Ensure factor is within reasonable bounds
        factor = max(0.8, min(1.5, factor))
        
        for year, temp in base_pathway.items():
            # Calculate baseline warming (from 2020)
            baseline_warming = temp - base_pathway.get("2020", 1.0)
            
            # Apply factor to additional warming only
            additional_warming = baseline_warming * factor
            
            # New temperature is 2020 baseline plus adjusted additional warming
            adjusted[year] = base_pathway.get("2020", 1.0) + additional_warming
        
        return adjusted
    
    def _adjust_carbon_price(self, 
                            base_price: Dict[str, float], 
                            price_factor: float,
                            delay_years: int) -> Dict[str, float]:
        """Adjust carbon price trajectory based on factors"""
        adjusted = {}
        
        # Ensure factors are within reasonable bounds
        price_factor = max(0.5, min(3.0, price_factor))
        delay_years = max(0, min(15, delay_years))
        
        # Get years as integers
        years = [int(year) for year in base_price.keys()]
        years.sort()
        
        # Create a mapping of year to price
        year_to_price = {int(year): price for year, price in base_price.items()}
        
        for year in years:
            if delay_years > 0:
                # Find the delayed year price
                delayed_year = max(years[0], year - delay_years)
                base_value = year_to_price.get(delayed_year, 0)
            else:
                # No delay, use the base price for this year
                base_value = year_to_price.get(year, 0)
            
            # Apply price factor
            adjusted[str(year)] = base_value * price_factor
        
        return adjusted
    
    def _adjust_energy_mix(self, 
                          base_mix: Dict[str, Dict[str, float]], 
                          renewable_factor: float) -> Dict[str, Dict[str, float]]:
        """Adjust energy mix based on renewable adoption factor"""
        adjusted = {"fossil": {}, "renewable": {}, "nuclear": {}}
        
        # Ensure factor is within reasonable bounds
        renewable_factor = max(0.5, min(2.0, renewable_factor))
        
        # Get years (assuming all sources have the same years)
        years = list(base_mix["fossil"].keys())
        
        for year in years:
            # Get baseline values
            base_fossil = base_mix["fossil"].get(year, 0)
            base_renewable = base_mix["renewable"].get(year, 0)
            base_nuclear = base_mix["nuclear"].get(year, 0)
            
            # Calculate new renewable percentage
            new_renewable = min(95, base_renewable * renewable_factor)
            
            # Calculate the change in renewable
            renewable_change = new_renewable - base_renewable
            
            # Reduce fossil fuels to compensate (protecting nuclear)
            if renewable_change > 0:
                # Cap reduction to not go below 5% fossil
                max_fossil_reduction = base_fossil - 5
                applied_reduction = min(renewable_change, max_fossil_reduction)
                new_fossil = base_fossil - applied_reduction
                
                # If we couldn't reduce fossil enough, adjust the new renewable
                if applied_reduction < renewable_change:
                    new_renewable = base_renewable + applied_reduction
            else:
                # Renewable decreased, increase fossil
                new_fossil = base_fossil - renewable_change
            
            # Keep nuclear constant
            new_nuclear = base_nuclear
            
            # Normalize to 100% if needed
            total = new_fossil + new_renewable + new_nuclear
            if abs(total - 100) > 0.01:  # If more than 0.01% off
                scaling_factor = 100 / total
                new_fossil *= scaling_factor
                new_renewable *= scaling_factor
                new_nuclear *= scaling_factor
            
            # Store adjusted values
            adjusted["fossil"][year] = new_fossil
            adjusted["renewable"][year] = new_renewable
            adjusted["nuclear"][year] = new_nuclear
        
        return adjusted
    
    def _adjust_co2_emissions(self, 
                             base_emissions: Dict[str, float],
                             new_energy_mix: Dict[str, Dict[str, float]],
                             base_energy_mix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Adjust CO2 emissions based on changes in energy mix"""
        adjusted = {}
        
        # Get years
        years = list(base_emissions.keys())
        
        for year in years:
            # Get base emission for this year
            base_emission = base_emissions.get(year, 0)
            
            # Get energy mix for this year
            if year in base_energy_mix["fossil"] and year in new_energy_mix["fossil"]:
                # Calculate the change in fossil fuel percentage
                base_fossil_pct = base_energy_mix["fossil"][year]
                new_fossil_pct = new_energy_mix["fossil"][year]
                
                # Calculate emissions change factor based on fossil fuel change
                # This is a simplified model - in reality, the relationship is more complex
                if base_fossil_pct > 0:
                    emission_factor = new_fossil_pct / base_fossil_pct
                else:
                    emission_factor = 1.0
                
                # Apply the factor to base emissions
                adjusted[year] = base_emission * emission_factor
            else:
                # If we can't calculate based on energy mix, just use base emissions
                adjusted[year] = base_emission
        
        return adjusted
    
    def _adjust_extreme_weather(self, 
                               base_weather: Dict[str, Dict[str, float]],
                               weather_factor: float,
                               temperature_pathway: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Adjust extreme weather based on factors and temperature pathway"""
        adjusted = {"floods": {}, "droughts": {}, "tropical_cyclones": {}}
        
        # Ensure factor is within reasonable bounds
        weather_factor = max(0.8, min(1.5, weather_factor))
        
        # Get event types and years
        event_types = base_weather.keys()
        
        for event_type in event_types:
            years = list(base_weather[event_type].keys())
            
            for year in years:
                # Get base frequency for this event type and year
                base_freq = base_weather[event_type].get(year, 1.0)
                
                # Get temperature anomaly for this year
                temp_anomaly = temperature_pathway.get(year, 1.0) - 1.0  # Anomaly from 2020
                
                # Calculate temperature factor (non-linear relationship)
                # Higher temperatures have exponential impact on extreme weather
                temp_factor = 1.0 + temp_anomaly ** 2 * 0.2
                
                # Apply combined factor
                adjusted[event_type][year] = base_freq * weather_factor * temp_factor
        
        return adjusted
    
    def _generate_scenario_narrative(self, custom_scenario: Dict[str, Any], base_scenario: Dict[str, Any]) -> str:
        """Generate a narrative description of the custom scenario"""
        # Extract key parameters
        params = custom_scenario["parameters"]
        temp_factor = params.get("temperature_increase_factor", 1.0)
        carbon_price_factor = params.get("carbon_price_factor", 1.0)
        renewable_factor = params.get("renewable_adoption_factor", 1.0)
        policy_delay = params.get("policy_delay_years", 0)
        
        # Extract key metrics
        temp_2100 = custom_scenario["temperature_pathway"].get("2100", 4.0)
        carbon_2050 = custom_scenario["carbon_price"].get("2050", 0)
        renewable_2050 = custom_scenario["energy_mix"]["renewable"].get("2050", 0)
        emissions_peak = self._find_emissions_peak(custom_scenario["co2_emissions"])
        
        # Build narrative
        narrative = f"This custom scenario projects a global temperature increase of approximately {temp_2100:.1f}°C by 2100 "
        narrative += f"compared to pre-industrial levels. "
        
        # Emissions trajectory
        if emissions_peak:
            narrative += f"Emissions are projected to peak around {emissions_peak}. "
        else:
            narrative += "Emissions continue to rise throughout the century. "
        
        # Policy response
        if policy_delay > 0:
            narrative += f"Climate policies are delayed by approximately {policy_delay} years compared to the base scenario. "
        
        if carbon_price_factor > 1.2:
            narrative += f"Carbon pricing is aggressive, reaching ${carbon_2050:.0f}/tCO2 by 2050. "
        elif carbon_price_factor < 0.8:
            narrative += f"Carbon pricing is minimal, reaching only ${carbon_2050:.0f}/tCO2 by 2050. "
        else:
            narrative += f"Carbon pricing follows a moderate trajectory, reaching ${carbon_2050:.0f}/tCO2 by 2050. "
        
        # Energy transition
        if renewable_factor > 1.2:
            narrative += f"The energy transition is accelerated, with renewables comprising {renewable_2050:.0f}% of the energy mix by 2050. "
        elif renewable_factor < 0.8:
            narrative += f"The energy transition is delayed, with renewables reaching only {renewable_2050:.0f}% of the energy mix by 2050. "
        else:
            narrative += f"The energy transition proceeds at a moderate pace, with renewables reaching {renewable_2050:.0f}% of the energy mix by 2050. "
        
        # Physical impacts
        if temp_2100 > 3.0:
            narrative += "Physical climate impacts are severe, with substantial increases in extreme weather events, sea level rise, and ecological disruption. "
        elif temp_2100 > 2.0:
            narrative += "Physical climate impacts are significant but manageable with adaptation, including moderate increases in extreme weather and sea level rise. "
        else:
            narrative += "Physical climate impacts are limited, with climate change largely contained through mitigation efforts. "
        
        # Economic implications
        if carbon_price_factor > 1.2 and renewable_factor > 1.2:
            narrative += "The economic transformation is rapid and potentially disruptive for carbon-intensive industries."
        elif carbon_price_factor < 0.8 and renewable_factor < 0.8:
            narrative += "Economic structures remain largely unchanged, with high transition risks accumulating for future decades."
        else:
            narrative += "The economic transformation is gradual, balancing climate goals with economic stability."
        
        return narrative
    
    def _find_emissions_peak(self, emissions: Dict[str, float]) -> Optional[str]:
        """Find the year of peak emissions"""
        if not emissions:
            return None
        
        # Convert years to integers and sort
        years = [(int(year), emissions[year]) for year in emissions.keys()]
        years.sort()
        
        peak_year = None
        peak_value = -float('inf')
        
        for year, value in years:
            if value > peak_value:
                peak_value = value
                peak_year = year
        
        # Check if peak is at the end (i.e., no peak yet)
        if peak_year == years[-1][0]:
            return None
        
        return str(peak_year)
    
    def _calculate_scenario_risk_metrics(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate key risk metrics for the scenario"""
        risk_metrics = {}
        
        # Physical risk metrics
        temp_2050 = scenario["temperature_pathway"].get("2050", 0)
        temp_2100 = scenario["temperature_pathway"].get("2100", 0)
        
        # Transition risk metrics
        carbon_price_2030 = scenario["carbon_price"].get("2030", 0)
        carbon_price_2050 = scenario["carbon_price"].get("2050", 0)
        fossil_2030 = scenario["energy_mix"]["fossil"].get("2030", 0)
        fossil_2050 = scenario["energy_mix"]["fossil"].get("2050", 0)
        
        # Calculate physical risk score (1-10 scale)
        # Higher temperature = higher physical risk
        if temp_2100 <= 1.5:
            physical_risk = 2
        elif temp_2100 <= 2.0:
            physical_risk = 4
        elif temp_2100 <= 3.0:
            physical_risk = 6
        elif temp_2100 <= 4.0:
            physical_risk = 8
        else:
            physical_risk = 10
        
        # Calculate transition risk score (1-10 scale)
        # Higher carbon price and lower fossil fuel use = higher transition risk
        carbon_price_factor = min(10, carbon_price_2050 / 50)  # Normalize to 0-10
        fossil_reduction = (fossil_2030 - fossil_2050) / fossil_2030 if fossil_2030 > 0 else 0
        
        transition_risk = min(10, (carbon_price_factor * 0.5 + fossil_reduction * 10 * 0.5))
        
        risk_metrics["physical_risk_score"] = physical_risk
        risk_metrics["transition_risk_score"] = transition_risk
        risk_metrics["overall_risk_score"] = (physical_risk + transition_risk) / 2
        
        # Calculate estimated global GDP impact (%)
        # Simplified model based on literature
        gdp_impact_2050 = min(10, max(0, (temp_2050 - 1.5) * 2.0))
        gdp_impact_2100 = min(20, max(0, (temp_2100 - 1.5) * 3.0))
        
        risk_metrics["gdp_impact"] = {
            "2050": gdp_impact_2050,
            "2100": gdp_impact_2100
        }
        
        return risk_metrics