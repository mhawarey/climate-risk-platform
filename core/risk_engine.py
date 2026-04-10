"""
Risk engine for the Climate Risk Integration Platform.
Handles risk calculations, scenario modeling, and optimization.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from PyQt5.QtCore import QObject, pyqtSignal

from core.data_manager import DataManager
from core.quantum_optimizer import QuantumOptimizer
from core.ai_scenario_generator import AIScenarioGenerator
from utils.explainer import RiskExplainer


class RiskEngine(QObject):
    """
    Core risk calculation engine for climate risk assessment
    Integrates physical and transition risks with financial exposures
    """
    
    # Define signals for asynchronous operations
    calculation_progress = pyqtSignal(str, int, int)  # Signal for calculation progress (task, current, total)
    calculation_complete = pyqtSignal(str, object)  # Signal for completion (task, results)
    
    def __init__(self, data_manager: DataManager):
        super().__init__()
        self.logger = logging.getLogger("risk_engine")
        self.data_manager = data_manager
        
        # Initialize sub-components
        self._init_components()
        
        # Risk model parameters
        self.parameters = {
            "confidence_level": 0.95,
            "time_horizons": [2030, 2050],
            "monte_carlo_iterations": 1000,
            "correlation_lookback_years": 10,
            "risk_free_rate": 0.02  # 2% baseline
        }
        
        self.logger.info("RiskEngine initialized")
    
    def _init_components(self):
        """Initialize risk engine components"""
        # Quantum-inspired optimization engine
        self.optimizer = QuantumOptimizer()
        
        # AI-powered scenario generator
        self.scenario_generator = AIScenarioGenerator()
        
        # Explainable AI module
        self.explainer = RiskExplainer()
    
    def calculate_company_risk(self, 
                               company_id: str,
                               scenarios: Optional[List[str]] = None,
                               include_physical: bool = True,
                               include_transition: bool = True) -> Dict[str, Any]:
        """
        Calculate comprehensive climate risk for a company
        
        Args:
            company_id: Company identifier
            scenarios: List of scenarios to analyze
            include_physical: Whether to include physical risks
            include_transition: Whether to include transition risks
            
        Returns:
            Dictionary with risk metrics and details
        """
        self.logger.info(f"Calculating climate risk for company {company_id}")
        risk_results = {"company_id": company_id}
        
        # Default scenarios if none provided
        if not scenarios:
            scenarios = ["ipcc_ssp245", "ipcc_ssp370", "ipcc_ssp585"]
        
        # Track calculation progress
        total_steps = (1 + len(scenarios) * (include_physical + include_transition))
        current_step = 0
        
        # Fetch company financial data for context
        financials = self.data_manager.get_financial_data(
            "fundamentals", 
            [company_id],
            start_date=datetime.now() - timedelta(days=365*3),
            end_date=datetime.now()
        )
        
        risk_results["financial_data"] = financials.to_dict("records") if not financials.empty else {}
        current_step += 1
        self.calculation_progress.emit("company_risk", current_step, total_steps)
        
        # Calculate physical risks if requested
        if include_physical:
            physical_risks = {}
            for scenario in scenarios:
                physical_risk = self._calculate_physical_risk(company_id, scenario)
                physical_risks[scenario] = physical_risk
                current_step += 1
                self.calculation_progress.emit("company_risk", current_step, total_steps)
            
            risk_results["physical_risks"] = physical_risks
        
        # Calculate transition risks if requested
        if include_transition:
            transition_risks = {}
            for scenario in scenarios:
                transition_risk = self._calculate_transition_risk(company_id, scenario)
                transition_risks[scenario] = transition_risk
                current_step += 1
                self.calculation_progress.emit("company_risk", current_step, total_steps)
            
            risk_results["transition_risks"] = transition_risks
        
        # Calculate combined risk metrics
        risk_results["combined_metrics"] = self._calculate_combined_metrics(
            risk_results.get("physical_risks", {}),
            risk_results.get("transition_risks", {})
        )
        
        # Generate risk explanations
        risk_results["explanations"] = self.explainer.explain_company_risk(risk_results)
        
        # Signal completion
        self.calculation_complete.emit("company_risk", risk_results)
        
        return risk_results
    
    def _calculate_physical_risk(self, company_id: str, scenario: str) -> Dict[str, Any]:
        """Calculate physical climate risks for a company under a scenario"""
        self.logger.info(f"Calculating physical risk for {company_id} under {scenario}")
        
        # Get asset-level physical risks from data manager
        asset_risks = self.data_manager.get_asset_physical_risk(
            company_id,
            risk_types=["flood", "hurricane", "fire", "drought", "extreme_heat"]
        )
        
        # Filter for the requested scenario
        scenario_risks = asset_risks[asset_risks["scenario"] == scenario] if not asset_risks.empty else pd.DataFrame()
        
        if scenario_risks.empty:
            # Return default structure if no data
            return {
                "scenario": scenario,
                "overall_score": 0,
                "asset_risk_scores": [],
                "risk_breakdown": {}
            }
        
        # Calculate overall score (weighted average of asset risks)
        # In a real implementation, this would use more sophisticated weighting
        overall_score = scenario_risks["risk_score"].mean()
        
        # Calculate risk breakdown by type
        risk_breakdown = {}
        for risk_type in scenario_risks["risk_type"].unique():
            type_risks = scenario_risks[scenario_risks["risk_type"] == risk_type]
            risk_breakdown[risk_type] = {
                "score": type_risks["risk_score"].mean(),
                "asset_count": len(type_risks),
                "max_risk": type_risks["risk_score"].max(),
                "min_risk": type_risks["risk_score"].min()
            }
        
        # Prepare asset-level detail
        asset_risk_scores = []
        for asset_id in scenario_risks["asset_id"].unique():
            asset_data = scenario_risks[scenario_risks["asset_id"] == asset_id]
            if not asset_data.empty:
                first_row = asset_data.iloc[0]
                asset_risk_scores.append({
                    "asset_id": asset_id,
                    "asset_name": first_row["asset_name"],
                    "asset_type": first_row["asset_type"],
                    "overall_score": asset_data["risk_score"].mean(),
                    "latitude": first_row["latitude"],
                    "longitude": first_row["longitude"],
                    "risk_types": {
                        row["risk_type"]: row["risk_score"] 
                        for _, row in asset_data.iterrows()
                    }
                })
        
        # Monte Carlo simulation for Value at Risk (VaR)
        # This is a simplified implementation - a real version would be more sophisticated
        var_results = self._calculate_physical_var(scenario_risks, scenario)
        
        return {
            "scenario": scenario,
            "overall_score": overall_score,
            "asset_risk_scores": asset_risk_scores,
            "risk_breakdown": risk_breakdown,
            "value_at_risk": var_results
        }
    
    def _calculate_physical_var(self, risk_data: pd.DataFrame, scenario: str) -> Dict[str, float]:
        """
        Calculate Value at Risk for physical climate risks using Monte Carlo simulation
        
        This is a simplified implementation - a real model would be far more sophisticated
        """
        if risk_data.empty:
            return {"var_95": 0, "var_99": 0, "expected_loss": 0}
        
        # Number of simulations
        n_simulations = self.parameters["monte_carlo_iterations"]
        
        # Estimate asset values (this would come from actual data in a real implementation)
        # Here we're using a simple placeholder calculation
        if "asset_type" in risk_data.columns:
            asset_values = {
                asset_id: self._estimate_asset_value(
                    asset_type=asset_data["asset_type"].iloc[0]
                )
                for asset_id, asset_data in risk_data.groupby("asset_id")
            }
        else:
            # If no asset type, use a default value
            asset_values = {
                asset_id: 1e8 for asset_id in risk_data["asset_id"].unique()
            }
        
        # Run simulations
        losses = []
        
        for _ in range(n_simulations):
            # For each simulation, calculate potential losses across assets
            simulation_loss = 0
            
            for asset_id, asset_data in risk_data.groupby("asset_id"):
                # Get base risk score (average across risk types)
                base_risk = asset_data["risk_score"].mean()
                
                # Convert to a damage probability (simplified)
                # Risk score 1-10 converted to probability of damage 0-0.5
                damage_prob = base_risk / 20.0
                
                # Add randomness to damage probability based on scenario
                # More volatile outcomes in high-emission scenarios
                if scenario in ["ipcc_ssp370", "ipcc_ssp585"]:
                    volatility = 0.3
                else:
                    volatility = 0.2
                
                # Apply random factor to damage probability
                random_factor = np.random.normal(1.0, volatility)
                adjusted_prob = max(0, min(1, damage_prob * random_factor))
                
                # Determine if damage occurs in this simulation
                if np.random.random() < adjusted_prob:
                    # If damage occurs, calculate severity (random % of asset value)
                    severity = np.random.beta(2, 5)  # Beta distribution favoring smaller losses
                    asset_loss = asset_values[asset_id] * severity
                    simulation_loss += asset_loss
            
            losses.append(simulation_loss)
        
        # Calculate VaR metrics
        if losses:
            var_95 = np.percentile(losses, 95)
            var_99 = np.percentile(losses, 99)
            expected_loss = np.mean(losses)
            
            return {
                "var_95": var_95,
                "var_99": var_99,
                "expected_loss": expected_loss
            }
        else:
            return {"var_95": 0, "var_99": 0, "expected_loss": 0}
    
    def _estimate_asset_value(self, asset_type: str) -> float:
        """Estimate asset value based on type (simplified placeholder)"""
        # These are placeholder values - a real implementation would use actual data
        base_values = {
            "refinery": 1e9,  # $1 billion
            "well": 5e6,      # $5 million
            "pipeline": 5e8,  # $500 million
            "storage": 2e8,   # $200 million
            "terminal": 3e8   # $300 million
        }
        
        # Add some randomness
        base = base_values.get(asset_type, 1e8)
        return base * np.random.lognormal(0, 0.5)
    
    def _calculate_transition_risk(self, company_id: str, scenario: str) -> Dict[str, Any]:
        """Calculate transition climate risks for a company under a scenario"""
        self.logger.info(f"Calculating transition risk for {company_id} under {scenario}")
        
        # Get transition risk data from data manager
        transition_data = self.data_manager.get_transition_risk(
            company_id,
            scenarios=[scenario]
        )
        
        if transition_data.empty:
            # Return default structure if no data
            return {
                "scenario": scenario,
                "overall_score": 0,
                "risk_drivers": {},
                "financial_impact": {}
            }
        
        # Extract risk metrics
        row = transition_data.iloc[0]
        overall_score = row["risk_score"]
        
        # Risk drivers
        risk_drivers = {
            "carbon_price": {
                "2030": row["carbon_price_2030"],
                "2050": row["carbon_price_2050"]
            },
            "demand_changes": {
                "2030": row["demand_factor_2030"],
                "2050": row["demand_factor_2050"]
            }
        }
        
        # Financial impacts
        financial_impact = {
            "stranded_asset_risk": row["stranded_asset_risk"],
            "revenue_impact": row["revenue_impact"],
            "carbon_cost_impact": row["carbon_cost_impact"]
        }
        
        # Calculate additional metrics with Monte Carlo simulation
        # In a real implementation, this would be more sophisticated
        mc_results = self._calculate_transition_monte_carlo(company_id, scenario, transition_data)
        
        return {
            "scenario": scenario,
            "overall_score": overall_score,
            "risk_drivers": risk_drivers,
            "financial_impact": financial_impact,
            "monte_carlo_results": mc_results
        }
    
    def _calculate_transition_monte_carlo(self, 
                                         company_id: str, 
                                         scenario: str, 
                                         transition_data: pd.DataFrame) -> Dict[str, Any]:
        """Run Monte Carlo simulation for transition risk"""
        
        # Number of simulations
        n_simulations = self.parameters["monte_carlo_iterations"]
        
        # Extract base values from transition data
        if transition_data.empty:
            return {"npv_impact_mean": 0, "npv_impact_95ci": [0, 0]}
        
        row = transition_data.iloc[0]
        
        # Get financial data for the company
        financials = self.data_manager.get_financial_data(
            "fundamentals", 
            [company_id],
            start_date=datetime.now() - timedelta(days=365*3),
            end_date=datetime.now()
        )
        
        # Default financial values if not available
        revenue = 1e9  # $1 billion default
        ebitda_margin = 0.2  # 20% default
        
        if not financials.empty:
            if 'revenue' in financials.columns:
                revenue = financials['revenue'].mean()
            if 'ebitda_margin' in financials.columns:
                ebitda_margin = financials['ebitda_margin'].mean()
        
        # Base impact values
        base_revenue_impact = row["revenue_impact"]
        base_cost_impact = row["carbon_cost_impact"]
        base_stranded_impact = row["stranded_asset_risk"]
        
        # Run simulations
        npv_impacts = []
        yearly_cashflows = []
        
        for _ in range(n_simulations):
            # Time horizon for NPV calculation
            years = 30
            
            # Generate yearly impacts with randomness
            yearly_impact = []
            
            for year in range(years):
                # Scale factor increases over time (impacts grow)
                time_factor = min(1.0, (year + 1) / 20.0)
                
                # Add randomness to impacts
                revenue_impact = base_revenue_impact * time_factor * np.random.normal(1.0, 0.2)
                cost_impact = base_cost_impact * time_factor * np.random.normal(1.0, 0.3)
                
                # Stranded asset impact is more of a one-time effect
                # Simplified: distribute over first 10 years
                stranded_impact = 0
                if year < 10:
                    stranded_impact = (base_stranded_impact / 10) * np.random.normal(1.0, 0.4)
                
                # Calculate impact on yearly cash flow
                yearly_cf_impact = revenue_impact - cost_impact - stranded_impact
                yearly_impact.append(yearly_cf_impact)
            
            # Calculate NPV of impacts
            discount_rate = self.parameters["risk_free_rate"]
            
            # Adjust discount rate based on scenario
            if scenario == "ipcc_ssp119":
                discount_rate += 0.03  # Higher uncertainty in rapid transition
            elif scenario == "ipcc_ssp585":
                discount_rate += 0.01  # Higher uncertainty in high climate change
            
            npv = 0
            for year, cf_impact in enumerate(yearly_impact):
                npv += cf_impact / ((1 + discount_rate) ** (year + 1))
            
            npv_impacts.append(npv)
            yearly_cashflows.append(yearly_impact)
        
        # Calculate summary statistics
        mean_impact = np.mean(npv_impacts)
        ci_lower = np.percentile(npv_impacts, 2.5)
        ci_upper = np.percentile(npv_impacts, 97.5)
        
        # Calculate impact as percentage of company value
        company_value = revenue * 5  # Simplified valuation
        impact_percentage = mean_impact / company_value if company_value > 0 else 0
        
        # Calculate average yearly impacts for visualization
        avg_yearly_impacts = []
        if yearly_cashflows:
            # Convert to numpy array for easier calculations
            cf_array = np.array(yearly_cashflows)
            # Calculate mean for each year across all simulations
            yearly_means = np.mean(cf_array, axis=0)
            # Calculate 95% confidence intervals
            yearly_lower = np.percentile(cf_array, 2.5, axis=0)
            yearly_upper = np.percentile(cf_array, 97.5, axis=0)
            
            for year in range(len(yearly_means)):
                avg_yearly_impacts.append({
                    "year": year + 1,
                    "mean_impact": yearly_means[year],
                    "ci_lower": yearly_lower[year],
                    "ci_upper": yearly_upper[year]
                })
        
        return {
            "npv_impact_mean": mean_impact,
            "npv_impact_95ci": [ci_lower, ci_upper],
            "impact_percentage": impact_percentage,
            "yearly_impacts": avg_yearly_impacts
        }
    
    def _calculate_combined_metrics(self, 
                                   physical_risks: Dict[str, Any], 
                                   transition_risks: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate combined risk metrics across physical and transition risks"""
        
        combined_metrics = {}
        
        # Process each scenario that appears in both risk types
        common_scenarios = set(physical_risks.keys()).intersection(set(transition_risks.keys()))
        
        for scenario in common_scenarios:
            physical = physical_risks.get(scenario, {})
            transition = transition_risks.get(scenario, {})
            
            # Extract overall scores
            physical_score = physical.get("overall_score", 0)
            transition_score = transition.get("overall_score", 0)
            
            # Calculate combined score (weighted average)
            # This is a simplified approach - a real model would be more sophisticated
            combined_score = (physical_score * 0.5) + (transition_score * 0.5)
            
            # Extract financial impacts
            physical_var = physical.get("value_at_risk", {}).get("var_95", 0)
            transition_npv = transition.get("monte_carlo_results", {}).get("npv_impact_mean", 0)
            
            # Combine financial impacts
            # Note: Physical VAR and transition NPV impact are not directly comparable
            # This is a simplified approach
            financial_impact = abs(physical_var) + abs(transition_npv)
            
            combined_metrics[scenario] = {
                "combined_score": combined_score,
                "physical_score": physical_score,
                "transition_score": transition_score,
                "financial_impact": financial_impact
            }
        
        # Calculate overall metrics across scenarios
        if combined_metrics:
            avg_combined_score = sum(s["combined_score"] for s in combined_metrics.values()) / len(combined_metrics)
            avg_financial_impact = sum(s["financial_impact"] for s in combined_metrics.values()) / len(combined_metrics)
            
            # Find worst-case scenario
            worst_case = max(combined_metrics.items(), key=lambda x: x[1]["combined_score"])
            
            combined_metrics["overall"] = {
                "average_score": avg_combined_score,
                "average_financial_impact": avg_financial_impact,
                "worst_case_scenario": worst_case[0]
            }
        
        return combined_metrics
    
    def calculate_financial_institution_exposure(self, 
                                               institution_id: str,
                                               scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate climate risk exposure for a financial institution
        
        Args:
            institution_id: Financial institution identifier
            scenarios: List of scenarios to analyze
            
        Returns:
            Dictionary with exposure metrics and details
        """
        self.logger.info(f"Calculating climate risk exposure for institution {institution_id}")
        
        # Default scenarios if none provided
        if not scenarios:
            scenarios = ["ipcc_ssp245", "ipcc_ssp370"]
        
        # Get institution's exposure to oil and gas companies
        exposure_data = self.data_manager.get_bank_exposure(institution_id, sector="oil_and_gas")
        
        if exposure_data.empty:
            return {
                "institution_id": institution_id,
                "scenarios": scenarios,
                "exposure_summary": {},
                "company_exposures": []
            }
        
        # Calculate risk for each exposed company
        company_risks = {}
        total_exposure = 0
        
        for _, exposure in exposure_data.iterrows():
            company_id = exposure["company_ticker"]
            exposure_amount = exposure["amount"]
            total_exposure += exposure_amount
            
            # Calculate company risk
            risk_result = self.calculate_company_risk(
                company_id, 
                scenarios=scenarios,
                include_physical=True,
                include_transition=True
            )
            
            company_risks[company_id] = {
                "company_name": exposure["company_name"],
                "exposure_amount": exposure_amount,
                "exposure_type": exposure["exposure_type"],
                "risk_result": risk_result
            }
        
        # Calculate portfolio-level risk metrics
        portfolio_metrics = self._calculate_portfolio_metrics(company_risks, total_exposure, scenarios)
        
        # Prepare company exposure details
        company_exposures = []
        for company_id, risk_data in company_risks.items():
            company_result = {
                "company_id": company_id,
                "company_name": risk_data["company_name"],
                "exposure_amount": risk_data["exposure_amount"],
                "exposure_type": risk_data["exposure_type"],
                "risk_scores": {}
            }
            
            # Extract risk scores for each scenario
            risk_result = risk_data["risk_result"]
            combined_metrics = risk_result.get("combined_metrics", {})
            
            for scenario in scenarios:
                if scenario in combined_metrics:
                    company_result["risk_scores"][scenario] = combined_metrics[scenario].get("combined_score", 0)
            
            company_exposures.append(company_result)
        
        # Sort by exposure amount (descending)
        company_exposures.sort(key=lambda x: x["exposure_amount"], reverse=True)
        
        return {
            "institution_id": institution_id,
            "scenarios": scenarios,
            "exposure_summary": portfolio_metrics,
            "company_exposures": company_exposures
        }
    
    def _calculate_portfolio_metrics(self, 
                                    company_risks: Dict[str, Any], 
                                    total_exposure: float,
                                    scenarios: List[str]) -> Dict[str, Any]:
        """Calculate portfolio-level climate risk metrics"""
        
        if not company_risks or total_exposure == 0:
            return {}
        
        portfolio_metrics = {
            "total_exposure": total_exposure,
            "scenario_metrics": {},
            "risk_concentration": {}
        }
        
        # Calculate weighted risk scores and Value at Risk for each scenario
        for scenario in scenarios:
            weighted_score = 0
            weighted_var = 0
            
            for company_id, risk_data in company_risks.items():
                exposure_amount = risk_data["exposure_amount"]
                weight = exposure_amount / total_exposure
                
                # Get risk score for this scenario
                risk_result = risk_data["risk_result"]
                combined_metrics = risk_result.get("combined_metrics", {})
                scenario_metrics = combined_metrics.get(scenario, {})
                
                score = scenario_metrics.get("combined_score", 0)
                weighted_score += score * weight
                
                # Calculate Value at Risk contribution
                physical_var = 0
                if "physical_risks" in risk_result and scenario in risk_result["physical_risks"]:
                    physical_var = risk_result["physical_risks"][scenario].get("value_at_risk", {}).get("var_95", 0)
                
                transition_impact = 0
                if "transition_risks" in risk_result and scenario in risk_result["transition_risks"]:
                    transition_impact = abs(risk_result["transition_risks"][scenario]
                                           .get("monte_carlo_results", {})
                                           .get("npv_impact_mean", 0))
                
                # Combine physical and transition impacts
                # This is a simplified approach
                risk_impact = physical_var + transition_impact
                
                # Scale by exposure
                exposure_var = min(exposure_amount, risk_impact * exposure_amount / 1e9)
                weighted_var += exposure_var
            
            # Store scenario metrics
            portfolio_metrics["scenario_metrics"][scenario] = {
                "weighted_risk_score": weighted_score,
                "portfolio_var": weighted_var,
                "var_percentage": (weighted_var / total_exposure) if total_exposure > 0 else 0
            }
        
        # Calculate risk concentration metrics
        exposure_by_company = {company_id: data["exposure_amount"] for company_id, data in company_risks.items()}
        
        # Calculate Herfindahl-Hirschman Index (HHI) for concentration
        if exposure_by_company:
            squares = [(amt / total_exposure) ** 2 for amt in exposure_by_company.values()]
            hhi = sum(squares)
            portfolio_metrics["risk_concentration"]["hhi"] = hhi
        
        # Calculate top exposure percentages
        sorted_exposures = sorted(exposure_by_company.values(), reverse=True)
        
        portfolio_metrics["risk_concentration"]["top_5_percent"] = sum(sorted_exposures[:5]) / total_exposure if len(sorted_exposures) >= 5 and total_exposure > 0 else 0
        portfolio_metrics["risk_concentration"]["top_10_percent"] = sum(sorted_exposures[:10]) / total_exposure if len(sorted_exposures) >= 10 and total_exposure > 0 else 0
        
        return portfolio_metrics
    
    def optimize_portfolio(self, 
                          institution_id: str,
                          optimization_goal: str = "risk_reduction",
                          constraint_level: str = "moderate") -> Dict[str, Any]:
        """
        Optimize portfolio allocations to reduce climate risk exposure
        Uses quantum-inspired optimization algorithms
        
        Args:
            institution_id: Financial institution identifier
            optimization_goal: Goal of optimization ("risk_reduction", "return_preservation", "balanced")
            constraint_level: Level of portfolio constraints ("conservative", "moderate", "aggressive")
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info(f"Optimizing portfolio for institution {institution_id}")
        
        # Get current portfolio exposure
        exposure_data = self.data_manager.get_bank_exposure(institution_id, sector="oil_and_gas")
        
        if exposure_data.empty:
            return {
                "institution_id": institution_id,
                "status": "error",
                "message": "No exposure data available",
                "recommendations": []
            }
        
        # Calculate current risk metrics
        current_risk = self.calculate_financial_institution_exposure(
            institution_id,
            scenarios=["ipcc_ssp245", "ipcc_ssp370"]  # Use moderate scenarios for optimization
        )
        
        # Prepare optimization parameters
        opt_params = {
            "goal": optimization_goal,
            "constraints": constraint_level,
            "current_exposure": exposure_data.to_dict("records"),
            "current_risk": current_risk
        }
        
        # Run quantum-inspired optimization
        optimization_result = self.optimizer.optimize_portfolio(opt_params)
        
        # Generate recommendations based on optimization
        recommendations = self._generate_recommendations(optimization_result, current_risk)
        
        return {
            "institution_id": institution_id,
            "status": "success",
            "optimization_result": optimization_result,
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, 
                                 optimization_result: Dict[str, Any],
                                 current_risk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on optimization results"""
        
        if not optimization_result or "optimized_exposure" not in optimization_result:
            return []
        
        recommendations = []
        current_exposures = {exp["company_id"]: exp for exp in current_risk.get("company_exposures", [])}
        optimized_exposures = {exp["company_id"]: exp for exp in optimization_result.get("optimized_exposure", [])}
        
        # Identify major changes in allocation
        for company_id, current in current_exposures.items():
            if company_id in optimized_exposures:
                optimized = optimized_exposures[company_id]
                
                # Calculate change
                current_amount = current["exposure_amount"]
                optimized_amount = optimized["exposure_amount"]
                change = optimized_amount - current_amount
                change_pct = (change / current_amount) if current_amount > 0 else 0
                
                # Only include significant changes
                if abs(change_pct) >= 0.1:  # 10% threshold
                    action = "reduce" if change < 0 else "increase"
                    
                    recommendation = {
                        "company_id": company_id,
                        "company_name": current.get("company_name", company_id),
                        "action": action,
                        "current_amount": current_amount,
                        "suggested_amount": optimized_amount,
                        "change_amount": change,
                        "change_percent": change_pct,
                        "risk_score": current.get("risk_scores", {}).get("ipcc_ssp245", 0),
                        "rationale": self._generate_rationale(company_id, action, current, optimized)
                    }
                    
                    recommendations.append(recommendation)
        
        # Sort recommendations by absolute change amount (descending)
        recommendations.sort(key=lambda x: abs(x["change_amount"]), reverse=True)
        
        return recommendations
    
    def _generate_rationale(self, 
                           company_id: str, 
                           action: str,
                           current: Dict[str, Any],
                           optimized: Dict[str, Any]) -> str:
        """Generate explanation for portfolio recommendation"""
        
        # This is a simplified approach - a real implementation would be more sophisticated
        risk_score = current.get("risk_scores", {}).get("ipcc_ssp245", 0)
        
        if action == "reduce":
            if risk_score >= 7:
                return "High climate risk exposure relative to portfolio"
            elif risk_score >= 5:
                return "Moderate climate risk with unfavorable risk-return profile"
            else:
                return "Portfolio diversification to reduce concentration"
        else:  # increase
            if risk_score <= 3:
                return "Low climate risk with favorable risk-return profile"
            elif risk_score <= 5:
                return "Strategic sector exposure with acceptable risk"
            else:
                return "Potential undervaluation despite climate risks"
    
    def generate_custom_scenario(self, 
                                base_scenario: str,
                                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate custom climate scenario using AI scenario generator
        
        Args:
            base_scenario: Base IPCC scenario to customize
            parameters: Customization parameters
            
        Returns:
            Dictionary with scenario definition
        """
        self.logger.info(f"Generating custom scenario based on {base_scenario}")
        
        # Call the AI scenario generator
        scenario = self.scenario_generator.generate_scenario(base_scenario, parameters)
        
        return scenario
    
    def shutdown(self):
        """Clean shutdown of the risk engine"""
        self.logger.info("Shutting down RiskEngine")
        # Cleanup any resources