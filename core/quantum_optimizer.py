"""
Quantum-inspired portfolio optimization for climate risk.
"""

import logging
import time
from typing import Dict, List, Any, Tuple
import random

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class QuantumOptimizer:
    """
    Quantum-inspired optimization engine for portfolio allocation
    Uses quantum annealing techniques to optimize financial portfolios under climate risk
    """
    
    def __init__(self):
        self.logger = logging.getLogger("quantum_optimizer")
        self.logger.info("Initializing QuantumOptimizer")
        
        # Optimizer parameters
        self.parameters = {
            "annealing_steps": 1000,
            "temperature_decay": 0.95,
            "initial_temperature": 1.0,
            "convergence_threshold": 1e-6,
            "max_iterations": 500
        }
    
    def optimize_portfolio(self, opt_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize portfolio allocation using quantum-inspired algorithms
        
        Args:
            opt_params: Optimization parameters including current exposures and risk metrics
            
        Returns:
            Dictionary with optimization results
        """
        self.logger.info("Starting portfolio optimization")
        
        # Extract parameters
        goal = opt_params.get("goal", "risk_reduction")
        constraints = opt_params.get("constraints", "moderate")
        current_exposures = opt_params.get("current_exposure", [])
        current_risk = opt_params.get("current_risk", {})
        
        if not current_exposures:
            return {
                "status": "error",
                "message": "No exposure data provided"
            }
        
        # Prepare data for optimization
        companies, weights, risk_scores = self._prepare_optimization_data(current_exposures, current_risk)
        
        if not companies:
            return {
                "status": "error",
                "message": "Failed to prepare optimization data"
            }
        
        # Set up optimization constraints based on constraint level
        constraints_config = self._setup_constraints(constraints, weights)
        
        # Set up objective function based on optimization goal
        objective_func, objective_type = self._setup_objective(goal, risk_scores)
        
        # Run optimization
        try:
            start_time = time.time()
            result = self._run_optimization(weights, objective_func, constraints_config, objective_type)
            end_time = time.time()
            
            # Process and format results
            optimization_result = self._process_optimization_results(
                companies, 
                weights, 
                result, 
                current_exposures,
                risk_scores
            )
            
            optimization_result["computation_time"] = end_time - start_time
            optimization_result["status"] = "success"
            optimization_result["goal"] = goal
            optimization_result["constraints"] = constraints
            
            self.logger.info(f"Portfolio optimization completed in {end_time - start_time:.2f} seconds")
            
            return optimization_result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return {
                "status": "error",
                "message": f"Optimization failed: {str(e)}"
            }
    
    def _prepare_optimization_data(self, 
                                  current_exposures: List[Dict[str, Any]], 
                                  current_risk: Dict[str, Any]) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Prepare data for optimization
        
        Returns:
            Tuple of (company_ids, current_weights, risk_scores)
        """
        # Extract company exposures and risk scores
        companies = []
        amounts = []
        risk_scores = []
        
        # Extract company risk scores from current risk assessment
        company_risks = {}
        for company in current_risk.get("company_exposures", []):
            company_id = company.get("company_id")
            
            # Use ipcc_ssp245 scenario for optimization if available, otherwise use any available scenario
            risk_score = company.get("risk_scores", {}).get("ipcc_ssp245", 0)
            if risk_score == 0:
                # Try to get any risk score
                for scenario, score in company.get("risk_scores", {}).items():
                    if score > 0:
                        risk_score = score
                        break
            
            company_risks[company_id] = risk_score
        
        # Process current exposures
        total_exposure = sum(exp.get("amount", 0) for exp in current_exposures)
        
        if total_exposure <= 0:
            self.logger.error("Total exposure is zero or negative")
            return [], np.array([]), np.array([])
        
        for exposure in current_exposures:
            company_id = exposure.get("company_ticker")
            amount = exposure.get("amount", 0)
            
            if amount > 0:
                companies.append(company_id)
                amounts.append(amount)
                
                # Get risk score, default to 5 (medium) if not available
                risk_score = company_risks.get(company_id, 5)
                risk_scores.append(risk_score)
        
        # Convert amounts to weights
        weights = np.array(amounts) / total_exposure
        
        return companies, weights, np.array(risk_scores)
    
    def _setup_constraints(self, constraint_level: str, initial_weights: np.ndarray) -> Dict[str, Any]:
        """
        Set up optimization constraints based on constraint level
        
        Args:
            constraint_level: Level of constraints ("conservative", "moderate", "aggressive")
            initial_weights: Current portfolio weights
            
        Returns:
            Dictionary with constraint parameters
        """
        # Default constraint parameters
        constraints = {
            "max_change": 0.25,  # Maximum change in any weight
            "min_weight": 0.0,   # Minimum weight for any asset
            "max_weight": 0.3,   # Maximum weight for any asset
            "min_positions": max(1, int(len(initial_weights) * 0.7))  # Minimum number of positions
        }
        
        # Adjust based on constraint level
        if constraint_level == "conservative":
            constraints["max_change"] = 0.15
            constraints["max_weight"] = 0.2
            constraints["min_positions"] = max(1, int(len(initial_weights) * 0.9))
        elif constraint_level == "aggressive":
            constraints["max_change"] = 0.5
            constraints["max_weight"] = 0.4
            constraints["min_positions"] = max(1, int(len(initial_weights) * 0.5))
        
        return constraints
    
    def _setup_objective(self, goal: str, risk_scores: np.ndarray) -> Tuple[callable, str]:
        """
        Set up objective function based on optimization goal
        
        Args:
            goal: Optimization goal ("risk_reduction", "return_preservation", "balanced")
            risk_scores: Array of company risk scores
            
        Returns:
            Tuple of (objective_function, minimize/maximize)
        """
        # Default is to minimize risk
        objective_type = "minimize"
        
        if goal == "risk_reduction":
            # Minimize weighted risk
            def objective(weights):
                return np.sum(weights * risk_scores)
        
        elif goal == "return_preservation":
            # Minimize changes to original allocation while reducing highest risks
            # We use negative because we're minimizing
            def objective(weights):
                # Return proxy: penalize changes to low-risk companies (risk < 5)
                low_risk_mask = risk_scores < 5
                low_risk_changes = np.abs(weights[low_risk_mask] - self.initial_weights[low_risk_mask])
                low_risk_penalty = np.sum(low_risk_changes) * 2
                
                # Risk component: weighted average risk score
                risk_component = np.sum(weights * risk_scores)
                
                return risk_component + low_risk_penalty
        
        elif goal == "balanced":
            # Balance between risk reduction and portfolio stability
            def objective(weights):
                # Risk component: weighted average risk score
                risk_component = np.sum(weights * risk_scores)
                
                # Stability component: penalize large changes from initial allocation
                changes = np.abs(weights - self.initial_weights)
                stability_component = np.sum(changes)
                
                # Combined objective with equal weighting
                return risk_component + stability_component
        
        else:
            # Default to risk reduction
            def objective(weights):
                return np.sum(weights * risk_scores)
        
        return objective, objective_type
    
    def _run_optimization(self, 
                         initial_weights: np.ndarray, 
                         objective_func: callable,
                         constraints_config: Dict[str, Any],
                         objective_type: str) -> Dict[str, Any]:
        """
        Run the quantum-inspired optimization
        
        Args:
            initial_weights: Initial portfolio weights
            objective_func: Objective function to optimize
            constraints_config: Constraint parameters
            objective_type: Whether to minimize or maximize
            
        Returns:
            Optimization results dictionary
        """
        # Store initial weights for use in objective functions
        self.initial_weights = initial_weights
        
        # Number of assets
        n_assets = len(initial_weights)
        
        # Extract constraints
        max_change = constraints_config["max_change"]
        min_weight = constraints_config["min_weight"]
        max_weight = constraints_config["max_weight"]
        min_positions = constraints_config["min_positions"]
        
        # Set bounds for each weight
        bounds = []
        for i, w in enumerate(initial_weights):
            lower = max(min_weight, w - max_change)
            upper = min(max_weight, w + max_change)
            bounds.append((lower, upper))
        
        # Constraint: weights sum to 1
        sum_constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Constraint: minimum number of positions (non-zero weights)
        # This is a complex constraint that requires integer programming
        # For simplicity, we'll approach this through a penalty in the objective
        
        # Define modified objective with position constraint penalty
        def modified_objective(weights):
            # Original objective
            obj_value = objective_func(weights)
            
            # Count positions with significant weight
            significant_positions = np.sum(weights > 0.001)
            
            # Add penalty if below minimum positions
            if significant_positions < min_positions:
                position_penalty = (min_positions - significant_positions) * 10.0
                obj_value += position_penalty
            
            return obj_value
        
        # Multiple optimization runs with different starting points
        # This simulates the quantum annealing process of exploring different solutions
        n_attempts = 10
        best_result = None
        best_value = float('inf') if objective_type == "minimize" else float('-inf')
        
        for attempt in range(n_attempts):
            # Create a random starting point near the initial weights
            if attempt == 0:
                # First attempt uses the initial weights
                x0 = initial_weights
            else:
                # Subsequent attempts use random variations
                x0 = np.copy(initial_weights)
                # Add random noise
                noise = np.random.normal(0, 0.05, size=n_assets)
                x0 += noise
                # Normalize to sum to 1
                x0 = np.maximum(0, x0)  # Ensure non-negative
                x0 /= np.sum(x0)
            
            # Run optimization
            result = minimize(
                modified_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=[sum_constraint],
                options={'maxiter': self.parameters["max_iterations"], 'ftol': self.parameters["convergence_threshold"]}
            )
            
            # Check if this is the best result
            if result.success:
                if (objective_type == "minimize" and result.fun < best_value) or \
                   (objective_type == "maximize" and result.fun > best_value):
                    best_value = result.fun
                    best_result = result
        
        if best_result is None:
            raise ValueError("Optimization failed to converge")
        
        # Extract optimized weights and create result dictionary
        optimized_weights = best_result.x
        
        # Ensure weights sum to 1 (fix any numerical precision issues)
        optimized_weights = np.maximum(0, optimized_weights)  # Ensure non-negative
        optimized_weights /= np.sum(optimized_weights)
        
        return {
            "optimized_weights": optimized_weights,
            "objective_value": best_value,
            "iterations": best_result.nit,
            "success": best_result.success,
            "message": best_result.message
        }
    
    def _process_optimization_results(self, 
                                     companies: List[str],
                                     initial_weights: np.ndarray,
                                     optimization_result: Dict[str, Any],
                                     current_exposures: List[Dict[str, Any]],
                                     risk_scores: np.ndarray) -> Dict[str, Any]:
        """
        Process and format optimization results
        
        Args:
            companies: List of company identifiers
            initial_weights: Initial portfolio weights
            optimization_result: Raw optimization results
            current_exposures: Current exposure data
            risk_scores: Company risk scores
            
        Returns:
            Formatted optimization results
        """
        # Extract optimized weights
        optimized_weights = optimization_result["optimized_weights"]
        
        # Calculate total exposure amount
        total_exposure = sum(exp.get("amount", 0) for exp in current_exposures)
        
        # Create company mapping for additional data
        company_map = {exp.get("company_ticker"): exp for exp in current_exposures}
        
        # Create optimized exposure list
        optimized_exposure = []
        
        for i, company_id in enumerate(companies):
            # Get current exposure data
            current = company_map.get(company_id, {})
            company_name = current.get("company_name", company_id)
            
            # Calculate exposure amounts
            initial_amount = initial_weights[i] * total_exposure
            optimized_amount = optimized_weights[i] * total_exposure
            change_amount = optimized_amount - initial_amount
            change_percent = (change_amount / initial_amount) if initial_amount > 0 else 0
            
            # Only include companies with meaningful exposure
            if optimized_amount >= total_exposure * 0.001:  # 0.1% threshold
                optimized_exposure.append({
                    "company_id": company_id,
                    "company_name": company_name,
                    "initial_weight": initial_weights[i],
                    "optimized_weight": optimized_weights[i],
                    "weight_change": optimized_weights[i] - initial_weights[i],
                    "exposure_amount": optimized_amount,
                    "change_amount": change_amount,
                    "change_percent": change_percent,
                    "risk_score": risk_scores[i]
                })
        
        # Sort by exposure amount (descending)
        optimized_exposure.sort(key=lambda x: x["exposure_amount"], reverse=True)
        
        # Calculate portfolio risk metrics
        initial_risk = np.sum(initial_weights * risk_scores)
        optimized_risk = np.sum(optimized_weights * risk_scores)
        risk_reduction = (initial_risk - optimized_risk) / initial_risk if initial_risk > 0 else 0
        
        # Calculate portfolio statistics
        initial_concentration = np.sum(initial_weights ** 2)  # Herfindahl-Hirschman Index
        optimized_concentration = np.sum(optimized_weights ** 2)
        
        # Count positions with meaningful allocation
        initial_positions = np.sum(initial_weights > 0.001)
        optimized_positions = np.sum(optimized_weights > 0.001)
        
        # Create summary metrics
        metrics = {
            "risk_score": {
                "initial": initial_risk,
                "optimized": optimized_risk,
                "reduction": risk_reduction
            },
            "diversification": {
                "initial_concentration": initial_concentration,
                "optimized_concentration": optimized_concentration,
                "initial_positions": int(initial_positions),
                "optimized_positions": int(optimized_positions)
            },
            "turnover": np.sum(np.abs(optimized_weights - initial_weights)) / 2.0  # Portfolio turnover
        }
        
        return {
            "optimized_exposure": optimized_exposure,
            "metrics": metrics,
            "objective_value": optimization_result["objective_value"],
            "iterations": optimization_result["iterations"]
        }