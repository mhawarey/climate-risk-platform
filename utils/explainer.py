"""
Explainable AI module for climate risk assessment.
Provides natural language explanations for risk scores and model decisions.
"""

import logging
from typing import Dict, List, Any, Tuple
import json

import numpy as np


class RiskExplainer:
    """
    Explainable AI module for climate risk assessments
    Generates human-readable explanations for risk scores and model outputs
    """
    
    def __init__(self):
        self.logger = logging.getLogger("risk_explainer")
        self.logger.info("Initializing RiskExplainer")
    
    def explain_company_risk(self, risk_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explanations for company risk assessment results
        
        Args:
            risk_results: Risk calculation results
            
        Returns:
            Dictionary with natural language explanations
        """
        explanations = {
            "summary": self._generate_risk_summary(risk_results),
            "physical_risk": {},
            "transition_risk": {},
            "factor_contributions": self._explain_risk_factors(risk_results)
        }
        
        # Generate physical risk explanations for each scenario
        if "physical_risks" in risk_results:
            for scenario, risk_data in risk_results["physical_risks"].items():
                explanations["physical_risk"][scenario] = self._explain_physical_risk(risk_data, scenario)
        
        # Generate transition risk explanations for each scenario
        if "transition_risks" in risk_results:
            for scenario, risk_data in risk_results["transition_risks"].items():
                explanations["transition_risk"][scenario] = self._explain_transition_risk(risk_data, scenario)
        
        # Generate recommendations based on risk profile
        explanations["recommendations"] = self._generate_recommendations(risk_results)
        
        return explanations
    
    def _generate_risk_summary(self, risk_results: Dict[str, Any]) -> str:
        """Generate a summary explanation of overall risk results"""
        
        # Extract key metrics
        company_id = risk_results.get("company_id", "Unknown")
        
        # Extract combined metrics if available
        combined_metrics = risk_results.get("combined_metrics", {})
        overall = combined_metrics.get("overall", {})
        
        avg_score = overall.get("average_score", 0)
        worst_case = overall.get("worst_case_scenario", "unknown")
        
        # Get scenario-specific scores
        scenarios = {}
        for scenario, metrics in combined_metrics.items():
            if scenario != "overall":
                scenarios[scenario] = metrics.get("combined_score", 0)
        
        # Generate summary text
        if avg_score == 0:
            return f"No significant climate risks were identified for this company."
        
        # Risk level description
        if avg_score < 3:
            risk_level = "low"
        elif avg_score < 5:
            risk_level = "moderate"
        elif avg_score < 7:
            risk_level = "high"
        else:
            risk_level = "very high"
        
        summary = f"Overall, this company faces a {risk_level} level of climate risk "
        summary += f"with an average risk score of {avg_score:.1f} out of 10. "
        
        # Add scenario comparison
        if scenarios:
            best_scenario = min(scenarios.items(), key=lambda x: x[1])
            worst_scenario = max(scenarios.items(), key=lambda x: x[1])
            
            scenario_names = {
                "ipcc_ssp119": "rapid transition",
                "ipcc_ssp126": "strong mitigation",
                "ipcc_ssp245": "moderate action",
                "ipcc_ssp370": "limited policy action",
                "ipcc_ssp585": "high emissions"
            }
            
            best_name = scenario_names.get(best_scenario[0], best_scenario[0])
            worst_name = scenario_names.get(worst_scenario[0], worst_scenario[0])
            
            summary += f"The company performs best under a {best_name} scenario "
            summary += f"(score: {best_scenario[1]:.1f}) and worst under a {worst_name} scenario "
            summary += f"(score: {worst_scenario[1]:.1f}). "
        
        # Add risk balance
        if "physical_risks" in risk_results and "transition_risks" in risk_results:
            physical_scores = []
            transition_scores = []
            
            for scenario in scenarios.keys():
                if scenario in risk_results["physical_risks"]:
                    physical_scores.append(risk_results["physical_risks"][scenario].get("overall_score", 0))
                
                if scenario in risk_results["transition_risks"]:
                    transition_scores.append(risk_results["transition_risks"][scenario].get("overall_score", 0))
            
            if physical_scores and transition_scores:
                avg_physical = sum(physical_scores) / len(physical_scores)
                avg_transition = sum(transition_scores) / len(transition_scores)
                
                if avg_physical > avg_transition * 1.5:
                    summary += "The company's climate risk exposure is predominantly physical rather than transition-related. "
                elif avg_transition > avg_physical * 1.5:
                    summary += "The company's climate risk exposure is predominantly transition-related rather than physical. "
                else:
                    summary += "The company faces a balanced exposure to both physical and transition climate risks. "
        
        return summary
    
    def _explain_physical_risk(self, risk_data: Dict[str, Any], scenario: str) -> str:
        """Generate explanation for physical risk results"""
        
        # Extract key metrics
        overall_score = risk_data.get("overall_score", 0)
        risk_breakdown = risk_data.get("risk_breakdown", {})
        asset_risks = risk_data.get("asset_risk_scores", [])
        
        # Scenario names mapping
        scenario_names = {
            "ipcc_ssp119": "1.5°C (low emissions)",
            "ipcc_ssp126": "2°C (strong mitigation)",
            "ipcc_ssp245": "2.5-3°C (moderate action)",
            "ipcc_ssp370": "3-4°C (limited policy action)",
            "ipcc_ssp585": "4-5°C (high emissions)"
        }
        
        scenario_name = scenario_names.get(scenario, scenario)
        
        # Risk level description
        if overall_score < 3:
            risk_level = "low"
        elif overall_score < 5:
            risk_level = "moderate"
        elif overall_score < 7:
            risk_level = "high"
        else:
            risk_level = "very high"
        
        explanation = f"Under a {scenario_name} scenario, this company faces {risk_level} physical climate risks "
        explanation += f"with an overall score of {overall_score:.1f} out of 10. "
        
        # Add risk breakdown by type if available
        if risk_breakdown:
            # Sort risk types by score
            sorted_risks = sorted(risk_breakdown.items(), key=lambda x: x[1].get("score", 0), reverse=True)
            
            # Add top risks
            top_risks = sorted_risks[:3]
            risk_type_names = {
                "flood": "flooding",
                "hurricane": "tropical cyclones",
                "fire": "wildfires",
                "drought": "drought conditions",
                "extreme_heat": "extreme heat events"
            }
            
            if top_risks:
                explanation += "The most significant physical risks include "
                
                for i, (risk_type, risk_info) in enumerate(top_risks):
                    readable_name = risk_type_names.get(risk_type, risk_type)
                    score = risk_info.get("score", 0)
                    
                    if i > 0:
                        explanation += ", " if i < len(top_risks) - 1 else " and "
                    
                    explanation += f"{readable_name} (score: {score:.1f})"
                
                explanation += ". "
        
        # Add asset-specific insights if available
        if asset_risks:
            # Sort assets by risk score
            sorted_assets = sorted(asset_risks, key=lambda x: x.get("overall_score", 0), reverse=True)
            
            # Count assets by type
            asset_type_counts = {}
            for asset in asset_risks:
                asset_type = asset.get("asset_type", "unknown")
                asset_type_counts[asset_type] = asset_type_counts.get(asset_type, 0) + 1
            
            # Get high-risk assets (score > 7)
            high_risk_assets = [a for a in sorted_assets if a.get("overall_score", 0) > 7]
            high_risk_count = len(high_risk_assets)
            
            if high_risk_count > 0:
                explanation += f"{high_risk_count} of {len(asset_risks)} assets ({high_risk_count/len(asset_risks)*100:.0f}%) "
                explanation += f"are at high risk under this scenario. "
                
                # Add details about most vulnerable asset
                if high_risk_assets:
                    most_vulnerable = high_risk_assets[0]
                    explanation += f"The most vulnerable asset is {most_vulnerable.get('asset_name', 'Unknown')} "
                    explanation += f"({most_vulnerable.get('asset_type', 'Unknown')}) "
                    explanation += f"with a risk score of {most_vulnerable.get('overall_score', 0):.1f}. "
            else:
                explanation += f"None of the company's assets are at high risk under this scenario. "
        
        # Add financial impact if available
        var_data = risk_data.get("value_at_risk", {})
        if var_data:
            var_95 = var_data.get("var_95", 0)
            expected_loss = var_data.get("expected_loss", 0)
            
            if var_95 > 0:
                explanation += f"The estimated Value at Risk (95% confidence) is ${var_95/1e6:.1f} million, "
                explanation += f"with an expected loss of ${expected_loss/1e6:.1f} million. "
        
        return explanation
    
    def _explain_transition_risk(self, risk_data: Dict[str, Any], scenario: str) -> str:
        """Generate explanation for transition risk results"""
        
        # Extract key metrics
        overall_score = risk_data.get("overall_score", 0)
        risk_drivers = risk_data.get("risk_drivers", {})
        financial_impact = risk_data.get("financial_impact", {})
        monte_carlo = risk_data.get("monte_carlo_results", {})
        
        # Scenario names mapping
        scenario_names = {
            "ipcc_ssp119": "1.5°C (low emissions)",
            "ipcc_ssp126": "2°C (strong mitigation)",
            "ipcc_ssp245": "2.5-3°C (moderate action)",
            "ipcc_ssp370": "3-4°C (limited policy action)",
            "ipcc_ssp585": "4-5°C (high emissions)"
        }
        
        scenario_name = scenario_names.get(scenario, scenario)
        
        # Risk level description
        if overall_score < 3:
            risk_level = "low"
        elif overall_score < 5:
            risk_level = "moderate"
        elif overall_score < 7:
            risk_level = "high"
        else:
            risk_level = "very high"
        
        explanation = f"Under a {scenario_name} scenario, this company faces {risk_level} transition risks "
        explanation += f"with an overall score of {overall_score:.1f} out of 10. "
        
        # Add risk driver explanation if available
        if risk_drivers:
            carbon_price = risk_drivers.get("carbon_price", {})
            demand_changes = risk_drivers.get("demand_changes", {})
            
            carbon_2030 = carbon_price.get("2030", 0)
            carbon_2050 = carbon_price.get("2050", 0)
            demand_2030 = demand_changes.get("2030", 1.0)
            demand_2050 = demand_changes.get("2050", 1.0)
            
            explanation += f"Under this scenario, carbon prices are projected to reach "
            explanation += f"${carbon_2030:.0f}/tCO2 by 2030 and ${carbon_2050:.0f}/tCO2 by 2050. "
            
            if demand_2050 < 0.8:
                explanation += f"Demand for oil and gas is projected to decline significantly "
                explanation += f"to {demand_2050*100:.0f}% of current levels by 2050. "
            elif demand_2050 < 0.95:
                explanation += f"Demand for oil and gas is projected to decline moderately "
                explanation += f"to {demand_2050*100:.0f}% of current levels by 2050. "
            elif demand_2050 < 1.05:
                explanation += f"Demand for oil and gas is projected to remain relatively stable through 2050. "
            else:
                explanation += f"Demand for oil and gas is projected to increase "
                explanation += f"to {demand_2050*100:.0f}% of current levels by 2050. "
        
        # Add financial impact explanation if available
        if financial_impact and monte_carlo:
            stranded_risk = financial_impact.get("stranded_asset_risk", 0)
            revenue_impact = financial_impact.get("revenue_impact", 0)
            carbon_cost = financial_impact.get("carbon_cost_impact", 0)
            
            npv_impact = monte_carlo.get("npv_impact_mean", 0)
            impact_pct = monte_carlo.get("impact_percentage", 0)
            
            if abs(npv_impact) > 0:
                explanation += f"The projected NPV impact of transition risks is "
                
                if npv_impact < 0:
                    explanation += f"negative ${abs(npv_impact)/1e9:.1f} billion "
                    explanation += f"({abs(impact_pct)*100:.1f}% of company value). "
                else:
                    explanation += f"positive ${npv_impact/1e9:.1f} billion "
                    explanation += f"({impact_pct*100:.1f}% of company value). "
            
            # Add breakdown of financial impacts
            impacts = []
            if abs(stranded_risk) > 1e6:
                impacts.append(f"stranded asset risk (${abs(stranded_risk)/1e9:.1f}B)")
            if abs(revenue_impact) > 1e6:
                impacts.append(f"revenue impacts (${abs(revenue_impact)/1e9:.1f}B)")
            if abs(carbon_cost) > 1e6:
                impacts.append(f"carbon costs (${abs(carbon_cost)/1e9:.1f}B)")
            
            if impacts:
                explanation += "Key financial impacts include " + ", ".join(impacts) + ". "
        
        return explanation
    
    def _explain_risk_factors(self, risk_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Explain key risk factors and their contributions"""
        
        factors = []
        
        # Identify key physical risk factors
        if "physical_risks" in risk_results:
            physical_risks = risk_results["physical_risks"]
            
            # Collect all risk types across scenarios
            risk_types = set()
            for scenario, risk_data in physical_risks.items():
                risk_breakdown = risk_data.get("risk_breakdown", {})
                risk_types.update(risk_breakdown.keys())
            
            # Calculate average score for each risk type
            for risk_type in risk_types:
                scores = []
                
                for scenario, risk_data in physical_risks.items():
                    risk_breakdown = risk_data.get("risk_breakdown", {})
                    if risk_type in risk_breakdown:
                        scores.append(risk_breakdown[risk_type].get("score", 0))
                
                if scores:
                    avg_score = sum(scores) / len(scores)
                    
                    # Only include significant factors
                    if avg_score >= 3:
                        # Risk type names mapping
                        risk_type_names = {
                            "flood": "Flooding",
                            "hurricane": "Tropical Cyclones",
                            "fire": "Wildfires",
                            "drought": "Drought Conditions",
                            "extreme_heat": "Extreme Heat"
                        }
                        
                        factor = {
                            "name": risk_type_names.get(risk_type, risk_type.title()),
                            "type": "physical",
                            "score": avg_score,
                            "description": self._generate_factor_description(risk_type, "physical", avg_score)
                        }
                        
                        factors.append(factor)
        
        # Identify key transition risk factors
        if "transition_risks" in risk_results:
            transition_risks = risk_results["transition_risks"]
            
            # Calculate policy risk (carbon price)
            policy_scores = []
            for scenario, risk_data in transition_risks.items():
                risk_drivers = risk_data.get("risk_drivers", {})
                carbon_price = risk_drivers.get("carbon_price", {})
                
                if carbon_price:
                    carbon_2050 = carbon_price.get("2050", 0)
                    # Normalize to 0-10 scale
                    policy_score = min(10, carbon_2050 / 50)
                    policy_scores.append(policy_score)
            
            if policy_scores:
                avg_policy_score = sum(policy_scores) / len(policy_scores)
                
                if avg_policy_score >= 3:
                    factor = {
                        "name": "Carbon Pricing",
                        "type": "transition",
                        "score": avg_policy_score,
                        "description": self._generate_factor_description("carbon_price", "transition", avg_policy_score)
                    }
                    
                    factors.append(factor)
            
            # Calculate market risk (demand changes)
            market_scores = []
            for scenario, risk_data in transition_risks.items():
                risk_drivers = risk_data.get("risk_drivers", {})
                demand_changes = risk_drivers.get("demand_changes", {})
                
                if demand_changes:
                    demand_2050 = demand_changes.get("2050", 1.0)
                    # Convert to score (1.0 = no change, <1.0 = declining demand)
                    # Score is higher for greater demand decline
                    market_score = max(0, min(10, (1.0 - demand_2050) * 10))
                    market_scores.append(market_score)
            
            if market_scores:
                avg_market_score = sum(market_scores) / len(market_scores)
                
                if avg_market_score >= 3:
                    factor = {
                        "name": "Demand Reduction",
                        "type": "transition",
                        "score": avg_market_score,
                        "description": self._generate_factor_description("demand_change", "transition", avg_market_score)
                    }
                    
                    factors.append(factor)
            
            # Calculate stranded asset risk
            stranded_scores = []
            for scenario, risk_data in transition_risks.items():
                financial_impact = risk_data.get("financial_impact", {})
                
                if financial_impact:
                    stranded_risk = financial_impact.get("stranded_asset_risk", 0)
                    
                    # Normalize to 0-10 scale (assuming stranded risk is in $)
                    # This is a simplified approach
                    stranded_score = min(10, stranded_risk / 1e9)
                    stranded_scores.append(stranded_score)
            
            if stranded_scores:
                avg_stranded_score = sum(stranded_scores) / len(stranded_scores)
                
                if avg_stranded_score >= 3:
                    factor = {
                        "name": "Stranded Assets",
                        "type": "transition",
                        "score": avg_stranded_score,
                        "description": self._generate_factor_description("stranded_assets", "transition", avg_stranded_score)
                    }
                    
                    factors.append(factor)
        
        # Sort factors by score (descending)
        factors.sort(key=lambda x: x["score"], reverse=True)
        
        return factors
    
    def _generate_factor_description(self, factor_name: str, factor_type: str, score: float) -> str:
        """Generate description for a risk factor"""
        
        # Physical risk factor descriptions
        physical_descriptions = {
            "flood": {
                "high": "Assets are located in areas with high flood risk that is expected to intensify with climate change. This could lead to damage, operational disruptions, and increased insurance costs.",
                "medium": "Some assets face moderate flood risks, potentially requiring adaptive measures such as improved drainage or flood barriers in the medium term.",
                "low": "Flooding poses minimal risk to the company's assets, with only occasional minor impacts expected."
            },
            "hurricane": {
                "high": "Critical assets are located in hurricane-prone regions where storm intensity is projected to increase. This creates risks of catastrophic damage, extended operational disruptions, and rising insurance premiums.",
                "medium": "The company has moderate exposure to hurricane risk, with some assets located in areas that may experience more frequent or intense storms.",
                "low": "Hurricane risk is limited, affecting only a small portion of non-critical assets."
            },
            "fire": {
                "high": "Operations are located in areas with rapidly increasing wildfire risk due to rising temperatures and changing precipitation patterns. This threatens direct asset damage, supply chain disruptions, and air quality impacts.",
                "medium": "Wildfire risk is present for some company assets, particularly in regions experiencing increasing drought conditions.",
                "low": "Wildfire exposure is minimal and limited to peripheral operations only."
            },
            "drought": {
                "high": "Operations depend heavily on water resources in regions projected to experience severe drought conditions. This threatens production capacity, regulatory challenges, and potential conflicts with other water users.",
                "medium": "Water stress poses moderate risks to operations in some regions, potentially requiring investment in water efficiency or alternative sources.",
                "low": "Water stress poses limited operational risks to the company, with adequate resources likely available in most scenarios."
            },
            "extreme_heat": {
                "high": "Operations are concentrated in regions facing extreme temperature increases, threatening worker health and safety, equipment reliability, and energy costs for cooling.",
                "medium": "Rising temperatures will create moderate operational challenges, including potential efficiency losses and increased cooling costs.",
                "low": "Temperature increases are expected to have minimal operational impacts on the company's assets."
            }
        }
        
        # Transition risk factor descriptions
        transition_descriptions = {
            "carbon_price": {
                "high": "The company faces significant exposure to carbon pricing, with the potential for material impacts on operational costs and competitiveness as prices rise under climate policy scenarios.",
                "medium": "Carbon pricing presents moderate risks, particularly in the medium to long term as policies mature and prices increase.",
                "low": "Carbon pricing poses limited risks due to the company's relatively low emissions intensity or operations in regions with limited carbon regulation."
            },
            "demand_change": {
                "high": "The company faces substantial market risks as demand for fossil fuels is projected to decline significantly in medium to long-term horizons, particularly under stringent climate policy scenarios.",
                "medium": "Moderate demand reduction is expected in low-carbon scenarios, requiring business model adaptation in the medium term.",
                "low": "Demand for the company's products is relatively resilient even under climate transition scenarios."
            },
            "stranded_assets": {
                "high": "The company has significant risk of stranded assets, with substantial reserves that may become uneconomic to extract under climate transition scenarios.",
                "medium": "Some assets may become stranded under stringent climate policies, particularly those with higher extraction costs or emissions intensity.",
                "low": "The company has limited exposure to stranded asset risk due to lower reserves, shorter-lived assets, or focus on lower-carbon resources."
            }
        }
        
        # Determine risk level based on score
        if score >= 7:
            level = "high"
        elif score >= 4:
            level = "medium"
        else:
            level = "low"
        
        # Get description based on factor type and name
        if factor_type == "physical" and factor_name in physical_descriptions:
            return physical_descriptions[factor_name][level]
        elif factor_type == "transition" and factor_name in transition_descriptions:
            return transition_descriptions[factor_name][level]
        else:
            # Generic description if specific one not found
            return f"This factor presents a {level} level of risk with a score of {score:.1f} out of 10."
    
    def _generate_recommendations(self, risk_results: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate recommendations based on risk assessment"""
        
        recommendations = []
        
        # Extract combined metrics
        combined_metrics = risk_results.get("combined_metrics", {})
        overall = combined_metrics.get("overall", {})
        avg_score = overall.get("average_score", 0)
        
        # Extract physical and transition risk data
        physical_risks = risk_results.get("physical_risks", {})
        transition_risks = risk_results.get("transition_risks", {})
        
        # Calculate average physical and transition risk scores
        physical_scores = [data.get("overall_score", 0) for data in physical_risks.values()]
        transition_scores = [data.get("overall_score", 0) for data in transition_risks.values()]
        
        avg_physical = sum(physical_scores) / len(physical_scores) if physical_scores else 0
        avg_transition = sum(transition_scores) / len(transition_scores) if transition_scores else 0
        
        # General recommendation based on overall risk
        if avg_score >= 6:
            recommendations.append({
                "category": "Strategic",
                "recommendation": "Develop a comprehensive climate risk management strategy that integrates both physical and transition risks into core business planning.",
                "rationale": "High overall climate risk requires systematic management at the strategic level."
            })
        elif avg_score >= 3:
            recommendations.append({
                "category": "Strategic",
                "recommendation": "Integrate climate risk considerations into existing risk management frameworks and strategic planning processes.",
                "rationale": "Moderate climate risk warrants structured management approach integrated with existing systems."
            })
        
        # Physical risk recommendations
        if avg_physical >= 6:
            # High physical risk
            recommendations.append({
                "category": "Physical Risk",
                "recommendation": "Conduct detailed vulnerability assessments for high-risk assets and develop asset-specific resilience plans.",
                "rationale": "High physical risk exposure requires asset-level intervention and planning."
            })
            
            # Check for specific physical risks
            risk_types = set()
            for scenario, risk_data in physical_risks.items():
                risk_breakdown = risk_data.get("risk_breakdown", {})
                for risk_type, risk_info in risk_breakdown.items():
                    if risk_info.get("score", 0) >= 6:
                        risk_types.add(risk_type)
            
            if "flood" in risk_types or "hurricane" in risk_types:
                recommendations.append({
                    "category": "Physical Risk",
                    "recommendation": "Enhance flood and storm protection for vulnerable assets and review insurance coverage adequacy.",
                    "rationale": "High exposure to precipitation and storm-related risks requires specific protection measures."
                })
            
            if "drought" in risk_types or "extreme_heat" in risk_types:
                recommendations.append({
                    "category": "Physical Risk",
                    "recommendation": "Implement water efficiency measures and develop contingency plans for operations in water-stressed regions.",
                    "rationale": "Exposure to water stress and heat requires resource efficiency and operational adaptation."
                })
        
        elif avg_physical >= 3:
            # Moderate physical risk
            recommendations.append({
                "category": "Physical Risk",
                "recommendation": "Incorporate climate projections into infrastructure planning and maintenance schedules.",
                "rationale": "Moderate physical risk can be managed through proactive planning and incremental adaptation."
            })
        
        # Transition risk recommendations
        if avg_transition >= 6:
            # High transition risk
            recommendations.append({
                "category": "Transition Risk",
                "recommendation": "Develop a low-carbon transition strategy with clear emissions reduction targets and diversification initiatives.",
                "rationale": "High transition risk requires comprehensive strategic response and business model evolution."
            })
            
            # Check for specific transition risks
            has_high_carbon_price_risk = False
            has_high_demand_risk = False
            has_high_stranded_asset_risk = False
            
            for scenario, risk_data in transition_risks.items():
                # Check carbon price risk
                risk_drivers = risk_data.get("risk_drivers", {})
                carbon_price = risk_drivers.get("carbon_price", {})
                if carbon_price and carbon_price.get("2050", 0) >= 100:
                    has_high_carbon_price_risk = True
                
                # Check demand risk
                demand_changes = risk_drivers.get("demand_changes", {})
                if demand_changes and demand_changes.get("2050", 1.0) <= 0.7:
                    has_high_demand_risk = True
                
                # Check stranded asset risk
                financial_impact = risk_data.get("financial_impact", {})
                if financial_impact and financial_impact.get("stranded_asset_risk", 0) >= 1e9:
                    has_high_stranded_asset_risk = True
            
            if has_high_carbon_price_risk:
                recommendations.append({
                    "category": "Transition Risk",
                    "recommendation": "Implement an internal carbon price for capital allocation decisions and develop an emissions reduction strategy.",
                    "rationale": "High carbon price exposure requires embedding carbon considerations in financial decision-making."
                })
            
            if has_high_demand_risk:
                recommendations.append({
                    "category": "Transition Risk",
                    "recommendation": "Diversify business model toward lower-carbon products and services to reduce dependency on fossil fuels.",
                    "rationale": "High market transition risk requires strategic diversification and new growth areas."
                })
            
            if has_high_stranded_asset_risk:
                recommendations.append({
                    "category": "Transition Risk",
                    "recommendation": "Review capital allocation strategy with focus on shortening payback periods and prioritizing flexible, lower-carbon assets.",
                    "rationale": "Stranded asset risk requires more conservative capital allocation approach with climate scenarios as key inputs."
                })
        
        elif avg_transition >= 3:
            # Moderate transition risk
            recommendations.append({
                "category": "Transition Risk",
                "recommendation": "Enhance emissions monitoring and reporting, and evaluate low-carbon technology options relevant to operations.",
                "rationale": "Moderate transition risk requires improved measurement and gradual technology adoption."
            })
        
        # Disclosure recommendations
        if avg_score >= 3:
            recommendations.append({
                "category": "Disclosure",
                "recommendation": "Enhance climate risk disclosure in line with TCFD recommendations, including scenario analysis results.",
                "rationale": "Transparent disclosure of material climate risks is increasingly expected by investors and regulators."
            })
        
        # Governance recommendations
        if avg_score >= 5:
            recommendations.append({
                "category": "Governance",
                "recommendation": "Establish board-level oversight of climate risks and integrate climate KPIs into executive compensation.",
                "rationale": "Significant climate risks warrant governance at the highest organizational level."
            })
        
        return recommendations