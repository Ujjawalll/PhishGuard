import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class Explainer:
    def __init__(self, model_pipeline, feature_cols: List[str]):
        """
        model_pipeline: Should be the scikit-learn Pipeline containing the scaler and XGBoost.
        feature_cols: List of feature names in order.
        """
        self.pipeline = model_pipeline
        self.feature_cols = feature_cols
        
        # We need the underlying Tree model to use TreeExplainer
        self.model = self.pipeline.named_steps['classifier']
        self.scaler = self.pipeline.named_steps['scaler']
        
        # Initialize SHAP explainer
        # For XGBoost, TreeExplainer is ideal
        self.explainer = shap.TreeExplainer(self.model)

    def get_shap_values(self, features_dict: Dict[str, float]) -> Dict[str, float]:
        """Returns the SHAP contribution of each feature for a single instance."""
        # Convert to DataFrame
        df = pd.DataFrame([features_dict], columns=self.feature_cols)
        
        # We MUST scale the features first since the model was trained on scaled data
        scaled_features = self.scaler.transform(df)
        
        # Get SHAP values
        shap_vals = self.explainer.shap_values(scaled_features)
        
        # Depending on XGBoost objective, shap_vals could be a list (multiclass) or 2D array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1][0]  # Positive class
        elif len(shap_vals.shape) == 2:
            shap_vals = shap_vals[0]
            
        # Map back to feature names
        return {col: float(val) for col, val in zip(self.feature_cols, shap_vals)}

    def build_analyst_explanation(
        self, 
        features_dict: Dict[str, Any], 
        ml_prob: float, 
        rule_score: float, 
        fused_score: float, 
        triggered_rules: List[Dict[str, Any]],
        metadata: Dict[str, str]
    ) -> Dict[str, Any]:
        
        shap_dict = self.get_shap_values(features_dict)
        
        # Sort SHAP features by absolute importance
        top_shap = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
        
        return {
            "fused_score": float(fused_score),
            "ml_probability": float(ml_prob),
            "rule_score": float(rule_score),
            "triggered_rules": triggered_rules,
            "top_shap_features": {k: float(v) for k, v in top_shap.items()},
            "feature_values": {k: (float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v) for k, v in features_dict.items()},
            "metadata": metadata
        }

    def build_user_explanation(
        self,
        risk_level: str,
        triggered_rules: List[Dict[str, Any]],
        shap_dict: Dict[str, float]
    ) -> Dict[str, Any]:
        
        top_reasons = []
        
        # 1. Add rule-based reasons (most severe first)
        sorted_rules = sorted(triggered_rules, key=lambda x: x["score"], reverse=True)
        for r in sorted_rules[:2]:
            top_reasons.append(r["description"])
            
        # 2. Add ML-based reasons if we need more (positive SHAP contributors)
        if len(top_reasons) < 3:
            positive_shap = sorted([item for item in shap_dict.items() if item[1] > 0], key=lambda x: x[1], reverse=True)
            for feat, val in positive_shap:
                if len(top_reasons) >= 3:
                    break
                
                # Friendly mapping for features
                friendly_names = {
                    "url_length": "The URL is unusually long.",
                    "digit_count": "The URL contains an abnormal number of digits.",
                    "url_entropy": "The URL contains a suspicious random sequence of characters.",
                    "suspicious_token_count": "The URL contains deceptive words often used to trick users."
                }
                
                reason = friendly_names.get(feat, f"The structural pattern of the URL is highly unusual ({feat}).")
                if reason not in top_reasons:
                    top_reasons.append(reason)
                    
        # Fallback if entirely safe
        if not top_reasons:
            top_reasons.append("No suspicious indicators were detected.")
            
        recommendation = "Proceed with caution." if risk_level == "SUSPICIOUS" else "Do not enter credentials." if risk_level == "HIGH_RISK" else "Appears safe."
        
        return {
            "risk_level": risk_level,
            "top_reasons": top_reasons,
            "recommendation": recommendation
        }
