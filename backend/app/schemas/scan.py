from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime

class RuleResult(BaseModel):
    rule_id: str
    triggered: bool
    score: float
    severity: Literal["Low", "Medium", "High", "Critical"]
    description: str
    evidence: str

class UserExplanation(BaseModel):
    risk_level: Literal["SAFE", "SUSPICIOUS", "HIGH_RISK"]
    top_reasons: list[str]
    recommendation: str

class AnalystExplanation(BaseModel):
    fused_score: float
    ml_probability: float
    triggered_rules: list[RuleResult]
    top_shap_features: dict[str, float]
    feature_values: dict[str, float | str | int | None]
    metadata: dict[str, str]

class ScanResult(BaseModel):
    scan_id: str
    url: str
    domain: str
    risk_level: Literal["SAFE", "SUSPICIOUS", "HIGH_RISK"]
    ml_probability: float
    rule_score: float
    fused_score: float
    stage: Literal["fast", "deep"]
    triggered_rules: list[RuleResult]
    explanation: UserExplanation
    analyst_explanation: Optional[AnalystExplanation] = None
    model_version: str
    feature_schema_version: str
    rule_config_version: str
    scan_timestamp: datetime
    deep_analysis_available: bool
    metadata_failures: list[str]

class ScanRequest(BaseModel):
    url: str

class ScanResponse(BaseModel):
    scan_id: str
    url: str
    domain: str
    risk_level: Literal["SAFE", "SUSPICIOUS", "HIGH_RISK"]
    confidence: float
    explanation: UserExplanation
    deep_analysis_recommended: bool
    model_version: str
    timestamp: datetime
