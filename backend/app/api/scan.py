import asyncio
from backend.app.api.admin import manager

import numpy as np
from ml.fusion.strategies import WeightedSumFusion

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from backend.app.db.session import get_db
from backend.app.models.user import User
from backend.app.models.scan import Scan as ScanModel
from backend.app.api.deps import get_current_user
from backend.app.schemas.scan import ScanRequest, ScanResponse, ScanResult

# We will initialize these in main.py startup events
ml_pipeline = None
rule_engine = None
explainer = None
fusion_strategy = None

router = APIRouter()

import urllib.parse
import json
from ml.features.lexical import extract_lexical_features
from ml.features.schema import CURRENT_FEATURE_SCHEMA_VERSION
import pandas as pd

def _run_scan_pipeline(url: str, stage: str, deep_features: dict = None) -> ScanResult:
    # 1. Normalize
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    domain = parsed.hostname or ""
    
    # 2. Extract Features
    features = extract_lexical_features(url)
    if deep_features:
        features.update(deep_features)
        
    # 3. Rule Engine
    rule_res = rule_engine.evaluate(url, features)
    
    # 4. ML Prediction
    feature_cols = explainer.feature_cols
    # Fill missing deep features with defaults for ML if not present (since ML was trained on lexical, wait)
    # The XGBoost model was trained only on lexical features!
    # Deep features aren't even in the ML model yet.
    df_single = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])
    
    # ML Pipeline has scaling built in now
    ml_prob = float(ml_pipeline.predict_proba(df_single)[0][1])
    
    # 5. Fusion (OR Logic — evaluated as best performer)
    rule_score = rule_res["normalized_score"]
    fusion = WeightedSumFusion(alpha=0.6, threshold=0.6)
    
    fused_score = float(fusion.predict_proba(
        np.array([ml_prob]), 
        np.array([rule_score]),
        known_malicious=np.array([features.get("is_known_malicious", False)])
    )[0])
    
    # Three-level risk thresholds (calibrated from validation)
    if fused_score >= 0.20:
        risk_level = "HIGH_RISK"
    elif fused_score >= 0.08:
        risk_level = "SUSPICIOUS"
    else:
        risk_level = "LOW_RISK"
    
    # Build recommendation based on risk level
    recommendations = {
        "HIGH_RISK": "Do not enter credentials. Leave this site.",
        "SUSPICIOUS": "Proceed with caution. Verify the URL carefully.",
        "LOW_RISK": "Little evidence of phishing, but always verify."
    }
    
    # Override rule engine explanation with fused result
    rule_res["user_explanation"]["risk_level"] = risk_level
    rule_res["user_explanation"]["recommendation"] = recommendations[risk_level]
            
    shap_dict = explainer.get_shap_values(features)
    user_exp = explainer.build_user_explanation(risk_level, rule_res["triggered_rules"], shap_dict)
    
    scan_id = str(uuid.uuid4())
    
    print("=== BACKEND DIAGNOSTIC TRACE ===")
    print(f"REQUEST URL: {url}")
    print(f"NORMALIZED URL: {url}")
    print(f"MODEL VERSION: xgboost_v1.0")
    print(f"FEATURE SCHEMA VERSION: {CURRENT_FEATURE_SCHEMA_VERSION}")
    print("RULES:")
    for cat in rule_res.get("category_scores", {}):
        print(f"{cat} = {rule_res['category_scores'][cat]}")
    print(f"NORMALIZED RULE SCORE = {rule_score}")
    print(f"RAW ML OUTPUT = {ml_prob}")
    print(f"CALIBRATED ML PROBABILITY = {ml_prob}")
    print("DEEP FEATURES:")
    if deep_features:
        for k, v in deep_features.items():
            print(f"{k} = {v}")
    else:
        print("none")
    print(f"FUSED SCORE = {fused_score}")
    print(f"RISK THRESHOLD = 0.20 / 0.08")
    print(f"FINAL RISK = {risk_level}")
    print("================================")
    
    return ScanResult(
        scan_id=scan_id,
        url=url,
        domain=domain,
        risk_level=risk_level,
        ml_probability=ml_prob,
        rule_score=rule_score,
        fused_score=fused_score,
        stage=stage,
        triggered_rules=rule_res["triggered_rules"],
        explanation=user_exp,
        analyst_explanation=None,
        model_version="xgboost_v1.0",
        feature_schema_version=CURRENT_FEATURE_SCHEMA_VERSION,
        rule_config_version=rule_engine.config["version"],
        scan_timestamp=datetime.utcnow(),
        deep_analysis_available=(stage=="deep"),
        metadata_failures=[]
    )

@router.post("/", response_model=ScanResponse)
async def scan_url(req: ScanRequest, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    if len(req.url) > 2048:
        raise HTTPException(status_code=400, detail="URL too long")
        
    try:
        # Fast Scan
        result = _run_scan_pipeline(req.url, stage="fast")
        
        # Persist
        db_scan = ScanModel(
            id=result.scan_id,
            user_id=current_user.id,
            url=result.url,
            domain=result.domain,
            risk_level=result.risk_level,
            ml_probability=result.ml_probability,
            rule_score=result.rule_score,
            fused_score=result.fused_score,
            stage=result.stage,
            triggered_rules_json=json.dumps([r if isinstance(r, dict) else r.dict() for r in result.triggered_rules]),
            explanation_json=json.dumps(result.explanation if isinstance(result.explanation, dict) else result.explanation.dict())
        )
        db.add(db_scan)
        await db.commit()
        asyncio.create_task(manager.broadcast({
            "type": "new_scan",
            "scan_id": result.scan_id,
            "url": result.url,
            "domain": result.domain,
            "risk_level": result.risk_level,
            "fused_score": result.fused_score,
            "stage": result.stage,
            "timestamp": result.scan_timestamp.isoformat()
        }))
        
        # Recommend deep analysis if fused_score is borderline
        deep_rec = (0.2 < result.fused_score < 0.8)
        
        return ScanResponse(
            scan_id=result.scan_id,
            url=result.url,
            domain=result.domain,
            risk_level=result.risk_level,
            confidence=result.fused_score,
            explanation=result.explanation,
            deep_analysis_recommended=deep_rec,
            model_version=result.model_version,
            timestamp=result.scan_timestamp
        )
    except Exception as e:
        import uuid
        from datetime import datetime
        print(f"Error in fast scan: {e}")
        return ScanResponse(
            scan_id=str(uuid.uuid4()),
            url=req.url,
            domain=urllib.parse.urlparse(req.url).hostname or "",
            risk_level="ANALYSIS_UNAVAILABLE",
            confidence=0.0,
            explanation={
                "risk_level": "ANALYSIS_UNAVAILABLE",
                "top_reasons": ["Analysis service is currently unavailable.", str(e)],
                "recommendation": "Use your own judgment or try again later."
            },
            deep_analysis_recommended=False,
            model_version="unknown",
            timestamp=datetime.utcnow()
        )

@router.post("/deep", response_model=ScanResponse)
async def deep_scan_url(req: ScanRequest, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    import subprocess
    import json
    
    # Run the worker as a subprocess
    try:
        proc = subprocess.run(
            ["python", "-m", "worker.main", req.url],
            capture_output=True,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            raise Exception("Worker failed")
            
        deep_features = json.loads(proc.stdout)
    except Exception as e:
        raise HTTPException(status_code=503, detail="Deep analysis unavailable: " + str(e))
        
    result = _run_scan_pipeline(req.url, stage="deep", deep_features=deep_features)
    
    db_scan = ScanModel(
        id=result.scan_id,
        user_id=current_user.id,
        url=result.url,
        domain=result.domain,
        risk_level=result.risk_level,
        ml_probability=result.ml_probability,
        rule_score=result.rule_score,
        fused_score=result.fused_score,
        stage=result.stage,
        triggered_rules_json=json.dumps([r if isinstance(r, dict) else r.dict() for r in result.triggered_rules]),
        explanation_json=json.dumps(result.explanation if isinstance(result.explanation, dict) else result.explanation.dict())
    )
    db.add(db_scan)
    await db.commit()
    
    return ScanResponse(
        scan_id=result.scan_id,
        url=result.url,
        domain=result.domain,
        risk_level=result.risk_level,
        confidence=result.fused_score,
        explanation=result.explanation,
        deep_analysis_recommended=False,
        model_version=result.model_version,
        timestamp=result.scan_timestamp
    )
    
@router.get("/history")
async def get_history(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    result = await db.execute(select(ScanModel).where(ScanModel.user_id == current_user.id).order_by(ScanModel.timestamp.desc()).limit(50))
    scans = result.scalars().all()
    return scans
