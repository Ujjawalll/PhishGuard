from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.app.api import auth, scan, admin
from backend.app.db.session import engine, Base
import backend.app.api.scan as scan_module
import os
import joblib
from glob import glob
from ml.rules.engine import RuleEngine
from ml.explainability.explainer import Explainer
import pandas as pd

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    try:
        import json
        with open("configs/production.json") as f:
            prod_config = json.load(f)
            
        latest_path = prod_config["model"]["artifact_path"]
        pipeline = joblib.load(os.path.join(latest_path, "model.joblib"))
        from ml.features.schema import CURRENT_FEATURE_SCHEMA_VERSION
        with open(os.path.join(latest_path, "metadata.json")) as f:
            metadata = json.load(f)
            feature_cols = metadata["features"]
            
        from ml.features.schema import FEATURE_SCHEMA
        if feature_cols != FEATURE_SCHEMA:
            raise ValueError("Model feature schema does not match canonical FEATURE_SCHEMA")
            
        scan_module.ml_pipeline = pipeline
        scan_module.rule_engine = RuleEngine()
        scan_module.explainer = Explainer(pipeline, feature_cols)
        scan_module.prod_config = prod_config
        
        print("=== MODEL LOADING VERIFICATION ===")
        print(f"MODEL_PATH: {latest_path}")
        print(f"MODEL_TYPE: XGBoost (Calibrated)")
        print(f"MODEL_VERSION: {metadata.get('model_version', '1.0')}")
        print(f"FEATURE_SCHEMA_VERSION: {CURRENT_FEATURE_SCHEMA_VERSION}")
        print(f"MODEL_ARTIFACT_TIMESTAMP: {metadata.get('training_timestamp', 'Unknown')}")
        print("==================================")
        
    except Exception as e:
        print("Failed to load models:", e)
        scan_module.ml_pipeline = None # Force ANALYSIS_UNAVAILABLE
        
    yield
    # Shutdown
    await engine.dispose()

app = FastAPI(title="PhishGuard API", version="1.0.0", lifespan=lifespan)

ALLOWED_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "chrome-extension://*,http://localhost:3000,http://localhost:5173"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scan.router, prefix="/scan", tags=["scan"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/model")
def model_info():
    import json, os
    from glob import glob

    model_version = "not_loaded"
    rule_version = "not_loaded"
    
    if scan_module.rule_engine:
        rule_version = scan_module.rule_engine.config.get("version", "unknown")
    
    # Read model version from saved metadata
    try:
        paths = glob("ml/models/xgboost_*")
        latest_path = sorted(paths)[-1]
        with open(os.path.join(latest_path, "metadata.json")) as f:
            meta = json.load(f)
            model_version = meta.get("model_version", os.path.basename(latest_path))
    except Exception:
        pass

    return {
        "model_name": "xgboost",
        "model_version": model_version,
        "feature_schema_version": "v1.0",
        "rule_config_version": rule_version,
        "fusion_strategy": "or_logic"
    }
