from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.app.api import auth, scan
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
        paths = glob("ml/models/xgboost_*")
        latest_path = sorted(paths)[-1]
        pipeline = joblib.load(os.path.join(latest_path, "model.joblib"))
        
        import json
        with open(os.path.join(latest_path, "metadata.json")) as f:
            metadata = json.load(f)
            feature_cols = metadata["features"]
            
        scan_module.ml_pipeline = pipeline
        scan_module.rule_engine = RuleEngine()
        scan_module.explainer = Explainer(pipeline, feature_cols)
        print("Models loaded successfully.")
    except Exception as e:
        print("Failed to load models:", e)
        
    yield
    # Shutdown
    await engine.dispose()

app = FastAPI(title="PhishGuard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(scan.router, prefix="/scan", tags=["scan"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
