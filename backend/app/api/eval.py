import json
import os
import subprocess
import sys
import uuid
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from backend.app.api.deps import get_current_admin_user
from backend.app.models.user import User

router = APIRouter()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JOBS_DIR = os.path.join("experiments", "reports", "jobs")
DEFAULT_REPORT_DIR = os.path.join("experiments", "reports", "rf_vs_xgb")
PLOT_FILES = {
    "confusion_matrices.png",
    "roc_curves.png",
    "pr_curves.png",
    "feature_importances.png",
}
MIN_ROWS = 50


def _job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)


def _read_job(job_id: str) -> dict:
    path = os.path.join(_job_dir(job_id), "job.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Job not found")
    with open(path) as f:
        return json.load(f)


def _write_job(job_id: str, meta: dict) -> None:
    with open(os.path.join(_job_dir(job_id), "job.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _validate_dataset(path: str) -> dict:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    if "url" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'url' column")

    if "label" not in df.columns:
        if "source" in df.columns:
            df["label"] = (df["source"].str.lower() != "tranco").astype(int)
            df.to_csv(path, index=False)
        else:
            raise HTTPException(
                status_code=400,
                detail="CSV must contain a 'label' column (0=legitimate, 1=phishing) or a 'source' column",
            )

    if len(df) < MIN_ROWS:
        raise HTTPException(status_code=400, detail=f"Dataset too small: {len(df)} rows (need >= {MIN_ROWS})")

    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise HTTPException(status_code=400, detail="Dataset must contain both classes (label 0 and label 1)")

    return {
        "rows": int(len(df)),
        "phishing": int(counts.get(1, 0)),
        "legitimate": int(counts.get(0, 0)),
    }


@router.get("/results")
async def get_results(job_id: str | None = None, _: User = Depends(get_current_admin_user)):
    report_dir = _job_dir(job_id) if job_id else DEFAULT_REPORT_DIR
    results_path = os.path.join(report_dir, "results.json")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="No results available yet")
    with open(results_path) as f:
        results = json.load(f)
    results["plots"] = sorted(p for p in PLOT_FILES if os.path.exists(os.path.join(report_dir, p)))
    results["job_id"] = job_id
    return results


@router.get("/plots/{name}")
async def get_plot(name: str, job_id: str | None = None, _: User = Depends(get_current_admin_user)):
    if name not in PLOT_FILES:
        raise HTTPException(status_code=404, detail="Unknown plot")
    report_dir = _job_dir(job_id) if job_id else DEFAULT_REPORT_DIR
    path = os.path.join(report_dir, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(path, media_type="image/png")


@router.get("/jobs")
async def list_jobs(_: User = Depends(get_current_admin_user)):
    if not os.path.isdir(JOBS_DIR):
        return []
    jobs = []
    for job_id in sorted(os.listdir(JOBS_DIR), reverse=True):
        meta_path = os.path.join(JOBS_DIR, job_id, "job.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("status") != "done" and os.path.exists(os.path.join(JOBS_DIR, job_id, "results.json")):
                meta["status"] = "done"
                meta.setdefault("finished_at", datetime.now().isoformat())
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
            jobs.append(meta)
    return jobs


@router.post("/train")
async def start_training(
    file: UploadFile = File(...),
    quick: bool = True,
    _: User = Depends(get_current_admin_user),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    job_dir = _job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)

    dataset_path = os.path.join(job_dir, "dataset.csv")
    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Dataset too large (max 100MB)")
    with open(dataset_path, "wb") as f:
        f.write(contents)

    summary = _validate_dataset(dataset_path)

    meta = {
        "job_id": job_id,
        "status": "queued",
        "filename": os.path.basename(file.filename),
        "created_at": datetime.now().isoformat(),
        "dataset": summary,
        "quick": quick,
    }
    _write_job(job_id, meta)

    cmd = [
        sys.executable, os.path.join("experiments", "rf_vs_xgb_eval.py"),
        "--dataset", dataset_path,
        "--out", job_dir,
    ]
    if quick:
        cmd.append("--quick")

    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    log_path = os.path.join(job_dir, "eval.log")
    log_file = open(log_path, "wb")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    meta.update(status="running", pid=proc.pid)
    _write_job(job_id, meta)

    return {"job_id": job_id, "status": "running", "dataset": summary}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, _: User = Depends(get_current_admin_user)):
    meta = _read_job(job_id)
    proc_alive = meta.get("pid") is not None and _pid_alive(meta["pid"])
    has_results = os.path.exists(os.path.join(_job_dir(job_id), "results.json"))
    if has_results and meta["status"] != "done":
        meta["status"] = "done"
        meta.setdefault("finished_at", datetime.now().isoformat())
        _write_job(job_id, meta)
    elif meta["status"] == "running" and not proc_alive:
        meta["status"] = "failed"
        meta["finished_at"] = datetime.now().isoformat()
        _write_job(job_id, meta)
    if os.path.exists(os.path.join(_job_dir(job_id), "eval.log")):
        with open(os.path.join(_job_dir(job_id), "eval.log"), errors="replace") as f:
            lines = f.readlines()
        meta["log_tail"] = "".join(lines[-15:])
    return meta


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        # On Linux, verify it's not a zombie process
        status_path = f"/proc/{pid}/status"
        if os.path.exists(status_path):
            with open(status_path) as f:
                for line in f:
                    if line.startswith("State:"):
                        return "Z" not in line
        return True
    except OSError:
        return False

