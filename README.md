# PhishGuard: Hybrid Explainable Phishing Detection System

PhishGuard is an end-to-end, machine learning-powered anti-phishing system designed to protect users from malicious websites in real-time. It combines a robust Machine Learning model (XGBoost) with a deterministic Rule Engine and Deep Scan analysis to provide accurate, explainable risk assessments.

## Components
- **FastAPI Backend:** Orchestrates ML prediction, rule engine evaluation, and deep scan web scraping with strict SSRF protection.
- **Chrome Extension (Manifest V3):** Intercepts browser navigation and overlays a full-screen warning for `HIGH_RISK` and `SUSPICIOUS` sites.
- **Real-Time Admin Dashboard:** A React SPA that connects via WebSockets to provide a live feed of scanned URLs, system health, and overall threat distribution.

## Quick Start (Docker)

To run the entire system locally:

1. Clone the repository and navigate to the root directory.
2. Build and start the containers:
   ```bash
   docker compose up --build -d
   ```
3. Initialize the admin user:
   ```bash
   # Make sure your virtual environment is active, then run:
   python -m scripts.create_admin
   ```
4. Access the **Admin Dashboard** at [http://localhost:3000](http://localhost:3000).

## Chrome Extension Setup

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle on **Developer mode** in the top right corner.
3. Click **Load unpacked** and select the `extension/dist` folder located in this repository.
4. Click the PhishGuard puzzle piece in the browser toolbar to register or log in.
