# Hybrid Explainable Phishing Detection System: Plan Overview

> **Document role**: The WHAT — scope, architecture, research questions, schedule, and priorities (single source of truth for priorities and timeline).
>
> **Related documents**:
> - [PLANNER.md](./PLANNER.md) — Implementation phases and checklists
> - [AGENT.md](./AGENT.md) — Agent behavioral rules and coding standards
> - [RESEARCH.md](./RESEARCH.md) — Research foundation, threat model, hypotheses, paper outline

---

## 1. Tech Stack

| Component | Technology | Version |
|---|---|---|
| **Language (Backend)** | Python | 3.12+ |
| **Backend Framework** | FastAPI | latest stable |
| **Database** | PostgreSQL | 16+ |
| **ORM / Migrations** | SQLAlchemy + Alembic | latest stable |
| **Authentication** | JWT (python-jose or PyJWT) + bcrypt | — |
| **Language (Extension)** | TypeScript | 5.x+ |
| **Extension UI** | React | 18+ |
| **Extension Manifest** | Chrome Extension Manifest V3 | — |
| **Containerization** | Docker + docker-compose | — |
| **Node.js** (extension build) | Node.js | 20+ |
| **ML Stack** | scikit-learn, XGBoost, SHAP | flexible on versions |
| **Package Manager (Python)** | pip / uv | — |
| **Package Manager (Extension)** | npm / pnpm | — |

## 2. Project Context
A six-week individual college project aimed at building a functional browser-extension-based phishing detection product while conducting research-quality experiments. The project evaluates a hybrid phishing detector combining transparent expert rules with machine learning, prioritizing accuracy, interpretability, and robust performance. 

Timeline: 6 weeks starting late August 2026.

## 3. Core Research Questions
- Does a calibrated hybrid detector (expert rules + ML) outperform rule-only and ML-only approaches in detection, robustness, and interpretability?
- How much does performance degrade under temporal/cross-dataset shifts, and which adversarial URL perturbations evade detection?
- Can explainable AI (SHAP) expose meaningful signals without misleading certainty?
- Can fast detection remain below latency targets while supporting deeper enrichment?

## 4. Scope
| Category | In Scope | Out of Scope |
|---|---|---|
| **Core Features** | URL phishing detection, Browser extension, REST API, Auth & Scan history | Email/SMS/Vishing detection, Image-based phishing, Multi-browser support |
| **Detection** | URL normalization, Lexical/Host/DNS/TLS/Static HTML features | Deep learning on raw HTML/images, Automated online learning |
| **ML & Logic** | Rule engine, LR/RF/XGBoost, Hybrid fusion, SHAP explanations | GAN-based URL generation (unless P0/P1 done), Enterprise threat intel platform |
| **Evaluation** | Adversarial URL transformations, Temporal & Cross-dataset evaluation, Ablations | Large distributed crawling infrastructure |

## 5. Product Architecture

```text
Browser
  |
  v
Chrome Extension (React/TS, Manifest V3)
  |
  v
FastAPI Backend
  |
  +--> URL normalization
  +--> Lexical features & Lightweight rules
  +--> ML prediction
  |
  +--> SAFE ----------------------> Allow/Indicator
  |
  +--> HIGH RISK -----------------> Warning/Interstitial
  |
  +--> UNCERTAIN -----------------> Deep Analysis (Isolated Worker)
                                      |
                                      +--> DNS / WHOIS / TLS / Static HTML Fetch
                                      v
                                  Feature Enrichment
                                      |
                                      v
                                  Rules + ML + Fusion
                                      |
                                      v
                                  SHAP + Explanations -> Final Risk Result
```

## 6. Detection Layers
- **Stage 1 (Fast Detection)**: Local computation and lightweight API calls (<100 ms). Evaluates URL, lexical features, and lightweight rules. Outputs ML probability, rule score, and preliminary risk.
- **Stage 2 (Deep Detection)**: Triggered for uncertain URLs. Uses DNS, WHOIS, TLS, and static HTML fetched via an isolated worker. Outputs enriched feature vectors, final hybrid risk, and SHAP contributions.

## 7. Risk Model & Explainability
- **UI Risk Categories**: `SAFE`, `SUSPICIOUS`, `HIGH RISK` (Calibrated from validation data).
- **User Explanations**: Risk level, concise reasoning, top signals, and a recommended action.
- **Analyst Explanations**: Final fused score, ML probability, triggered rules, top SHAP features, metadata failures, and scan timestamps.
*Note: Do not conflate model probability, rule score, fused score, and UI risk category.*

## 8. Feature Architecture
High-level feature categories extracted for the pipelines:
- **URL/Lexical**: Lengths, counts, special characters, tokens, and structural statistics.
- **Host/DNS**: DNS records, ASNs, and network metadata.
- **WHOIS/Domain**: Registration dates, privacy indicators, and registrar info.
- **TLS**: Certificate validity, issuers, and hostname matching.
- **Static HTML**: Element counts (forms, scripts), link ratios, and suspicious tokens.

## 9. Data Strategy
- **Sources**: PhishTank/OpenPhish (phishing), Tranco (legitimate).
- **Integrity**: Retain provenance, original URL, and labels. Avoid domain leakage across splits.
- **Splits**: Requires random baseline, temporal, and cross-dataset splits.

## 10. Research Evaluation Design
- **Baselines**: Rule-only, LR, RF, XGBoost, ML-only best model, and hybrid fusion strategies.
- **Metrics**: Precision, Recall, F1, ROC-AUC, PR-AUC, FPR, Latency.
- **Robustness**: Temporal drift, cross-dataset shift, and evasion attacks (homoglyphs, typosquatting).

## 11. Security & Persistence Architecture
- **Isolated Worker (Deep Fetch)**: Must be containerized, outbound-restricted, time/resource-limited, and prevented from executing arbitrary JavaScript. Sanitization of returned content is mandatory.
- **Persistence Model**: PostgreSQL storing Users, Sessions/Tokens, Scans, URL metadata, and Model versions. Only store essential sensitive data.

## 12. 6-Week Delivery Schedule

| Week | Phase | Key Deliverables | Exit Criterion |
|---|---|---|---|
| **Week 1** | Data & Foundations | Repo setup, Dataset acquisition & splits, Lexical extraction, Initial rule engine | Reproducible dataset producing feature tables and rule-only metrics. |
| **Week 2** | ML Models | LR, RF, XGBoost pipelines, Tuning, Baseline metrics | All models train and evaluate through a common pipeline. |
| **Week 3** | Hybrid & Core Research | Fusion strategies, Calibration, Feature ablations, SHAP integration | A defensible final hybrid candidate is selected. |
| **Week 4** | Deep Analysis Backend | DNS/WHOIS/TLS/HTML extraction, Isolated worker, Scan persistence | Backend safely performs an end-to-end deep scan. |
| **Week 5** | Browser Extension | Extension UI, Auth, URL interception, Fast & Deep scan integration, Scan history | User can browse a URL and receive a useful risk decision. |
| **Week 6** | Hardening & Artifacts | Temporal/Cross-dataset/Adversarial experiments, Docker deployment, Demo prep, Paper artifacts | App and paper can be demonstrated from a clean environment. |

## 13. Priority System

| Level | Definition | Scope |
|---|---|---|
| **P0** | Never compromise | Working extension, FastAPI backend, Rules + ML + Hybrid logic, Fast/Deep scans, Explanations, Auth, Scan history, Final evaluation. |
| **P1** | Required for research quality | Temporal/Cross-dataset splits, Adversarial testing, Ablations, Calibration, Statistical testing, Reproducibility. |
| **P2** | Only after P0/P1 | Complex dashboard, Redis/caching, Multi-browser support, GAN attacks, Advanced intel/analyst tooling, Automated retraining. |

## 14. Definition of Done
The project is officially complete when:
- Extension requests a scan and backend returns a risk classification.
- System utilizes both rules and ML (hybrid), with deep analysis for complex cases.
- Explanations are meaningful and based on actual signals.
- Auth and scan history function correctly.
- Model pipeline is reproducibly retrainable.
- Experimental comparisons (Rule-only vs. ML-only vs. Hybrid) and ablations are complete.
- Temporal, cross-dataset, and adversarial robustness tests are available.
- Metrics/figures can be regenerated.
- Entire system runs from documented setup instructions.
- The research paper's claims are fully supported by generated results.

## 15. Engineering Principle
**Do not optimize for feature count.** Optimize for a defensible chain:  
`Reliable data → leakage-safe features → reproducible baselines → calibrated hybrid → robustness tests → explainability → working extension → reproducible evidence`  
A smaller, trustworthy system is strictly better than a larger system with unverified claims.
