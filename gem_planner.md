# Implementation Planner — Six-Week Solo Build

> **Document role**: The HOW — phased implementation tasks, checklists, API contracts, feature lists, experiment matrix, and acceptance gates.
>
> **Related documents**:
> - [PLAN_OVERVIEW.md](./PLAN_OVERVIEW.md) — Scope, architecture, schedule, priorities (source of truth)
> - [AGENT.md](./AGENT.md) — Agent behavioral rules and coding standards
> - [RESEARCH.md](./RESEARCH.md) — Research foundation, threat model, hypotheses, paper outline

---

## Operating Assumptions

- Solo project assisted by a single AI coding agent.
- Human owner decides: architecture, security, scientific claims, scope, datasets, deployment, paper conclusions.
- Schedule and priorities defined in `PLAN_OVERVIEW.md`. This document details *how* to implement each phase.

---

## Directory Structure

```text
phishguard/
├── backend/
│   ├── app/
│   │   ├── api/              # route handlers
│   │   ├── auth/             # JWT, password hashing
│   │   ├── services/         # detection orchestration
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── schemas/          # Pydantic request/response schemas
│   │   ├── core/             # config, deps, security utils
│   │   └── main.py
│   └── tests/
├── ml/
│   ├── data/                 # dataset scripts, manifests, splits
│   ├── features/             # feature extractors, schema
│   ├── models/               # trained artifacts, training scripts
│   ├── rules/                # rule engine, rule catalog
│   ├── fusion/               # hybrid strategies
│   ├── explainability/       # SHAP, explanation builders
│   └── training/             # pipeline, hyperparameter configs
├── worker/
│   ├── fetcher/              # HTTP fetch with security controls
│   ├── extractors/           # DNS, WHOIS, TLS, HTML extractors
│   └── tests/
├── extension/
│   ├── src/                  # React/TypeScript source
│   ├── public/               # static assets, icons
│   └── manifest.json         # Manifest V3
├── experiments/
│   ├── configs/              # experiment YAML configs
│   ├── runners/              # experiment execution scripts
│   ├── results/              # stored raw results (JSON/CSV)
│   └── reports/              # generated charts, tables
├── scripts/                  # utility scripts
├── tests/                    # cross-cutting integration tests
├── docs/                     # setup, API docs, research notes
├── docker/                   # Dockerfiles, docker-compose
├── pyproject.toml
├── .env.example
└── README.md
```

Simplify if evidence suggests a flatter layout, but preserve separation between product code (`backend/`, `worker/`, `extension/`) and research code (`ml/`, `experiments/`).

---

## Phase 0 — Repository & Contracts

**Goal**: Project skeleton and interface contracts before feature work.

### Tasks

- [ ] Initialize repository with `pyproject.toml`, `.gitignore`, `README.md`
- [ ] Create package stubs: `backend/`, `ml/`, `worker/`, `extension/`, `experiments/`
- [ ] Create `.env.example` with all required env vars
- [ ] Create `docker-compose.yml` (FastAPI + PostgreSQL + worker)
- [ ] Define API contracts (see API section below)
- [ ] Define feature schema version format (e.g., `v1.0`)
- [ ] Define model artifact format: `{model_name}_v{version}_{timestamp}/`
- [ ] Define risk result schema (see below)
- [ ] Set up linting/formatting (ruff, mypy, eslint for extension)
- [ ] Create test infrastructure (pytest, vitest for extension)

### Risk Result Schema

```python
class ScanResult:
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
    analyst_explanation: AnalystExplanation | None
    model_version: str
    feature_schema_version: str
    rule_config_version: str
    scan_timestamp: datetime
    deep_analysis_available: bool
    metadata_failures: list[str]
```

### Deliverables

- Working repo skeleton
- Docker compose starts backend + PostgreSQL
- API contract documented
- Schema definitions committed

---

## Phase 1 — Dataset Pipeline

**Goal**: Trustworthy, deduplicated, leakage-checked dataset with documented provenance.

### Tasks

- [ ] Identify accessible phishing sources (PhishTank, OpenPhish, or alternatives)
- [ ] Identify legitimate source (Tranco top-list)
- [ ] Record source, license, collection date, and snapshot version for each
- [ ] Download/snapshot data
- [ ] Normalize URLs (lowercase scheme/host, strip fragments, decode percent-encoding)
- [ ] Extract registrable domain (using `tldextract` or similar)
- [ ] Remove exact URL duplicates
- [ ] Detect and log domain-level overlap between phishing/legitimate sets
- [ ] Detect and resolve label conflicts (same URL with different labels)
- [ ] Build source-specific manifests (JSON)
- [ ] Build splits:
  - [ ] Random train/val/test (domain-aware: no domain in both train and test)
  - [ ] Temporal split (train on older data, test on newer)
  - [ ] Cross-dataset split (train on source A, test on source B)

### Required Data Checks

- [ ] No URL duplicates across train/test
- [ ] No domain leakage across splits
- [ ] Labels are source-derived and documented
- [ ] Collection dates retained
- [ ] Dataset version recorded
- [ ] Missing values explicit (not silently dropped)

### Deliverables

- Dataset manifest (JSON: source, size, date, label distribution per split)
- Cleaning script (`scripts/clean_dataset.py`)
- Split script (`scripts/build_splits.py`)
- Dataset statistics report
- Leakage verification report

---

## Phase 2 — Feature Engine

**Goal**: Deterministic, testable feature extraction with versioned schema.

### Lexical Features

- [ ] `url_length` — total URL character count
- [ ] `hostname_length` — hostname character count
- [ ] `path_length` — path character count
- [ ] `query_length` — query string character count
- [ ] `dot_count` — dots in full URL
- [ ] `subdomain_count` — number of subdomain levels
- [ ] `hyphen_count` — hyphens in hostname
- [ ] `digit_count` — digits in URL
- [ ] `special_char_count` — `@`, `~`, `!`, `$`, `&`, etc.
- [ ] `has_at_symbol` — boolean
- [ ] `has_ip_hostname` — hostname is IPv4/IPv6
- [ ] `suspicious_token_count` — tokens like `login`, `secure`, `account`, `verify`, `update`, `bank`
- [ ] `url_entropy` — Shannon entropy of URL string
- [ ] `has_encoding` — `%xx` presence
- [ ] `has_punycode` — `xn--` prefix
- [ ] `digit_letter_ratio` — digits / (letters + 1)
- [ ] `path_token_count` — path segments
- [ ] `query_param_count` — query parameters

### Host/DNS Features (Stage 2)

- [ ] `has_dns_a` — A record exists
- [ ] `has_dns_aaaa` — AAAA record exists
- [ ] `ns_count` — nameserver count
- [ ] `has_mx` — MX record exists
- [ ] `dns_record_count` — total DNS records
- [ ] `asn` — Autonomous System Number (where practical)

### WHOIS Features (Stage 2)

- [ ] `domain_age_days` — days since creation
- [ ] `domain_update_age_days` — days since last update
- [ ] `domain_expiry_days` — days until expiration
- [ ] `registrar` — registrar name (categorical/hashed)
- [ ] `whois_privacy` — privacy protection enabled

### TLS Features (Stage 2)

- [ ] `cert_valid` — certificate currently valid
- [ ] `cert_issuer` — issuer organization (categorical/hashed)
- [ ] `cert_age_days` — days since certificate issuance
- [ ] `san_count` — Subject Alternative Name count
- [ ] `cert_hostname_match` — certificate matches hostname

### HTML Features (Stage 2)

- [ ] `form_count` — `<form>` elements
- [ ] `password_input_count` — `<input type="password">`
- [ ] `input_count` — total `<input>` elements
- [ ] `script_count` — `<script>` elements
- [ ] `iframe_count` — `<iframe>` elements
- [ ] `external_link_ratio` — external links / total links
- [ ] `has_redirect` — meta refresh or JS redirect patterns
- [ ] `title_length` — `<title>` character count
- [ ] `suspicious_text_count` — tokens like "verify", "suspended", "confirm"

### Engineering Requirements

- [ ] Every extractor is deterministic for deterministic input
- [ ] External lookup failures return explicit missing values (not defaults)
- [ ] Feature schema has a version string
- [ ] Unit tests cover: normal URLs, malformed URLs, IDN/Punycode, IP-host, edge cases
- [ ] Extraction latency is measurable per feature group

### Deliverables

- Feature extraction module (`ml/features/`)
- Feature schema definition (v1.0)
- Feature extraction tests
- Feature extraction latency benchmarks

---

## Phase 3 — Rule Engine

**Goal**: Independently testable, transparent rule-based detector.

### Candidate Rules

| Rule ID | Signal | Severity | Feature Dependencies |
|---|---|---|---|
| `R_AT_SYMBOL` | `@` in URL | Medium | `has_at_symbol` |
| `R_LONG_URL` | URL > threshold length | Low | `url_length` |
| `R_IP_HOST` | IP address as hostname | High | `has_ip_hostname` |
| `R_DEEP_SUBDOMAINS` | > N subdomain levels | Medium | `subdomain_count` |
| `R_BRAND_TOKEN` | Suspicious brand-like token | High | `suspicious_token_count` |
| `R_SUSPICIOUS_TLD` | Known-risky TLD | Medium | domain TLD |
| `R_NO_HTTPS` | HTTP instead of HTTPS | Medium | URL scheme |
| `R_CERT_MISMATCH` | Certificate/domain mismatch | High | `cert_hostname_match` |
| `R_NEW_DOMAIN` | Domain < N days old | High | `domain_age_days` |
| `R_SUSPICIOUS_FORM` | Login form on suspicious page | High | `form_count`, `password_input_count` |
| `R_EXCESSIVE_EXTERNAL` | High external link ratio | Medium | `external_link_ratio` |
| `R_AUTO_REDIRECT` | Meta/JS redirect detected | Medium | `has_redirect` |
| `R_PUNYCODE` | Punycode/IDN in hostname | Medium | `has_punycode` |

### Tasks

- [ ] Implement rule registry (load from config)
- [ ] Implement rule IDs, descriptions, weights, severity
- [ ] Implement per-rule trigger logic
- [ ] Implement raw rule score computation (sum of triggered weights)
- [ ] Implement rule score normalization (0–1)
- [ ] Implement threshold configuration
- [ ] Record every triggered rule with structured output
- [ ] Produce rule explanation objects
- [ ] Unit test every rule individually
- [ ] Evaluate rule-only baseline on validation split

### Deliverables

- Rule engine (`ml/rules/`)
- Rule catalog config (YAML/JSON)
- Rule unit tests
- Rule-only evaluation metrics

---

## Phase 4 — ML Training

**Goal**: Controlled, reproducible baselines for LR, RF, XGBoost.

### Tasks

- [ ] Implement preprocessing pipeline (imputation, encoding, scaling — fit on train only)
- [ ] Train Logistic Regression
- [ ] Train Random Forest
- [ ] Train XGBoost
- [ ] Hyperparameter tuning (grid/random search on validation set)
- [ ] Save model artifacts: `model + preprocessor + schema + hyperparams + seed + metrics`
- [ ] Record training time and inference time per model
- [ ] Evaluate all models on validation set
- [ ] Generate model comparison report

### Training Rules

- Same split definitions for all models
- Same label definitions
- No test-set tuning
- Preprocessing fit on training data only
- Random seeds recorded and reproducible

### Metrics (compute for each model)

| Metric | Required |
|---|---|
| Precision | ✅ |
| Recall | ✅ |
| F1 | ✅ |
| Accuracy | ✅ (secondary) |
| ROC-AUC | ✅ |
| PR-AUC | ✅ |
| FPR | ✅ |
| Confusion Matrix | ✅ |
| Calibration Curve | If feasible |

### Deliverables

- Model artifacts (`ml/models/`)
- Training configs
- Model comparison report (table + plots)
- Reproducible training command (`python -m ml.training.train --config ...`)

---

## Phase 5 — Fusion

**Goal**: Scientifically compare hybrid strategies, select winner on validation.

### Implement

- [ ] **Weighted sum**: `fused = α × ml_prob + (1 − α) × rule_score`
- [ ] **Meta-classifier**: train a small model on `(ml_prob, rule_score, rule_count)` → label
- [ ] **OR logic**: flag as phishing if *either* ML or rules exceed threshold
- [ ] **Hierarchical logic**: rules first → if uncertain → ML → if still uncertain → deep

### Protocol

```text
Train models + rules on training set
    ↓
Evaluate fusion strategies on validation set
    ↓
Select best fusion strategy + parameters
    ↓
Freeze configuration
    ↓
Evaluate once on final test set (no further changes)
```

### Deliverables

- Fusion implementations (`ml/fusion/`)
- Validation comparison table
- Selected fusion config (frozen)
- Final hybrid evaluation on test set

---

## Phase 6 — Explainability

**Goal**: Evidence-based explanations for both users and analysts.

### Tasks

- [ ] Implement `UserExplanation` schema: risk level, top 3 reasons, recommendation
- [ ] Implement `AnalystExplanation` schema: all scores, all triggered rules, SHAP values, feature values, metadata
- [ ] Integrate SHAP (TreeSHAP for XGBoost/RF)
- [ ] Extract top positive and negative SHAP contributors
- [ ] Generate global feature importance plot
- [ ] Map rule triggers to human-readable text
- [ ] Combine rule explanations + SHAP into unified output

### Deliverables

- Explanation builders (`ml/explainability/`)
- Example explanation outputs (user + analyst)
- Global feature importance chart

---

## Phase 7 — Deep-Analysis Worker

**Goal**: Safely enrich uncertain URLs with DNS, WHOIS, TLS, and HTML data.

### Security Controls

- [ ] Isolated execution (subprocess or Docker container)
- [ ] Request timeout (configurable, default 10s)
- [ ] Response-size limit (configurable, default 5 MB)
- [ ] Redirect limit (max 5)
- [ ] Content-type whitelist (`text/html` only for HTML fetch)
- [ ] No arbitrary JavaScript execution
- [ ] No host filesystem access
- [ ] No environment-secret access
- [ ] No cloud metadata endpoint access (block 169.254.169.254)
- [ ] SSRF prevention (reject private/loopback IPs)
- [ ] Worker cleanup after each analysis
- [ ] Structured logging of all fetch operations

### Pipeline

```text
URL → Fetch policy check → DNS lookup → WHOIS lookup → TLS check
    → Restricted HTTP GET → Static HTML parse → Feature extraction → Cleanup
```

### Deliverables

- Worker module (`worker/`)
- Security control tests
- Integration test: full deep-scan pipeline
- Failure-mode documentation

---

## Phase 8 — FastAPI Backend

**Goal**: Production-quality API serving the extension.

### API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | No | Register new user |
| `POST` | `/auth/login` | No | Login, returns JWT |
| `POST` | `/scan` | Yes | Fast scan a URL |
| `POST` | `/scan/deep` | Yes | Trigger deep analysis |
| `GET` | `/scans` | Yes | List user's scan history |
| `GET` | `/scans/{id}` | Yes | Get single scan detail |
| `GET` | `/health` | No | Health check |
| `GET` | `/model` | No | Current model/rule version info |

### Request/Response Sketches

```python
# POST /scan
class ScanRequest(BaseModel):
    url: str  # max 2048 chars

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

# POST /auth/login
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

### Tasks

- [ ] Set up FastAPI app with CORS, structured error handling
- [ ] PostgreSQL connection (SQLAlchemy async + Alembic migrations)
- [ ] User model + JWT auth (python-jose, bcrypt)
- [ ] Scan model + CRUD
- [ ] `/scan` endpoint: normalize URL → extract features → run rules + ML → fuse → respond
- [ ] `/scan/deep` endpoint: trigger worker → enrich features → re-score → respond
- [ ] Input validation (URL length, format, SSRF checks)
- [ ] Rate limiting
- [ ] Request timeouts
- [ ] Model version in all scan responses
- [ ] API documentation (auto-generated OpenAPI)
- [ ] Unit + integration tests

### Deliverables

- FastAPI application (`backend/`)
- Database migrations
- API tests
- OpenAPI spec

---

## Phase 9 — Browser Extension

**Goal**: User-facing Chrome extension (Manifest V3, React/TypeScript).

### Core Flow

```text
Navigation event → Extract URL → POST /scan → Display risk
    → If uncertain: POST /scan/deep → Update display
```

### UI States

- [ ] Loading (scanning in progress)
- [ ] Safe (green indicator)
- [ ] Suspicious (yellow indicator + explanation)
- [ ] High Risk (red warning + interstitial)
- [ ] Deep analysis in progress
- [ ] Analysis unavailable (backend error/timeout)
- [ ] Authentication required (login prompt)
- [ ] Backend unreachable (offline fallback)

### Features

- [ ] Automatic URL check on navigation
- [ ] Warning interstitial for high-risk URLs
- [ ] Explanation popup (user mode)
- [ ] Manual rescan button
- [ ] Scan history view
- [ ] Settings (API URL, auto-scan toggle)
- [ ] Login/logout
- [ ] Badge icon color change by risk level

### Deliverables

- Chrome extension (`extension/`)
- Manifest V3 configuration
- Extension ↔ API integration tests
- Screenshots / UI documentation

---

## Phase 10 — Research Experiments

**Goal**: Generate all evidence required for the paper.

### Experiment Matrix

**Models** × **Splits** × **Conditions**:

| Code | Model/System |
|---|---|
| R | Rule-only |
| LR | Logistic Regression |
| RF | Random Forest |
| XGB | XGBoost |
| H-ws | Hybrid (weighted sum) |
| H-mc | Hybrid (meta-classifier) |
| H-or | Hybrid (OR logic) |
| H-hi | Hybrid (hierarchical) |
| H* | Selected best hybrid |

| Code | Split |
|---|---|
| S1 | Random (domain-aware) |
| S2 | Temporal |
| S3 | Cross-dataset |

### Adversarial Transformations

| Code | Transformation | Example |
|---|---|---|
| A1 | Homoglyph substitution | `paypal.com` → `pаypal.com` (Cyrillic а) |
| A2 | Typosquatting | `paypal.com` → `paypall.com` |
| A3 | Subdomain injection | `paypal.com.evil.com` |
| A4 | Encoding/Punycode | `%70aypal.com` |
| A5 | Case manipulation | `PAYPAL.COM` |

### Feature Ablations

| Code | Feature Set |
|---|---|
| F1 | Lexical only |
| F2 | Lexical + Host/DNS |
| F3 | Lexical + Host/DNS + TLS |
| F4 | Lexical + Host/DNS + TLS + HTML (full) |

### Required Outputs

- [ ] CSV/JSON raw experiment results
- [ ] Metrics comparison table (all models × all splits)
- [ ] Confusion matrices
- [ ] ROC curves (per model, overlaid)
- [ ] PR curves (per model, overlaid)
- [ ] Feature importance plot (global)
- [ ] SHAP waterfall/beeswarm examples
- [ ] Adversarial degradation table (F1 drop per attack)
- [ ] Ablation table (F1 by feature group)
- [ ] Latency table (p50, p95, p99 per stage)
- [ ] Statistical tests (McNemar's for model pairs, confidence intervals)

All charts generated programmatically from stored results — never manually edited.

---

## Phase 11 — Paper Integration

The implementation must generate evidence for each paper section:

| Paper Section | Evidence Source |
|---|---|
| Dataset methodology | Dataset manifest, split scripts, leakage report |
| Feature engineering | Feature schema, extraction benchmarks |
| Rule component | Rule catalog, rule-only evaluation |
| ML component | Model comparison report, training configs |
| Hybrid strategy | Fusion comparison table, selected config |
| Explainability | SHAP examples, explanation outputs |
| Experiments | All Phase 10 outputs |
| Temporal/cross-dataset | S2/S3 results |
| Adversarial | A1–A5 degradation tables |
| Ablations | F1–F4 results |
| Limitations | Known failures, coverage gaps |

**Rule**: Do not write numerical claims into the paper until the corresponding experiment has been executed and stored.

---

## Acceptance Gates

| Gate | Timing | Criterion |
|---|---|---|
| G1 | End of Week 1 | Dataset + lexical features + rule-only metrics work |
| G2 | End of Week 2 | LR/RF/XGBoost train and evaluate reproducibly |
| G3 | End of Week 3 | Hybrid system experimentally selected on validation |
| G4 | End of Week 4 | Deep backend performs end-to-end scan safely |
| G5 | End of Week 5 | Browser extension works end-to-end |
| G6 | End of Week 6 | Research experiments and paper evidence complete |

**If a gate is missed**: cut P2 scope immediately (see `plan_overview.md` §13).

---

## Risk Register

| Risk | Impact | Likelihood | Response |
|---|---|---|---|
| Dataset source inaccessible | High | Medium | Use currently available public alternatives; document |
| WHOIS rate limits / blocks | Medium | High | Cache aggressively, batch requests, tolerate missing |
| HTML fetching unsafe | High | Medium | Isolated worker; reduce/disable HTML features if needed |
| Model performance weak | High | Low | Analyze data/features before adding model complexity |
| Dataset leakage discovered | Critical | Medium | Domain-aware splits, separate provenance, automated checks |
| Extension bugs block demo | High | Medium | Keep backend API independently testable via curl/Postman |
| Deep worker delays | Medium | Medium | Fast-stage fallback always operational |
| Scope creep | Critical | High | Strict P0/P1/P2 prioritization per `plan_overview.md` |
| Paper claims unsupported | Critical | Medium | Every quantitative claim references stored experiment output |
