# Research Foundation — Hybrid Explainable Phishing Detection

> **Document role**: Scientific backbone. Defines the research hypothesis, threat model, literature context, initial rule/threshold rationale, and paper outline. Referenced by `PLAN_OVERVIEW.md`, `PLANNER.md`, and `AGENT.md`.
>
> **Related documents**:
> - [PLAN_OVERVIEW.md](./PLAN_OVERVIEW.md) — Scope, architecture, schedule, priorities
> - [PLANNER.md](./PLANNER.md) — Implementation phases and checklists
> - [AGENT.md](./AGENT.md) — Agent behavioral rules and coding standards

---

## 1. Problem Statement

Phishing remains one of the most prevalent and effective cyber attacks. Attackers create deceptive URLs and web pages that impersonate legitimate services to steal credentials, financial data, and personal information. Despite decades of research, phishing detection faces persistent challenges:

- **Evasion**: Attackers continuously adapt URLs to bypass static rules and blacklists.
- **Opacity**: ML-based detectors achieve high accuracy but provide no actionable explanation for their decisions, limiting trust and adoption.
- **Brittleness**: Models trained on one dataset or time period degrade when deployed against new attack patterns (temporal drift) or different URL populations (cross-dataset shift).
- **Latency vs. depth tradeoff**: Deep analysis (DNS, WHOIS, HTML) improves detection but introduces latency incompatible with real-time browsing.

This project addresses these challenges through a hybrid architecture that combines transparent expert rules with machine learning, providing both accuracy and interpretability.

---

## 2. Threat Model

### Attacker Profile

- **Goal**: Trick users into visiting a phishing URL and submitting credentials or sensitive data.
- **Capabilities**: Register domains, create convincing page clones, use URL obfuscation (encoding, homoglyphs, subdomain injection, typosquatting), obtain free TLS certificates, use URL shorteners and redirects.
- **Limitations**: Cannot modify the target brand's legitimate domain, cannot prevent DNS/WHOIS lookups, cannot avoid all structural URL anomalies.

### Attack Surface (In Scope)

| Attack Vector | Description |
|---|---|
| Deceptive URLs | URLs crafted to resemble legitimate domains |
| Homoglyph attacks | Visually similar Unicode characters replacing ASCII |
| Typosquatting | Common misspellings of brand domains |
| Subdomain abuse | Brand name as subdomain of attacker domain |
| IP-based hosting | Numeric IP instead of domain name |
| URL encoding tricks | Percent-encoding, Punycode obfuscation |
| Young domains | Freshly registered domains for short-lived campaigns |
| Free TLS abuse | Let's Encrypt certificates on phishing sites |
| Credential harvesting pages | Login forms on cloned pages |

### Attack Surface (Out of Scope)

- Email/SMS delivery mechanisms
- Image-based phishing (screenshots of login pages)
- Browser exploits / drive-by downloads
- Compromised legitimate sites (same domain, different content)
- Vishing / voice phishing

### Defender Assumptions

- The detector sees the full URL at navigation time.
- The detector can perform network lookups (DNS, WHOIS, TLS, HTTP GET) for uncertain cases.
- The detector cannot guarantee real-time access to WHOIS/DNS (rate limits, timeouts).
- The fast-stage detector must function even when network enrichment is unavailable.

---

## 3. Research Hypothesis

### Primary Hypothesis

> A calibrated hybrid detector that combines transparent expert rules with machine-learning predictions provides better phishing detection accuracy, robustness to adversarial evasion, and practical interpretability than either rule-only or ML-only approaches.

### Secondary Hypotheses

| ID | Hypothesis |
|---|---|
| H1 | Lexical URL features alone provide a strong baseline (>85% F1) because phishing URLs exhibit structural anomalies. |
| H2 | Adding DNS/WHOIS/TLS features improves detection for edge cases but has diminishing returns. |
| H3 | The hybrid system reduces false positives at useful recall levels compared to ML-only. |
| H4 | Rule explanations combined with SHAP expose meaningful phishing signals without creating misleading certainty. |
| H5 | Performance degrades measurably under temporal shift (>2% F1 drop) and cross-dataset shift (>5% F1 drop). |
| H6 | Homoglyph and Punycode attacks are the most effective evasion techniques against lexical features. |
| H7 | The hierarchical fusion strategy outperforms simple weighted sum because it leverages rule confidence to short-circuit. |

All hypotheses are testable. If experiments contradict them, the paper must report the actual findings.

---

## 4. Literature Context

### Existing Approaches

| Approach | Strengths | Weaknesses |
|---|---|---|
| **Blacklists** (Google Safe Browsing, PhishTank) | High precision, fast lookup | Zero-day blind spot, maintenance burden |
| **Heuristic rules** (URL pattern matching) | Transparent, fast, no training data needed | Rigid, high false positives, easily evaded |
| **Classical ML** (LR, RF, SVM on URL features) | Good accuracy, handles feature interactions | Opaque decisions, dataset-dependent |
| **Deep learning** (CNN/LSTM on raw URLs) | Learns representations automatically | Requires large data, harder to explain, overkill for structured features |
| **Hybrid approaches** | Combines strengths | Under-studied for explainability and robustness |

### Research Gap

Most hybrid systems combine rules and ML for accuracy but:
- Do not rigorously evaluate **explainability quality** — whether explanations actually help users.
- Do not test **adversarial robustness** systematically.
- Do not evaluate **temporal generalization** — whether the system works on future phishing campaigns.
- Do not compare **multiple fusion strategies** on equal footing.

This project fills these gaps by implementing and comparing four fusion strategies with full explainability, adversarial, and temporal evaluation.

### Key References (Indicative)

- Mohammad, Thabtah & McCluskey (2014) — URL-based phishing features taxonomy
- Sahingoz et al. (2019) — ML-based phishing detection comparison
- Rao & Pais (2020) — Feature analysis for phishing URL detection
- Lundberg & Lee (2017) — SHAP (SHapley Additive exPlanations)
- Le et al. (2018) — URLNet: deep learning on URLs
- Marchal et al. (2017) — PhishStorm: phishing detection via URL analysis

*(Replace with actually cited papers during the literature review phase.)*

---

## 5. Initial Rule Weight Rationale

> **⚠️ IMPORTANT**: These weights are starting hypotheses derived from literature patterns and domain intuition. They are NOT validated values. Validation experiments (Phase 3/5 in the planner) must determine final weights and thresholds.

### Proposed Initial Weights

| Rule ID | Signal | Initial Weight | Rationale |
|---|---|---|---|
| `R_IP_HOST` | IP as hostname | 3.0 | Strong phishing indicator; legitimate sites rarely use raw IPs |
| `R_AT_SYMBOL` | `@` in URL | 2.5 | Used to obfuscate the actual destination; rare in legitimate URLs |
| `R_NEW_DOMAIN` | Domain < 30 days old | 3.0 | Most phishing domains are short-lived; new registration is suspicious |
| `R_CERT_MISMATCH` | Cert/domain mismatch | 3.0 | Direct evidence of impersonation attempt |
| `R_BRAND_TOKEN` | Suspicious brand token in path/subdomain | 2.0 | Attackers embed brand names to appear legitimate |
| `R_LONG_URL` | URL > 75 characters | 1.0 | Mildly correlated; many legitimate URLs are long too |
| `R_DEEP_SUBDOMAINS` | > 3 subdomain levels | 1.5 | Subdomain depth is unusual for legitimate sites |
| `R_SUSPICIOUS_TLD` | Risky TLD (.tk, .ml, .ga, .cf, .gq) | 1.5 | Free TLDs are disproportionately used for phishing |
| `R_NO_HTTPS` | HTTP (no TLS) | 1.0 | Decreasingly useful as free certs proliferate, but still a signal |
| `R_SUSPICIOUS_FORM` | Login form on suspicious page | 2.5 | Credential harvesting is the primary phishing goal |
| `R_EXCESSIVE_EXTERNAL` | > 80% external links | 1.5 | Cloned pages often link externally to the real site |
| `R_AUTO_REDIRECT` | Meta/JS redirect | 2.0 | Used to chain through multiple phishing domains |
| `R_PUNYCODE` | Punycode/IDN hostname | 2.0 | Used for homoglyph attacks |

### Initial Thresholds (Subject to Calibration)

| Parameter | Initial Value | Notes |
|---|---|---|
| Rule score → SAFE | < 2.0 | Low rule activation |
| Rule score → SUSPICIOUS | 2.0 – 5.0 | Moderate signals present |
| Rule score → HIGH RISK | > 5.0 | Strong phishing indicators |
| ML probability → phishing | > 0.5 | Standard classification threshold |
| Confidence → trigger deep analysis | 0.3 – 0.7 | Uncertain zone |

These thresholds must be calibrated using validation set performance (precision-recall tradeoff).

---

## 6. Feature Engineering Rationale

### Why These Feature Groups?

| Group | Rationale | Available At |
|---|---|---|
| **Lexical/URL** | URL structure reveals obfuscation patterns; available instantly with zero network cost | Fast stage |
| **Host/DNS** | Legitimate domains have established DNS infrastructure; missing records are suspicious | Deep stage |
| **WHOIS/Domain** | Phishing domains are typically young and short-lived; domain age is a strong signal | Deep stage |
| **TLS** | Free certificate abuse means TLS presence ≠ legitimacy; but cert metadata reveals patterns | Deep stage |
| **Static HTML** | Credential harvesting requires forms; cloned pages have structural anomalies | Deep stage |

### Known Feature Limitations

- **WHOIS**: Rate-limited, increasingly privacy-protected (GDPR), data quality varies by registrar.
- **DNS**: Can be manipulated; fast-flux hosting complicates analysis.
- **TLS**: Let's Encrypt makes free certs trivial; cert presence is nearly meaningless as a safety signal.
- **HTML**: Requires fetching potentially hostile content; must be sandboxed.
- **Lexical**: Susceptible to adversarial URL mutations (the project explicitly tests this).

---

## 7. Evaluation Philosophy

### Why Multiple Evaluation Axes?

| Axis | What It Tests | Why It Matters |
|---|---|---|
| **Random split** | Standard classification performance | Baseline sanity check |
| **Temporal split** | Generalization to future attack patterns | Real-world deployment requires temporal robustness |
| **Cross-dataset split** | Generalization across URL populations | Overfitting to one source's URL distribution is a common failure |
| **Adversarial** | Robustness to deliberate evasion | Attackers actively try to bypass detectors |
| **Ablation** | Contribution of each feature group | Identifies which features justify their cost (latency, complexity) |
| **Calibration** | Whether predicted probabilities are meaningful | A "90% phishing" prediction should be correct ~90% of the time |

### Statistical Rigor

- Use **McNemar's test** for pairwise model comparison (same test set, different models).
- Report **95% confidence intervals** for key metrics where feasible (bootstrap).
- Report **calibration curves** alongside discrimination metrics.
- Never report only accuracy — always include precision, recall, F1, and AUC.

---

## 8. Expected Contributions

This project aims to contribute:

1. **A working hybrid phishing detection system** — browser extension + backend, not just a notebook.
2. **Systematic fusion strategy comparison** — four strategies evaluated on equal footing.
3. **Explainability evaluation** — combining SHAP with rule traces for user and analyst audiences.
4. **Adversarial robustness assessment** — five evasion techniques tested against all model configurations.
5. **Temporal and cross-dataset evaluation** — honest assessment of real-world generalization.
6. **Reproducible experimental pipeline** — all results regenerable from documented commands.

---

## 9. Paper Outline

| # | Section | Content | Evidence Source |
|---|---|---|---|
| 1 | **Abstract** | Problem, approach, key results | Final experiment metrics |
| 2 | **Introduction** | Phishing problem, motivation for hybrid approach, contributions | Literature, threat model |
| 3 | **Related Work** | Existing detection approaches, research gaps | Literature review |
| 4 | **Threat Model** | Attacker capabilities, attack surface, defender assumptions | Section 2 of this document |
| 5 | **System Architecture** | Two-stage design, browser extension, backend | Architecture diagram |
| 6 | **Dataset & Methodology** | Sources, preprocessing, splits, leakage prevention | Dataset manifest, split scripts |
| 7 | **Feature Engineering** | Feature groups, extraction process, rationale | Feature schema, extraction code |
| 8 | **Rule Engine** | Rule design, weights, score normalization | Rule catalog, rule-only evaluation |
| 9 | **ML Models** | LR, RF, XGBoost training and comparison | Model comparison report |
| 10 | **Hybrid Fusion** | Four strategies, selection protocol, calibration | Fusion comparison table |
| 11 | **Explainability** | SHAP integration, user vs analyst explanations | Explanation examples |
| 12 | **Experiments & Results** | Full evaluation matrix results | Experiment outputs |
| 13 | **Temporal & Cross-Dataset** | Generalization analysis | S2/S3 results |
| 14 | **Adversarial Evaluation** | Evasion attack results | A1–A5 degradation tables |
| 15 | **Ablation Study** | Feature group contribution analysis | F1–F4 results |
| 16 | **Discussion** | Findings interpretation, practical implications | All results |
| 17 | **Limitations** | Known weaknesses, scope boundaries | Honest self-assessment |
| 18 | **Conclusion & Future Work** | Summary, next steps | All sections |
| — | **References** | All cited works | Literature review |
| — | **Appendices** | Full feature list, rule catalog, additional figures | Implementation artifacts |

### Paper Rules

- No numerical claim without a stored experiment result backing it.
- No performance comparison without identical evaluation protocol.
- Limitations must be genuine, not pro-forma.
- All figures and tables generated from code, not manually assembled.

---

## 10. Ethical Considerations

- **No operational phishing**: Adversarial tests generate mutated URLs for evaluation only. No phishing sites are deployed. No malicious emails are sent.
- **No user data collection**: The system is a detection tool, not a surveillance tool. Scan history is user-owned.
- **Responsible disclosure**: If the research identifies specific weaknesses in existing defenses, these should be reported responsibly, not exploited.
- **Transparency**: The system explicitly communicates uncertainty. It never claims definitive proof of malice.
