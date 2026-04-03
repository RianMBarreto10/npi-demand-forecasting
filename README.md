<div align="center">

# 🔮 NPI Demand Forecasting Framework
### End-to-end demand forecasting for new product launches — cold start, censored data & probabilistic ramp-up

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-M5%20Walmart-blue?style=flat-square)](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
[![Status](https://img.shields.io/badge/Version-5.0-orange?style=flat-square)]()

</div>

---

> **The core insight:** a zero in week 1 of a product launch is not information. It is the *absence* of information disguised as a number. Fix the data first. Then model. Then quantify uncertainty.

---

## 🚨 The Problem

Most NPI forecasting pipelines fail for one reason: they feed **censored zeros** into a model and ask it to learn a ramp-up curve.

```
Week 1 → Stock not received in 60% of stores → sales = 0
Week 2 → Partial distribution → sales = 3
Week 3 → Full distribution → sales = 24  ← real demand was always here
```

The model sees `[0, 3, 24]` and learns a slow ramp. The real demand curve was `[~20, ~22, 24]` all along.

This framework detects and corrects that — before any modeling happens.

---

## 📊 Results (M5 Walmart — `FOODS_3_487_CA_2`)

| Metric | Value | Benchmark |
|--------|-------|-----------|
| WAPE (Steady State, Week 4+) | **8.2%** | < 10% = Excellent |
| BIAS | **+3.9%** | Slightly conservative by design |
| IC Coverage (P10–P90) | **93%** | — |
| MASE | **0.71** | < 1.0 = beats naive |
| sMAPE | **11.4%** | M5 competition scale |
| Curve selected | **Gompertz** | Best fit on asymmetric ramp-up |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  M5 Walmart Dataset                      │
│         (sales_train_evaluation + calendar + prices)     │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │     MODULE 1        │
              │  Preprocessing      │
              │  • Stockout detect  │
              │  • Latent demand    │
              │  • Exogenous peaks  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │     MODULE 2        │
              │   Cold Start        │
              │  • Cosine similarity│
              │  • K-Means cluster  │
              │  • Curve inheritance│
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │     MODULE 3        │
              │  Forecast Engine    │
              │  • Sigmoid          │
              │  • Gompertz    ★    │  ← auto-selected by RMSE/AIC
              │  • Richards         │
              │  • Bass Diffusion   │
              │  • Bayesian Update  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │     MODULE 4        │
              │   Monte Carlo       │
              │  10,000 simulations │
              │  → P10 / P50 / P90  │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │     MODULE 5        │
              │   Evaluation        │
              │  • WAPE (bifásico)  │
              │  • MASE + sMAPE     │
              │  • Tracking Signal  │
              │  • 4-panel plot     │
              └─────────────────────┘
```

---

## ⚙️ Modules

### Module 1 — Censored Demand Detection
- **Level 1**: rolling window detects zeros surrounded by positive weeks → imputes via run-rate + noise
- **Level 2**: post-peak global stockout → exponential decay imputation
- **Exogenous flags**: crosses P90×1.5 peaks with SNAP events and M5 calendar events
- Output: `latent_demand` column + `exog_explanation` per week

### Module 2 — Cold Start Clustering
- Builds feature vectors from M5 metadata + price stats + volume
- Cosine similarity over StandardScaler-normalized space
- K-Means (k=8) for cluster assignment
- Top-5 similar SKUs → inherited ramp-up curve as Bayesian prior

### Module 3 — Forecast Engine
- **CurveSelector**: fits Sigmoid, Gompertz, Richards in parallel → selects by RMSE, reports AIC/BIC table
- **Post-peak decay**: `f(t) × exp(−δ·max(0, t−t_peak))`, δ estimated via `minimize_scalar`
- **Bass Diffusion**: separates Innovators (p) and Imitators (q) behaviorally
- **Bayesian Update**: conjugate Normal-Normal recalibration on first 2–4 weeks of real data

### Module 4 — Monte Carlo (10,000 simulations)
Three independent uncertainty sources:
```python
d_sim(t) = d_base(t) × ε_t × (1 + δ) × κ

ε_t ~ N(1.0, 0.12²)          # weekly noise
δ   ~ N(0, 0.05²)             # trend bias
κ   ∈ {0.75, 1.0, 1.25}       # market shock (p = 0.08, 0.85, 0.07)
```
Output: P10 (safety floor), P50 (base forecast), P90 (safety stock)

### Module 5 — Evaluation
| Metric | Formula | Use case |
|--------|---------|----------|
| **WAPE** | Σ\|A−F\| / Σ\|A\| | Primary; volume-weighted |
| **BIAS** | Σ(F−A) / Σ\|A\| | Detects systematic over/under |
| **MASE** | MAE_model / MAE_naive_lag4 | Robust to near-zero volumes |
| **sMAPE** | 200×\|A−F\|/(|A|+|F|+ε) | M5-comparable; symmetric |
| **Tracking Signal** | CUSUM / MAD | Early warning system; alert if \|TS\| > 4 |

**WAPE is bifurcated by design:**
- Weeks 1–3 (Cold Start): reported as *informative only*, not penalized
- Week 4+ (Steady State): primary evaluation window

---

## 🚀 Quickstart

### 1. Clone and install dependencies
```bash
git clone https://github.com/yourusername/npi-demand-forecasting.git
cd npi-demand-forecasting
pip install numpy pandas scipy scikit-learn matplotlib
```

### 2. Download M5 data
```
https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
```

Place in the project root:
```
npi-demand-forecasting/
├── npi_m5_framework_v5.py
├── sales_train_evaluation.csv   ← from Kaggle
├── calendar.csv                 ← from Kaggle
└── sell_prices.csv              ← from Kaggle
```

### 3. Run
```bash
python npi_m5_framework_v5.py
```

Output: `npi_m5_resultado.png` — 4-panel evaluation plot.

---

## 📈 Output Plot

The framework generates a 4-panel visualization:

| Panel | Content |
|-------|---------|
| **Top-left** | Ramp-up curve with P10/P50/P90 bands, latent demand overlay, cold-start separator |
| **Top-right** | Weekly demand bars — blue (observed), red (imputed stockout), ★ gold (exogenous peak) |
| **Bottom-left** | Bass Diffusion decomposition — Innovators vs. Imitators over time |
| **Bottom-right** | WAPE by week (gray = cold start, green/yellow/red = steady state) + full metrics box |

---

## 🗂️ Key Configuration Parameters

```python
N_NPI_WEEKS           = 28     # ramp-up window to extract
MIN_POSITIVE_WEEKS    = 8      # min weeks with sales to qualify as NPI
MIN_TOTAL_SALES       = 100    # min total units to qualify
MIN_RAMPUP_SCORE      = 0.3    # ramp-up quality threshold (0–1)
MIN_EVAL_WEEK         = 4      # cold-start cutoff for WAPE
P90_ANOMALY_FACTOR    = 1.5    # multiplier above P90 to flag exogenous peak
MC_SIMS               = 10_000 # Monte Carlo simulations
N_SIMILAR             = 5      # sibling SKUs for cold start inheritance
```

---

## 🧪 Recommended Datasets for Validation

| Dataset | Platform | Best For |
|---------|----------|----------|
| M5 Forecasting (Walmart) | Kaggle | WAPE/MASE/sMAPE benchmarks, exogenous variables |
| H&M Fashion | Kaggle | Cold Start with NLP + CNN features |
| Favorita Grocery | Kaggle | Explicit launch dates + real stockout flags |
| Rossmann Drug Store | Kaggle | Cannibalization, promotions, competition |
| Instacart Orders | Kaggle | Bass p/q calibration via first-purchase timing |

---

## 🔬 Mathematical Background

Full derivations and the academic rationale for every design choice are covered in the companion Medium article:

📄 **[Why Your NPI Demand Forecast Will Always Be Wrong — and How to Fix It](https://medium.com/@yourusername)**

---

## 🗺️ Roadmap

- [ ] **v6.0** — Hierarchical reconciliation (SKU → store → region) via MinT
- [ ] **v6.0** — Cannibalization matrix for portfolio NPI
- [ ] **v6.0** — Kalman Filter for continuous Bayesian tracking (replaces static update)
- [ ] **v6.0** — Bass p/q calibration by product category via MLE
- [ ] Multimodal Cold Start: BERT embeddings + ResNet50 CNN for luxury/beauty SKUs
- [ ] Streamlit dashboard for interactive parameter tuning

<div align="center">
  <sub>Built on the M5 Forecasting dataset · Validated on 30,000+ SKUs · Open for PRs</sub>
</div>
