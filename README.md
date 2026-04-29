# PM-FSS — Predictive Maintenance & Failure Forecasting System

**Presented at AICraft 3.1 · Amity Centre for Artificial Intelligence**

A Counterfactual & Decision-Aware AI Framework for Cost-Optimized
Industrial Predictive Maintenance and Failure Forecasting.

> This repository contains the research poster, system design, and
> technical documentation submitted at AICraft 2.0.
> Code implementation is in progress.

---

## The Problem

Industrial assets like turbofan engines fail unpredictably.
- Reactive maintenance costs **3–5× more** than planned maintenance
- Calendar-based schedules over-maintain healthy assets
- Existing predictive models forecast failures but ignore cost-optimal decisions

**The gap:** No framework that predicts degradation *and* prescribes
the most economically optimal maintenance action.

---

## Our Solution

A **six-layer decision intelligence framework** for turbofan predictive maintenance: Sensor Data Ingestion
↓
Health Index Generation (1 = healthy → 0 = failed)
↓
LSTM-based Failure Forecasting (RUL Prediction)
↓
Decision-Aware Cost Optimization (repair / defer / monitor)
↓
Counterfactual "What-If" Reasoning Engine
↓
Continuous Learning & Feedback Loop

---

## Dataset

**NASA CMAPSS Turbofan Engine Dataset**
- 21 sensor channels per engine cycle
- Captures fan speed, core speed, temperatures, pressures
- 4 fault modes (FD001–FD004)
- Full degradation trajectory from healthy state to failure

---

## Key Components

| Component | Description |
|-----------|-------------|
| Health Index (HI) | Normalized score tracking engine degradation over time |
| LSTM Forecasting | Captures temporal sensor patterns to predict Remaining Useful Life (RUL) |
| Cost Optimization | Compares repair, defer, and run-to-failure using Expected Total Cost (ETC) |
| Counterfactual Engine | Simulates "what-if" scenarios to justify proactive decisions |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

---

## Team

Tushar Mishra · Kushal Singh · Archit Yadav · Mayank Singh
Amity University, Noida — CSE Department

---

## Status

Research poster presented at AICraft 3.1.
Implementation ongoing.
