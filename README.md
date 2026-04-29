# Predictive Maintenance & Failure Forecasting System
### AICraft 2.0 — Amity Centre for Artificial Intelligence

A Counterfactual & Decision-Aware AI Framework for Cost-Optimized 
Industrial Predictive Maintenance and Failure Forecasting.

---

## Problem

Industrial assets like turbofan engines fail unpredictably.
Reactive maintenance is 3–5× more expensive than planned maintenance.
Existing predictive models forecast failures but don't answer:
**what's the cost-optimal action given economic uncertainty?**

---

## What We Built

A six-layer decision intelligence framework that:
- Tracks real-time engine health using a normalized Health Index (0–1)
- Forecasts Remaining Useful Life (RUL) using an LSTM-based model
- Recommends optimal action: repair now / delay / run-to-failure
- Simulates "what-if" counterfactual scenarios for strategic decisions

---

## Dataset

NASA CMAPSS Turbofan Engine Dataset
- 21 sensor channels per engine cycle
- Captures full degradation trajectories across 4 fault modes (FD001–FD004)
- Sensors: fan speed, core speed, temperatures, pressures, operating conditions

---

## Tech Stack

`Python` · `LSTM` · `DNN` · `NumPy` · `Pandas` · `Matplotlib`

## System Architecture

The system follows a multi-layer pipeline:

1. Data Ingestion → Sensor data preprocessing
2. Health Index Computation → Normalized degradation score
3. RUL Prediction → LSTM-based forecasting
4. Decision Layer → Cost-aware action selection
5. Simulation Layer → Counterfactual analysis

---

## Results

- Predictive maintenance reduces costs by 20–50% vs reactive approach
- LSTM model continuously updates RUL predictions as new sensor data arrives
- Cost optimizer selects actions using Expected Total Cost (ETC) formula

---

## Team

Tushar Mishra · Kushal Singh · Archit Yadav · Mayank Singh
Amity University, Noida — CSE Department
Presented at **AICraft 3.1**, Amity Centre for Artificial Intelligence

