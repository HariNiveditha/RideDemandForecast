# 🚕 Ride Demand Forecasting

A machine learning project that predicts ride demand using historical ride data, weather conditions, and time-based patterns.

## 📌 Overview

The project focuses on forecasting the number of rides within a given time window using historical demand patterns and external weather information.

The model was developed through multiple iterations by progressively incorporating weather data, lag features, rolling statistics, and cyclical time features.

---

## 🎯 Objectives

- Predict ride demand accurately using historical data.
- Capture recurring temporal patterns in ride demand.
- Analyze the contribution of weather conditions to demand prediction.
- Use historical demand through lag and rolling features.
- Compare model performance across different feature configurations.

---

## 📊 Features

The final model uses the following feature groups:

| Feature Group | Details |
|---|---|
| **Hourly Weather** | Hourly weather information associated with ride records |
| **Lag Features** | `lag_1`, `lag_24`, `lag_168` — demand 1 hour, 1 day, and 1 week earlier |
| **Rolling Features** | `rolling_mean_24` — 24-hour rolling average demand |
| **Cyclical Time Features** | Hour-of-day and day-of-week represented using sine/cosine transformations |

## 🤖 Model

**Algorithm:** Random Forest Regressor

The Random Forest model was evaluated across multiple feature engineering stages to measure the effect of weather and historical demand features.

---

## 📈 Results

| Version | Feature Configuration | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| Baseline | Random Forest without engineered features | 51.50 | 31.19 | 0.619 |
| + Hourly Weather | Hourly weather | 24.90 | — | 0.911 |
| Without Weather | Comparison model | 26.87 | — | 0.896 |
| Daily Weather + Features | Daily weather + lags + rolling + cyclical | 18.42 | 10.60 | 0.951 |
| **Final Model** | **Hourly weather + lags + rolling + cyclical** | **18.44** | — | **0.951** |

---

## 🔍 Key Findings

### Weather

Adding hourly weather features improved the R² score from **0.619 to 0.911** compared with the baseline model.

### Lag Features

Lag features had the largest contribution to prediction performance.

Among the engineered features, `lag_168` — demand from the same hour one week earlier — showed approximately **85% feature importance** in the corresponding feature-importance analysis.

### Weather Importance After Lag Features

After adding lag and rolling features, the relative importance of weather features decreased to approximately **1–1.3%**.

This indicates that historical demand patterns capture a large portion of the predictable variation in ride demand.

### Hourly vs. Daily Weather

Two feature configurations were compared:

- Daily weather + lag + rolling + cyclical features  
  **RMSE: 18.42 | R²: 0.951**

- Hourly weather + lag + rolling + cyclical features  
  **RMSE: 18.44 | R²: 0.951**

The results were very similar, indicating that increasing weather resolution had little effect once historical demand features were included.

---

## 🔬 Methodology

```text
Historical Ride Data
        ↓
Data Cleaning & Preprocessing
        ↓
Weather Data Integration
        ↓
Feature Engineering
        ↓
Lag Features
        ↓
Rolling Features
        ↓
Cyclical Time Encoding
        ↓
Random Forest Regressor
        ↓
Model Evaluation
        ↓
Feature Importance Analysis
```
