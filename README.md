# EcoPeak-Montreal 
# 🏙️ EcoPeak Montreal: Decarbonizing the Grid, One Building at a Time

### The Problem: The "Peak Demand" Challenge

In Montreal, large buildings face a hidden challenge: **Peak Demand Penalties.** Even if a building is efficient most of the time, a single 15-minute spike in power like turning on all HVAC units at once on a cold January morning can lead to thousands of dollars in "Demand Charges" and put immense stress on the Hydro-Québec grid.

**Research goal:** Can we use Machine Learning to give building managers "eyes" into the future, detecting these peaks *before* they happen, and diagnosing *why* they occur?

---

### 🧠 The Solution: An AI-Driven Decision Support System

This application is a **Decision Support System (DSS)** designed to bridge the gap between complex raw energy data and actionable management decisions. It is organized around four analytical layers:

#### **1. Behavioral Signatures (The Past)**
Before looking forward, we must look back. The system analyzes historical "Energy Signatures," comparing how a building behaves during workdays vs weekends, revealing occupancy patterns, HVAC cycles, and baseline inefficiencies invisible in raw data.

#### **2. Financial Audit (The Impact)**
The system calculates exactly how much money is lost to peak penalties each month using a **configurable contract threshold**, adjustable per building contract and reduction targets. This turns abstract "kW" numbers into "Dollars," making the case for energy efficiency tangible to non-technical stakeholders.

#### **3. XGBoost Forecasting (The Future)**
The forecasting core uses an **XGBoost Regressor** trained on historical load and temperature data to predict the next 7 days of consumption.

> *Originally built on LSTM for my 2019 M.Sc. thesis, rebuilt in 2026 with XGBoost for faster inference, easier deployment, and better performance on tabular time-series data*

If the model predicts a spike above the contract threshold next Tuesday, the manager can proactively shift load or schedule temporary reductions before the penalty window hits.

#### **4. Anomaly Diagnosis (The "Why")**
Most anomaly detectors stop at *what*  this system attempts to answer *why*. Using **Isolation Forest (Unsupervised Learning)**, the app automatically flags unusual energy behavior and runs it through a diagnostic engine that distinguishes between:

- **Thermal Response** — weather-driven HVAC stress (e.g., extreme cold snaps)
- **Occupancy Startup Pulse** — demand spikes from synchronized building startup
- **Off-Hours Low Load** — statistically unusual but operationally
### ⚠️ Demo Note
This prototype runs on **synthetically generated data** designed to simulate Montreal commercial building load profiles. In a production deployment, the system would connect directly to real Hydro-Québec interval meter data. The architecture and methodology are the contribution — the synthetic dataset allows clean demonstration without data privacy constraints.

---
### 🛠️ Technical Stack

| Layer | Technology |
|---|---|
| Web Interface | Streamlit |
| Forecasting | XGBoost Regressor |
| Anomaly Detection | Scikit-Learn (Isolation Forest) |
| Data Pipeline | Pandas & NumPy on 1-hour interval meter data |
| Visualization | Matplotlib |

---

### 🚀 Run Locally

```bash
git clone https://github.com/SaharJavadianG/EcoPeak-Montreal
cd ecopeak-montreal
pip install -r requirements.txt
streamlit run app.py
```

---

### 🎓 Academic Context
Developed as a 2026 update to my M.Sc. research on ML-driven demand response in commercial buildings — originally prototyped with LSTM in 2019, now rebuilt for real-world deployment readiness.

🔗 **Live Demo:** *Coming soon on Hugging Face Spaces*


A spike at 7am Monday looks identical in kW to a heat wave response — but they require completely different interventions. The diagnostic layer separates them.

---
