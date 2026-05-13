import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xgboost as xgb
from sklearn.ensemble import IsolationForest
import os

#APP SETUP & THEME
st.set_page_config(page_title="EcoPeak Montreal | Thesis Project", layout="wide")

# Custom CSS for the Concordia branding
st.markdown("""
    <div style="background-color:#912338; padding:20px; border-radius:10px; text-align:center; border-bottom: 5px solid #ffb81c; margin-bottom: 25px;">
        <h1 style="color:white; margin:0; font-family:serif; letter-spacing: 2px;">BUILDING ENERGY DSS</h1>
        <p style="color:#ffb81c; margin:5px 0 0 0; font-weight:bold; font-size:1.2em;">Decision Support System for Peak Demand Mitigation</p>
        <p style="color:white; font-style:italic; font-size:0.9em; opacity: 0.9;">Thesis Implementation - Montreal 2026</p>
    </div>
""", unsafe_allow_html=True)

# Keep the model in memory so it doesn't retrain on every click
if 'model' not in st.session_state:
    st.session_state.model = None

# SIDEBAR: USER INPUTS 
st.sidebar.header("📋 App Settings")
# The 1650kW limit is based on the building's specific utility contract
contract_limit = st.sidebar.number_input("Current Contract Limit (kW)", value=1650.0)
demand_rate = 18.20 
uploaded_file = st.sidebar.file_uploader("Upload New Load Data (.csv or .xlsx)", type=["csv", "xlsx"])

# --- DATA CLEANING FUNCTIONS ---
def prepare_data(df):
    # Clean up column names and handle naming variations
    df.columns = df.columns.str.strip()
    if 'Electricity Load (kW)' in df.columns:
        df = df.rename(columns={'Electricity Load (kW)': 'Electricity Load'})
    
    # Standardizing time and extracting features for ML
    df['Date and Time'] = pd.to_datetime(df['Date and Time'])
    df['Hour'] = df['Date and Time'].dt.hour
    df['DayOfWeek_Num'] = df['Date and Time'].dt.dayofweek
    df['DayOfWeek'] = df['Date and Time'].dt.day_name()
    df['Month'] = df['Date and Time'].dt.month_name()
    df['Month_Num'] = df['Date and Time'].dt.month
    
    # Adding 'Lag' features so the model can see previous consumption patterns
    df['load_lag_24h'] = df['Electricity Load'].shift(24)
    df['load_lag_1h'] = df['Electricity Load'].shift(1)
    return df.dropna().reset_index(drop=True)

def train_xgb(df):
    # Training the XGBoost regressor using time and temperature as main drivers
    features = ['Hour', 'DayOfWeek_Num', 'Month_Num', 'Temperature', 'load_lag_24h', 'load_lag_1h']
    X = df[features]
    y = df['Electricity Load']
    model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, objective='reg:squarederror')
    model.fit(X, y)
    st.session_state.model = model
    return model

# --- DATA LOADING LOGIC ---
df_raw = None

if uploaded_file is not None:
    # Handle user-provided data
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
else:
    # Default to the thesis benchmark file if no file is uploaded
    default_path = "data/benchmark_data.csv"
    if os.path.exists(default_path):
        df_raw = pd.read_csv(default_path)
        st.success(f"📂 Currently using: {default_path}")
    else:
        st.info("👋 Hello! Please upload your energy data to start the analysis.")

# APP TABS & ANALYSIS 
if df_raw is not None:
    df = prepare_data(df_raw)
    
    st.markdown("### 📊 Operational Overview")
    tab1, tab2, tab3, tab4 = st.tabs(["Energy Patterns", "Cost Analysis", "7-Day Prediction", "Problem Diagnosis"])

    # Tab 1: Visualizing the difference between workdays and weekends
    with tab1:
        st.header("Behavioral Energy Signatures")
        df['is_weekend'] = df['DayOfWeek_Num'] >= 5
        wd_avg = df[df['is_weekend'] == False].groupby('Hour')['Electricity Load'].mean()
        we_avg = df[df['is_weekend'] == True].groupby('Hour')['Electricity Load'].mean()
        
        # Keep scales identical so comparisons are fair
        global_y_max = max(wd_avg.max(), we_avg.max()) * 1.1 
        
        c1, c2 = st.columns(2)
        with c1:
            fig_wd, ax_wd = plt.subplots(figsize=(6, 4))
            ax_wd.plot(wd_avg.index, wd_avg.values, color='#912338', linewidth=2)
            ax_wd.fill_between(wd_avg.index, wd_avg.values, color='#912338', alpha=0.1)
            ax_wd.set_title("Weekday Load Profile (Avg)")
            ax_wd.set_ylim(0, global_y_max)
            ax_wd.set_ylabel("Load (kW)")
            ax_wd.grid(True, alpha=0.2)
            st.pyplot(fig_wd); plt.close(fig_wd)
            
        with c2:
            fig_we, ax_we = plt.subplots(figsize=(6, 4))
            ax_we.plot(we_avg.index, we_avg.values, color='#ffb81c', linewidth=2)
            ax_we.fill_between(we_avg.index, we_avg.values, color='#ffb81c', alpha=0.1)
            ax_we.set_title("Weekend Load Profile (Avg)")
            ax_we.set_ylim(0, global_y_max)
            ax_we.set_ylabel("Load (kW)")
            ax_we.grid(True, alpha=0.2)
            st.pyplot(fig_we); plt.close(fig_we)

    # Tab 2: Calculating peak demand penalties
    with tab2:
        st.header("Financial Penalty Audit")
        billing_results = []
        for month in df['Month'].unique():
            m_data = df[df['Month'] == month]
            m_peak = m_data['Electricity Load'].max()
            m_gap = max(0, m_peak - contract_limit)
            billing_results.append({
                "Month": month, 
                "Peak (kW)": round(m_peak, 2), 
                "Penalty ($)": round(m_gap * demand_rate, 2)
            })
            
            fig_p, ax_p = plt.subplots(figsize=(12, 2.5))
            ax_p.plot(m_data['Date and Time'], m_data['Electricity Load'], color='#2E86C1', alpha=0.6)
            ax_p.axhline(contract_limit, color='red', linestyle='--', label="Contract Limit")
            ax_p.set_title(f"Monthly Distribution: {month}")
            st.pyplot(fig_p); plt.close(fig_p)
            
        st.subheader("Billing Summary")
        st.table(pd.DataFrame(billing_results))

    # Tab 3: Running the XGBoost forecast
    with tab3:
        st.header("XGBoost Load Forecast (Next 168 Hours)")
        if st.session_state.model is None: 
            with st.spinner("Calculating forecast..."): train_xgb(df)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Model Architecture", "XGBoost")
        with m_col2:
            st.metric("Primary Metric", "RMSE")
        with m_col3:
            st.metric("Objective Function", "reg:squarederror")
            
        history_load = df['Electricity Load'].tolist()
        last_time = df['Date and Time'].max()
        preds = []
        for _ in range(168): 
            last_time += pd.Timedelta(hours=1)
            cur_feat = pd.DataFrame([{
                'Hour': last_time.hour, 
                'DayOfWeek_Num': last_time.dayofweek, 
                'Month_Num': last_time.month, 
                'Temperature': df['Temperature'].mean(), 
                'load_lag_24h': history_load[-24], 
                'load_lag_1h': history_load[-1]
            }])
            p = st.session_state.model.predict(cur_feat)[0]
            preds.append(p)
            history_load.append(p)
        
        f_times = pd.date_range(df['Date and Time'].max(), periods=len(preds)+1, freq='h')[1:]
        fig_f, ax_f = plt.subplots(figsize=(14, 6))
        ax_f.plot(f_times, preds, color='#912338', label="Predicted Consumption")
        ax_f.axhline(contract_limit, color='red', linestyle=':', label="Penalty Limit")
        ax_f.xaxis.set_major_formatter(mdates.DateFormatter('%a %d'))
        ax_f.legend()
        st.pyplot(fig_f); plt.close(fig_f)

    # Tab 4: Using Isolation Forest to find unusual events
    with tab4:
        st.header("Automated Anomaly Diagnostic")
        iso = IsolationForest(contamination=0.015, random_state=42)
        df['anomaly'] = iso.fit_predict(df[['Electricity Load', 'Temperature', 'Hour', 'DayOfWeek_Num']])
        anomalies = df[df['anomaly'] == -1].copy()
        
        t_mean = df['Temperature'].mean()
        def diagnose(row):
            if abs(row['Temperature'] - t_mean) > 8: return "Weather-Related (HVAC Stress)"
            elif 6 <= row['Hour'] <= 9 and row['DayOfWeek_Num'] < 5: return "Morning Startup Spike"
            else: return "Unknown/Technical Fault"
        
        anomalies['Root Cause Hint'] = anomalies.apply(diagnose, axis=1)
        st.dataframe(anomalies[['Date and Time', 'Electricity Load', 'Temperature', 'Root Cause Hint']].sort_values(by='Date and Time'))
