import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils.auth import check_authentication, init_session_state
from utils.data_generator import generate_real_time_data, load_sensor_metadata
import time

# Page configuration
st.set_page_config(
    page_title="SIMS - Smart Infrastructure Monitoring",
    page_icon="🚰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state and authentication
init_session_state()

# Authentication check
if not check_authentication():
    st.stop()

# Main page content
st.title("🚰 Smart Infrastructure Monitoring System (SIMS)")
st.markdown("---")

# Welcome message with user info
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown(f"### Welcome, {st.session_state.user_role.title()}")
    st.markdown("Monitor and optimize your infrastructure networks in real-time")

with col2:
    st.metric("Active Sensors", "247", "↑ 3")

with col3:
    st.metric("System Health", "98.7%", "↑ 0.3%")

# Quick overview metrics
st.markdown("### System Overview")

# Generate real-time data for overview
sensor_data = generate_real_time_data(50)
sensor_metadata = load_sensor_metadata()

# Overview metrics in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_pressure = sensor_data['pressure'].mean()
    st.metric(
        "Avg Pressure (PSI)", 
        f"{avg_pressure:.1f}",
        f"{(avg_pressure - 45):.1f}"
    )

with col2:
    avg_flow = sensor_data['flow_rate'].mean()
    st.metric(
        "Avg Flow Rate (L/min)", 
        f"{avg_flow:.1f}",
        f"{(avg_flow - 25):.1f}"
    )

with col3:
    avg_temp = sensor_data['temperature'].mean()
    st.metric(
        "Avg Temperature (°C)", 
        f"{avg_temp:.1f}",
        f"{(avg_temp - 22):.1f}"
    )

with col4:
    alerts_count = len(sensor_data[sensor_data['anomaly_score'] > 0.7])
    st.metric(
        "Active Alerts", 
        alerts_count,
        f"{alerts_count - 2}"
    )

# Real-time charts
st.markdown("### Real-Time Monitoring")

col1, col2 = st.columns(2)

with col1:
    # Pressure trend chart
    fig_pressure = px.line(
        sensor_data.head(20), 
        x='timestamp', 
        y='pressure',
        color='sensor_id',
        title="Recent Pressure Readings",
        labels={'pressure': 'Pressure (PSI)', 'timestamp': 'Time'}
    )
    fig_pressure.update_layout(height=300)
    st.plotly_chart(fig_pressure, use_container_width=True)

with col2:
    # Flow rate distribution
    fig_flow = px.histogram(
        sensor_data, 
        x='flow_rate',
        nbins=20,
        title="Flow Rate Distribution",
        labels={'flow_rate': 'Flow Rate (L/min)', 'count': 'Frequency'}
    )
    fig_flow.update_layout(height=300)
    st.plotly_chart(fig_flow, use_container_width=True)

# System status indicators
st.markdown("### System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("✅ Data Pipeline: Operational")
    st.info("ℹ️ Last Update: " + datetime.now().strftime("%H:%M:%S"))

with col2:
    st.success("✅ ML Models: Active")
    st.info("ℹ️ Anomaly Detection: Running")

with col3:
    st.success("✅ Database: Connected")
    st.info("ℹ️ Response Time: 12ms")

# Navigation help
st.markdown("---")
st.markdown("""
### Navigation Guide
- **📊 Dashboard**: Real-time metrics and KPI monitoring
- **🗺️ Infrastructure Map**: Interactive sensor location mapping
- **📈 Analytics Hub**: Advanced data analysis and insights
- **🚨 Anomaly Detection**: ML-powered anomaly alerts and management
- **🔧 Data Management**: Data governance, quality, and compliance
- **⚡ System Health**: Pipeline monitoring and performance metrics

Use the sidebar to navigate between different sections of the system.
""")

# Auto-refresh option
if st.checkbox("Auto-refresh data (every 30 seconds)"):
    time.sleep(30)
    st.rerun()
