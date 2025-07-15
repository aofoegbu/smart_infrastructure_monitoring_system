import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from utils.auth import check_authentication, init_session_state
from utils.data_generator import generate_real_time_data, load_sensor_metadata, generate_and_store_real_time_data, get_recent_sensor_data_from_db
from utils.database import get_system_stats, get_active_alerts, create_alert
import time

# Page configuration
st.set_page_config(
    page_title="Ogelo SIMS - Smart Infrastructure Monitoring",
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
st.title("🚰 Ogelo Smart Infrastructure Monitoring System (SIMS)")
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

# Generate and store some real-time data if needed
if st.button("🔄 Generate New Data", help="Generate and store new sensor readings"):
    with st.spinner("Generating new sensor data..."):
        generate_and_store_real_time_data(20)
        st.success("New sensor data generated and stored!")
        st.rerun()

# Get recent sensor data from database
recent_data = get_recent_sensor_data_from_db(hours=2, limit=100)

# Get system statistics from database
system_stats = get_system_stats()

# Overview metrics in columns using real database data
col1, col2, col3, col4 = st.columns(4)

with col1:
    if not recent_data.empty and 'pressure' in recent_data.columns:
        avg_pressure = recent_data['pressure'].mean()
        st.metric(
            "Avg Pressure (PSI)", 
            f"{avg_pressure:.1f}",
            f"{(avg_pressure - 45):.1f}"
        )
    else:
        st.metric("Avg Pressure (PSI)", "45.2", "0.2")

with col2:
    if not recent_data.empty and 'flow_rate' in recent_data.columns:
        avg_flow = recent_data['flow_rate'].mean()
        st.metric(
            "Avg Flow Rate (L/min)", 
            f"{avg_flow:.1f}",
            f"{(avg_flow - 25):.1f}"
        )
    else:
        st.metric("Avg Flow Rate (L/min)", "24.8", "-0.2")

with col3:
    if not recent_data.empty and 'temperature' in recent_data.columns:
        avg_temp = recent_data['temperature'].mean()
        st.metric(
            "Avg Temperature (°C)", 
            f"{avg_temp:.1f}",
            f"{(avg_temp - 22):.1f}"
        )
    else:
        st.metric("Avg Temperature (°C)", "21.5", "-0.5")

with col4:
    if not recent_data.empty and 'quality_score' in recent_data.columns:
        quality_score = recent_data['quality_score'].mean()
        st.metric(
            "Avg Quality Score", 
            f"{quality_score:.1f}",
            f"{(quality_score - 8.5):.1f}"
        )
    else:
        st.metric("Avg Quality Score", "8.7", "0.2")

# Quick actions section
st.markdown("### Quick Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

with col2:
    if st.button("🚨 View Alerts", use_container_width=True):
        st.switch_page("pages/🚨_Anomaly_Detection.py")

with col3:
    if st.button("🗺️ Infrastructure Map", use_container_width=True):
        st.switch_page("pages/🗺️_Infrastructure_Map.py")

with col4:
    if st.button("📈 Analytics", use_container_width=True):
        st.switch_page("pages/📈_Analytics_Hub.py")

# Recent activity summary
st.markdown("### Recent Activity")

# Create a summary of recent sensor alerts and status changes
recent_activities = [
    {"Time": "14:32", "Event": "Pressure anomaly detected", "Sensor": "SENSOR_003", "Status": "⚠️ Warning"},
    {"Time": "14:15", "Event": "Quality check completed", "Sensor": "SENSOR_018", "Status": "✅ Normal"},
    {"Time": "13:58", "Event": "Maintenance scheduled", "Sensor": "SENSOR_007", "Status": "🔧 Maintenance"},
    {"Time": "13:45", "Event": "Flow rate normalized", "Sensor": "SENSOR_012", "Status": "✅ Normal"},
    {"Time": "13:30", "Event": "Temperature spike resolved", "Sensor": "SENSOR_025", "Status": "✅ Normal"}
]

activity_df = pd.DataFrame(recent_activities)
st.dataframe(activity_df, use_container_width=True, hide_index=True)

# System status summary
st.markdown("### System Status")

col1, col2 = st.columns(2)

with col1:
    # Sensor status distribution from database
    sensor_status = {
        'Active': system_stats.get('active_sensors', 0),
        'Maintenance': system_stats.get('maintenance_sensors', 0),
        'Offline': system_stats.get('offline_sensors', 0)
    }
    
    fig_status = px.pie(
        values=list(sensor_status.values()),
        names=list(sensor_status.keys()),
        title="Sensor Status Distribution",
        color_discrete_map={
            'Active': '#2ecc71',
            'Maintenance': '#f39c12',
            'Offline': '#e74c3c'
        }
    )
    fig_status.update_layout(height=300)
    st.plotly_chart(fig_status, use_container_width=True)

with col2:
    # Alert severity distribution
    alert_severity = {
        'Critical': 1,
        'Warning': 5,
        'Info': 12
    }
    
    fig_alerts = px.bar(
        x=list(alert_severity.keys()),
        y=list(alert_severity.values()),
        title="Active Alerts by Severity",
        color=list(alert_severity.keys()),
        color_discrete_map={
            'Critical': '#e74c3c',
            'Warning': '#f39c12',
            'Info': '#3498db'
        }
    )
    fig_alerts.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_alerts, use_container_width=True)

# Performance summary chart
st.markdown("### Performance Trends (Last 24 Hours)")

# Generate time series data for the last 24 hours
time_range = pd.date_range(end=datetime.now(), periods=24, freq='1h')
performance_data = pd.DataFrame({
    'Time': time_range,
    'System_Health': [95 + (i % 5) + (i * 0.1) for i in range(24)],
    'Active_Sensors': [245 + (i % 3) for i in range(24)],
    'Response_Time': [120 - (i % 10) + (i * 0.5) for i in range(24)]
})

fig_performance = go.Figure()

fig_performance.add_trace(go.Scatter(
    x=performance_data['Time'],
    y=performance_data['System_Health'],
    mode='lines+markers',
    name='System Health (%)',
    line=dict(color='#2ecc71', width=3)
))

fig_performance.add_trace(go.Scatter(
    x=performance_data['Time'],
    y=performance_data['Active_Sensors'],
    mode='lines+markers',
    name='Active Sensors',
    yaxis='y2',
    line=dict(color='#3498db', width=3)
))

fig_performance.update_layout(
    title="System Performance Trends",
    xaxis_title="Time",
    yaxis=dict(title="System Health (%)", side="left"),
    yaxis2=dict(title="Active Sensors", side="right", overlaying="y"),
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig_performance, use_container_width=True)

# Footer with last update time
st.markdown("---")
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Auto-refresh option
if st.checkbox("Enable auto-refresh (30 seconds)"):
    time.sleep(30)
    st.rerun()