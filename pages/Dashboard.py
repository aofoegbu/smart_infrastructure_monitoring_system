import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from utils.auth import check_authentication
from utils.data_generator import generate_real_time_data, load_sensor_metadata
from utils.ml_models import detect_anomalies
import time

# Authentication check
if not check_authentication():
    st.stop()

st.title("📊 Real-Time Dashboard")
st.markdown("Monitor key performance indicators and real-time sensor data")

# Dashboard controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    refresh_rate = st.selectbox("Refresh Rate", [5, 10, 30, 60], index=2)

with col2:
    sensor_count = st.slider("Sensors to Display", 10, 100, 50)

with col3:
    time_window = st.selectbox("Time Window", ["Last 1 hour", "Last 6 hours", "Last 24 hours"], index=0)

with col4:
    if st.button("🔄 Refresh Now"):
        st.rerun()

st.markdown("---")

# Generate real-time data
sensor_data = generate_real_time_data(sensor_count)
sensor_metadata = load_sensor_metadata()

# KPI Cards
st.markdown("### Key Performance Indicators")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_sensors = len(sensor_data['sensor_id'].unique())
    st.metric("Active Sensors", total_sensors, f"↑ {total_sensors - 45}")

with col2:
    avg_pressure = sensor_data['pressure'].mean()
    pressure_change = avg_pressure - 45.0
    st.metric("Avg Pressure (PSI)", f"{avg_pressure:.1f}", f"{pressure_change:+.1f}")

with col3:
    avg_flow = sensor_data['flow_rate'].mean()
    flow_change = avg_flow - 25.0
    st.metric("Avg Flow Rate (L/min)", f"{avg_flow:.1f}", f"{flow_change:+.1f}")

with col4:
    anomaly_count = len(sensor_data[sensor_data['anomaly_score'] > 0.7])
    st.metric("Anomalies Detected", anomaly_count, f"{anomaly_count - 3:+d}")

with col5:
    efficiency = np.random.uniform(85, 98)
    st.metric("System Efficiency", f"{efficiency:.1f}%", f"{efficiency - 92:.1f}%")

# Real-time charts section
st.markdown("### Real-Time Monitoring")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Pressure & Flow", "Temperature", "Anomaly Scores", "Network Health"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Pressure time series
        recent_data = sensor_data.head(30).sort_values('timestamp')
        fig_pressure = px.line(
            recent_data,
            x='timestamp',
            y='pressure',
            color='sensor_type',
            title="Pressure Readings by Sensor Type",
            labels={'pressure': 'Pressure (PSI)', 'timestamp': 'Time'}
        )
        fig_pressure.update_layout(height=400)
        st.plotly_chart(fig_pressure, use_container_width=True)
    
    with col2:
        # Flow rate vs pressure scatter
        fig_scatter = px.scatter(
            sensor_data,
            x='pressure',
            y='flow_rate',
            color='sensor_type',
            size='anomaly_score',
            title="Flow Rate vs Pressure",
            labels={'pressure': 'Pressure (PSI)', 'flow_rate': 'Flow Rate (L/min)'}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature distribution
        fig_temp = px.histogram(
            sensor_data,
            x='temperature',
            color='sensor_type',
            title="Temperature Distribution by Sensor Type",
            labels={'temperature': 'Temperature (°C)', 'count': 'Frequency'}
        )
        fig_temp.update_layout(height=400)
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with col2:
        # Temperature heatmap by location
        temp_by_location = sensor_data.groupby(['latitude', 'longitude'])['temperature'].mean().reset_index()
        fig_heatmap = px.density_mapbox(
            temp_by_location,
            lat='latitude',
            lon='longitude',
            z='temperature',
            radius=10,
            center=dict(lat=temp_by_location['latitude'].mean(), lon=temp_by_location['longitude'].mean()),
            zoom=10,
            mapbox_style="open-street-map",
            title="Temperature Heatmap"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly score distribution
        fig_anomaly = px.histogram(
            sensor_data,
            x='anomaly_score',
            title="Anomaly Score Distribution",
            labels={'anomaly_score': 'Anomaly Score', 'count': 'Frequency'}
        )
        fig_anomaly.add_vline(x=0.7, line_dash="dash", line_color="red", 
                             annotation_text="Alert Threshold")
        fig_anomaly.update_layout(height=400)
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with col2:
        # High anomaly sensors
        high_anomaly = sensor_data[sensor_data['anomaly_score'] > 0.7].sort_values('anomaly_score', ascending=False)
        if not high_anomaly.empty:
            fig_bar = px.bar(
                high_anomaly.head(10),
                x='sensor_id',
                y='anomaly_score',
                color='sensor_type',
                title="Top 10 Sensors with High Anomaly Scores"
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No high anomaly scores detected at this time.")

with tab4:
    # Network health metrics
    col1, col2 = st.columns(2)
    
    with col1:
        # Sensor status pie chart
        sensor_status = pd.DataFrame({
            'Status': ['Online', 'Offline', 'Maintenance', 'Error'],
            'Count': [total_sensors - 8, 3, 2, 3]
        })
        
        fig_pie = px.pie(
            sensor_status,
            values='Count',
            names='Status',
            title="Sensor Status Distribution",
            color_discrete_map={
                'Online': '#00CC96',
                'Offline': '#FF6B6B',
                'Maintenance': '#FECB52',
                'Error': '#EF553B'
            }
        )
        fig_pie.update_layout(height=400)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Data quality metrics
        quality_metrics = pd.DataFrame({
            'Metric': ['Data Completeness', 'Data Accuracy', 'Data Freshness', 'Data Consistency'],
            'Score': [98.5, 97.2, 99.1, 96.8]
        })
        
        fig_quality = px.bar(
            quality_metrics,
            x='Metric',
            y='Score',
            title="Data Quality Metrics",
            color='Score',
            color_continuous_scale='RdYlGn'
        )
        fig_quality.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_quality, use_container_width=True)

# Alerts and notifications
st.markdown("### Active Alerts")

# Generate sample alerts
alerts_data = []
high_anomaly_sensors = sensor_data[sensor_data['anomaly_score'] > 0.7]

for _, sensor in high_anomaly_sensors.iterrows():
    severity = "High" if sensor['anomaly_score'] > 0.9 else "Medium"
    alerts_data.append({
        'Timestamp': sensor['timestamp'],
        'Sensor ID': sensor['sensor_id'],
        'Type': sensor['sensor_type'],
        'Severity': severity,
        'Message': f"Anomaly detected - Score: {sensor['anomaly_score']:.2f}",
        'Status': 'Active'
    })

if alerts_data:
    alerts_df = pd.DataFrame(alerts_data)
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.success("✅ No active alerts at this time")

# Data export section
st.markdown("### Data Export")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Current Data"):
        csv = sensor_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📊 Generate Report"):
        st.info("Report generation feature would be implemented here")

with col3:
    if st.button("📧 Email Summary"):
        st.info("Email notification feature would be implemented here")

# Auto-refresh functionality
if st.session_state.get('auto_refresh', False):
    time.sleep(refresh_rate)
    st.rerun()

# Auto-refresh toggle
auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", key='auto_refresh')
if auto_refresh:
    st.info(f"Dashboard will refresh every {refresh_rate} seconds")
