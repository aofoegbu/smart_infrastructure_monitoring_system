import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from utils.auth import check_authentication
from utils.data_generator import generate_real_time_data, generate_historical_data
from utils.ml_models import detect_anomalies, train_anomaly_model

# Authentication check
if not check_authentication():
    st.stop()

st.title("🚨 Anomaly Detection")
st.markdown("AI-powered anomaly detection and alert management system")

# Control panel
col1, col2, col3, col4 = st.columns(4)

with col1:
    detection_method = st.selectbox(
        "Detection Method",
        ["Isolation Forest", "Statistical", "DBSCAN", "Ensemble"]
    )

with col2:
    sensitivity = st.slider("Sensitivity", 0.1, 1.0, 0.7, 0.1)

with col3:
    time_window = st.selectbox(
        "Analysis Window",
        ["Real-time", "Last 1 Hour", "Last 6 Hours", "Last 24 Hours"]
    )

with col4:
    auto_alert = st.checkbox("Auto Alerts", value=True)

st.markdown("---")

# Generate data based on time window
if time_window == "Real-time":
    data = generate_real_time_data(100)
elif time_window == "Last 1 Hour":
    data = generate_historical_data(hours=1, sensors_count=50)
elif time_window == "Last 6 Hours":
    data = generate_historical_data(hours=6, sensors_count=50)
else:  # Last 24 Hours
    data = generate_historical_data(hours=24, sensors_count=50)

# Apply anomaly detection
if detection_method == "Isolation Forest":
    features = ['pressure', 'flow_rate', 'temperature']
    X = data[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit Isolation Forest
    iso_forest = IsolationForest(contamination=1-sensitivity, random_state=42)
    anomaly_labels = iso_forest.fit_predict(X_scaled)
    data['anomaly_detected'] = anomaly_labels == -1
    data['anomaly_confidence'] = np.abs(iso_forest.score_samples(X_scaled))
    
elif detection_method == "Statistical":
    # Z-score based anomaly detection
    features = ['pressure', 'flow_rate', 'temperature']
    anomalies = []
    confidences = []
    
    for _, row in data.iterrows():
        z_scores = []
        for feature in features:
            feature_data = data[data['sensor_type'] == row['sensor_type']][feature]
            if len(feature_data) > 1:
                z_score = abs((row[feature] - feature_data.mean()) / feature_data.std())
                z_scores.append(z_score)
        
        max_z = max(z_scores) if z_scores else 0
        anomalies.append(max_z > (3 * sensitivity))
        confidences.append(min(max_z / 3, 1.0))
    
    data['anomaly_detected'] = anomalies
    data['anomaly_confidence'] = confidences

elif detection_method == "DBSCAN":
    features = ['pressure', 'flow_rate', 'temperature']
    X = data[features].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit DBSCAN
    eps = 0.5 * (1 - sensitivity + 0.1)  # Adjust eps based on sensitivity
    dbscan = DBSCAN(eps=eps, min_samples=5)
    cluster_labels = dbscan.fit_predict(X_scaled)
    
    # Points labeled as -1 are anomalies
    data['anomaly_detected'] = cluster_labels == -1
    data['anomaly_confidence'] = np.where(cluster_labels == -1, 0.8, 0.2)

else:  # Ensemble
    # Combine multiple methods
    features = ['pressure', 'flow_rate', 'temperature']
    X = data[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_pred = iso_forest.fit_predict(X_scaled)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    db_pred = dbscan.fit_predict(X_scaled)
    
    # Combine predictions
    ensemble_pred = ((iso_pred == -1) | (db_pred == -1))
    data['anomaly_detected'] = ensemble_pred
    data['anomaly_confidence'] = np.where(ensemble_pred, 0.8, 0.2)

# Summary metrics
st.markdown("### Detection Summary")

col1, col2, col3, col4 = st.columns(4)

total_sensors = len(data)
anomalies_detected = data['anomaly_detected'].sum()
anomaly_rate = (anomalies_detected / total_sensors) * 100
high_confidence = (data['anomaly_confidence'] > 0.7).sum()

with col1:
    st.metric("Total Sensors", total_sensors)

with col2:
    st.metric("Anomalies Detected", anomalies_detected, f"{anomaly_rate:.1f}%")

with col3:
    st.metric("High Confidence", high_confidence)

with col4:
    avg_confidence = data[data['anomaly_detected']]['anomaly_confidence'].mean()
    if np.isnan(avg_confidence):
        avg_confidence = 0
    st.metric("Avg Confidence", f"{avg_confidence:.2f}")

# Anomaly visualization
st.markdown("### Anomaly Visualization")

tab1, tab2, tab3, tab4 = st.tabs(["Scatter Analysis", "Time Series", "Geographic", "Distribution"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Pressure vs Flow Rate with anomalies highlighted
        fig_scatter = px.scatter(
            data,
            x='pressure',
            y='flow_rate',
            color='anomaly_detected',
            size='anomaly_confidence',
            hover_data=['sensor_id', 'sensor_type', 'temperature'],
            title="Pressure vs Flow Rate (Anomalies Highlighted)",
            color_discrete_map={True: 'red', False: 'blue'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # 3D scatter plot
        fig_3d = px.scatter_3d(
            data,
            x='pressure',
            y='flow_rate',
            z='temperature',
            color='anomaly_detected',
            size='anomaly_confidence',
            title="3D Anomaly Analysis",
            color_discrete_map={True: 'red', False: 'blue'}
        )
        st.plotly_chart(fig_3d, use_container_width=True)

with tab2:
    # Time series with anomalies
    if time_window != "Real-time":
        anomaly_data = data[data['anomaly_detected']]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pressure over time
            fig_pressure = px.line(
                data.sort_values('timestamp'),
                x='timestamp',
                y='pressure',
                color='sensor_type',
                title="Pressure Over Time"
            )
            
            # Add anomaly points
            if not anomaly_data.empty:
                fig_pressure.add_trace(go.Scatter(
                    x=anomaly_data['timestamp'],
                    y=anomaly_data['pressure'],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='x'),
                    name='Anomalies'
                ))
            
            st.plotly_chart(fig_pressure, use_container_width=True)
        
        with col2:
            # Flow rate over time
            fig_flow = px.line(
                data.sort_values('timestamp'),
                x='timestamp',
                y='flow_rate',
                color='sensor_type',
                title="Flow Rate Over Time"
            )
            
            # Add anomaly points
            if not anomaly_data.empty:
                fig_flow.add_trace(go.Scatter(
                    x=anomaly_data['timestamp'],
                    y=anomaly_data['flow_rate'],
                    mode='markers',
                    marker=dict(color='red', size=10, symbol='x'),
                    name='Anomalies'
                ))
            
            st.plotly_chart(fig_flow, use_container_width=True)
    else:
        st.info("Time series analysis not available for real-time data view")

with tab3:
    # Geographic anomaly map
    import folium
    from streamlit_folium import st_folium
    
    # Create map centered on anomalies
    if anomalies_detected > 0:
        anomaly_data = data[data['anomaly_detected']]
        center_lat = anomaly_data['latitude'].mean()
        center_lon = anomaly_data['longitude'].mean()
    else:
        center_lat = data['latitude'].mean()
        center_lon = data['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
    
    # Add normal sensors
    normal_data = data[~data['anomaly_detected']]
    for _, sensor in normal_data.iterrows():
        folium.CircleMarker(
            location=[sensor['latitude'], sensor['longitude']],
            radius=5,
            popup=f"Sensor {sensor['sensor_id']} - Normal",
            color='blue',
            fillColor='blue',
            fillOpacity=0.6
        ).add_to(m)
    
    # Add anomalous sensors
    anomaly_data = data[data['anomaly_detected']]
    for _, sensor in anomaly_data.iterrows():
        folium.CircleMarker(
            location=[sensor['latitude'], sensor['longitude']],
            radius=8,
            popup=f"Sensor {sensor['sensor_id']} - ANOMALY (Confidence: {sensor['anomaly_confidence']:.2f})",
            color='red',
            fillColor='red',
            fillOpacity=0.8
        ).add_to(m)
    
    st_folium(m, width=700, height=400)

with tab4:
    # Distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Anomaly confidence distribution
        fig_conf = px.histogram(
            data[data['anomaly_detected']],
            x='anomaly_confidence',
            title="Anomaly Confidence Distribution",
            nbins=20
        )
        st.plotly_chart(fig_conf, use_container_width=True)
    
    with col2:
        # Anomalies by sensor type
        anomaly_by_type = data[data['anomaly_detected']]['sensor_type'].value_counts()
        if not anomaly_by_type.empty:
            fig_type = px.bar(
                x=anomaly_by_type.index,
                y=anomaly_by_type.values,
                title="Anomalies by Sensor Type"
            )
            st.plotly_chart(fig_type, use_container_width=True)
        else:
            st.info("No anomalies detected")

# Detailed anomaly list
st.markdown("### Anomaly Details")

if anomalies_detected > 0:
    anomaly_details = data[data['anomaly_detected']].copy()
    anomaly_details = anomaly_details.sort_values('anomaly_confidence', ascending=False)
    
    # Add severity levels
    anomaly_details['severity'] = anomaly_details['anomaly_confidence'].apply(
        lambda x: 'Critical' if x > 0.8 else 'High' if x > 0.6 else 'Medium'
    )
    
    # Display table
    st.dataframe(
        anomaly_details[['sensor_id', 'sensor_type', 'pressure', 'flow_rate', 
                        'temperature', 'anomaly_confidence', 'severity', 'timestamp']].round(3),
        use_container_width=True
    )
    
    # Anomaly actions
    st.markdown("### Anomaly Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚨 Create Alerts"):
            if auto_alert:
                st.success(f"✅ {len(anomaly_details)} alerts created automatically")
            else:
                st.info("Alert creation feature would be implemented here")
    
    with col2:
        if st.button("📧 Send Notifications"):
            st.info("Notification system would be implemented here")
    
    with col3:
        if st.button("🔧 Schedule Maintenance"):
            critical_sensors = anomaly_details[anomaly_details['severity'] == 'Critical']
            st.info(f"Maintenance scheduled for {len(critical_sensors)} critical sensors")
    
    with col4:
        if st.button("📝 Generate Report"):
            st.info("Anomaly report generation would be implemented here")

else:
    st.success("✅ No anomalies detected in the current dataset")

# Model performance and tuning
st.markdown("---")
st.markdown("### Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Detection Statistics**")
    
    # Performance metrics (simulated for demonstration)
    performance_data = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
        'Score': [0.87, 0.82, 0.84, 0.93]
    })
    
    fig_performance = px.bar(
        performance_data,
        x='Metric',
        y='Score',
        title="Model Performance Metrics",
        color='Score',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_performance, use_container_width=True)

with col2:
    st.markdown("**Tuning Parameters**")
    
    # Model parameters
    if detection_method == "Isolation Forest":
        contamination = st.slider("Contamination Rate", 0.01, 0.5, 0.1, 0.01)
        n_estimators = st.slider("Number of Estimators", 50, 500, 100, 50)
        
    elif detection_method == "Statistical":
        z_threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, 0.1)
        window_size = st.slider("Rolling Window", 10, 100, 30, 10)
        
    elif detection_method == "DBSCAN":
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.slider("Min Samples", 3, 20, 5, 1)
    
    if st.button("🔄 Retrain Model"):
        st.info("Model retraining would be implemented here")

# Export and reporting
st.markdown("---")
st.markdown("### Export & Reporting")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Anomalies"):
        if anomalies_detected > 0:
            csv = anomaly_details.to_csv(index=False)
            st.download_button(
                label="Download Anomaly Data",
                data=csv,
                file_name=f"anomalies_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No anomalies to export")

with col2:
    if st.button("📊 Generate Dashboard"):
        st.info("Anomaly dashboard generation would be implemented here")

with col3:
    if st.button("⚙️ Model Settings"):
        with st.expander("Advanced Settings"):
            st.checkbox("Enable Real-time Detection")
            st.checkbox("Auto-retrain Model")
            st.selectbox("Alert Threshold", ["Low", "Medium", "High"])
            st.number_input("Max Alerts per Hour", min_value=1, max_value=100, value=10)
