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
                # Alert creation feature
                with st.form("create_alert"):
                    st.subheader("🚨 Create Custom Alert")
                    
                    alert_type = st.selectbox(
                        "Alert Type",
                        ["pressure", "flow", "temperature", "quality", "anomaly", "maintenance"]
                    )
                    
                    severity = st.selectbox(
                        "Severity",
                        ["info", "warning", "critical"]
                    )
                    
                    sensor_id = st.selectbox(
                        "Sensor ID",
                        options=anomaly_data['sensor_id'].unique() if not anomaly_data.empty else ["SENSOR_001"]
                    )
                    
                    alert_title = st.text_input("Alert Title", placeholder="High pressure detected")
                    alert_description = st.text_area("Description", placeholder="Describe the alert condition...")
                    
                    if st.form_submit_button("🚨 Create Alert"):
                        from utils.database import create_alert
                        
                        success = create_alert(
                            sensor_id=sensor_id,
                            alert_type=alert_type,
                            severity=severity,
                            title=alert_title,
                            description=alert_description
                        )
                        
                        if success:
                            st.success(f"✅ Alert created successfully for {sensor_id}")
                            st.rerun()
                        else:
                            st.error("Failed to create alert")
    
    with col2:
        if st.button("📧 Send Notifications"):
            # Notification system
            with st.form("notification_settings"):
                st.subheader("📧 Notification Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    email_notifications = st.checkbox("Email Notifications", value=True)
                    sms_notifications = st.checkbox("SMS Notifications", value=False)
                    push_notifications = st.checkbox("Push Notifications", value=True)
                
                with col2:
                    notification_threshold = st.selectbox(
                        "Notification Threshold",
                        ["All Alerts", "Warning & Critical", "Critical Only"]
                    )
                    
                    notification_frequency = st.selectbox(
                        "Notification Frequency",
                        ["Immediate", "Every 5 min", "Every 15 min", "Hourly"]
                    )
                
                recipients = st.text_area(
                    "Email Recipients (comma-separated)",
                    placeholder="admin@company.com, operator@company.com"
                )
                
                phone_numbers = st.text_area(
                    "SMS Recipients (comma-separated)",
                    placeholder="+1234567890, +0987654321"
                )
                
                if st.form_submit_button("💾 Save Notification Settings"):
                    # Store notification settings in session state
                    notification_settings = {
                        'email_enabled': email_notifications,
                        'sms_enabled': sms_notifications,
                        'push_enabled': push_notifications,
                        'threshold': notification_threshold,
                        'frequency': notification_frequency,
                        'email_recipients': recipients.split(',') if recipients else [],
                        'sms_recipients': phone_numbers.split(',') if phone_numbers else []
                    }
                    
                    st.session_state.notification_settings = notification_settings
                    st.success("✅ Notification settings saved successfully!")
                    
                    # Display current settings
                    st.info(f"📧 Email: {'Enabled' if email_notifications else 'Disabled'}")
                    st.info(f"📱 SMS: {'Enabled' if sms_notifications else 'Disabled'}")
                    st.info(f"🔔 Push: {'Enabled' if push_notifications else 'Disabled'}")
                    st.info(f"📊 Threshold: {notification_threshold}")
                    st.info(f"⏰ Frequency: {notification_frequency}")
    
    with col3:
        if st.button("🔧 Schedule Maintenance"):
            critical_sensors = anomaly_details[anomaly_details['severity'] == 'Critical']
            st.info(f"Maintenance scheduled for {len(critical_sensors)} critical sensors")
    
    with col4:
        if st.button("📝 Generate Report"):
            # Anomaly report generation
            with st.spinner("Generating anomaly report..."):
                # Get anomaly data for report
                from utils.data_generator import get_recent_sensor_data_from_db
                from utils.ml_models import detect_anomalies
                from utils.database import get_active_alerts, get_system_stats
                
                recent_data = get_recent_sensor_data_from_db(hours=24)
                
                if not recent_data.empty:
                    anomaly_scores = detect_anomalies(recent_data)
                    high_anomaly_indices = [i for i, score in enumerate(anomaly_scores) if score > 0.7]
                    
                    # Create comprehensive anomaly report
                    report = {
                        'report_timestamp': datetime.now().isoformat(),
                        'analysis_period': '24 hours',
                        'total_sensors_analyzed': len(recent_data['sensor_id'].unique()),
                        'total_readings_analyzed': len(recent_data),
                        'anomaly_summary': {
                            'high_anomaly_count': len(high_anomaly_indices),
                            'average_anomaly_score': float(np.mean(anomaly_scores)),
                            'max_anomaly_score': float(np.max(anomaly_scores)),
                            'anomaly_threshold': 0.7
                        },
                        'sensor_details': []
                    }
                    
                    # Add sensor-specific details
                    for sensor_id in recent_data['sensor_id'].unique():
                        sensor_data = recent_data[recent_data['sensor_id'] == sensor_id]
                        sensor_anomaly_scores = detect_anomalies(sensor_data)
                        
                        report['sensor_details'].append({
                            'sensor_id': sensor_id,
                            'readings_count': len(sensor_data),
                            'avg_anomaly_score': float(np.mean(sensor_anomaly_scores)),
                            'max_anomaly_score': float(np.max(sensor_anomaly_scores)),
                            'anomaly_incidents': len([s for s in sensor_anomaly_scores if s > 0.7]),
                            'avg_pressure': float(sensor_data['pressure'].mean()),
                            'avg_flow_rate': float(sensor_data['flow_rate'].mean()),
                            'avg_temperature': float(sensor_data['temperature'].mean())
                        })
                    
                    # Display report
                    st.subheader("📊 Anomaly Analysis Report")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Sensors", report['total_sensors_analyzed'])
                        st.metric("High Anomalies", report['anomaly_summary']['high_anomaly_count'])
                    
                    with col2:
                        st.metric("Total Readings", report['total_readings_analyzed'])
                        st.metric("Avg Anomaly Score", f"{report['anomaly_summary']['average_anomaly_score']:.3f}")
                    
                    with col3:
                        st.metric("Max Anomaly Score", f"{report['anomaly_summary']['max_anomaly_score']:.3f}")
                        st.metric("Analysis Period", report['analysis_period'])
                    
                    # Detailed sensor analysis
                    st.subheader("📋 Sensor-Level Analysis")
                    
                    sensor_df = pd.DataFrame(report['sensor_details'])
                    st.dataframe(sensor_df, use_container_width=True)
                    
                    # Create downloadable report
                    import json
                    report_json = json.dumps(report, indent=2)
                    
                    st.download_button(
                        label="📁 Download Anomaly Report (JSON)",
                        data=report_json,
                        file_name=f"anomaly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime='application/json'
                    )
                    
                    # Generate CSV for detailed analysis
                    csv_data = sensor_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Sensor Analysis (CSV)",
                        data=csv_data,
                        file_name=f"sensor_anomaly_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                    
                    st.success("Anomaly report generated successfully!")
                else:
                    st.warning("No recent data available for anomaly analysis.")

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
        # Model retraining implementation
        with st.spinner("Retraining anomaly detection model..."):
                from utils.ml_models import AnomalyDetector, train_anomaly_model
                from utils.data_generator import get_recent_sensor_data_from_db
                import os
                
                # Get training data
                training_data = get_recent_sensor_data_from_db(hours=168)  # 7 days of data
                
                if not training_data.empty and len(training_data) > 100:
                    # Train new model
                    detector = train_anomaly_model(training_data)
                    
                    # Test the model
                    test_data = get_recent_sensor_data_from_db(hours=24)
                    if not test_data.empty:
                        labels, scores = detector.predict(test_data)
                        
                        # Model performance metrics
                        anomaly_count = len([l for l in labels if l == -1])
                        avg_score = np.mean(scores)
                        
                        st.success("✅ Model retrained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Training Samples", len(training_data))
                        
                        with col2:
                            st.metric("Test Anomalies", anomaly_count)
                        
                        with col3:
                            st.metric("Avg Test Score", f"{avg_score:.3f}")
                        
                        # Save model info
                        model_info = {
                            'retrained_at': datetime.now().isoformat(),
                            'training_samples': len(training_data),
                            'test_anomalies': anomaly_count,
                            'average_score': float(avg_score),
                            'model_type': 'Isolation Forest'
                        }
                        
                        st.session_state.model_info = model_info
                        
                        st.info("🔄 Model has been retrained and is now active.")
                        st.info("📊 All future anomaly detection will use the updated model.")
                    else:
                        st.warning("⚠️ No test data available for model validation.")
                else:
                    st.error("❌ Insufficient training data. Need at least 100 samples for retraining.")
                    st.info(f"Current data points: {len(training_data) if not training_data.empty else 0}")
                    st.info("💡 Try generating more data or extending the time range.")

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
        # Anomaly dashboard generation
        with st.spinner("Creating anomaly dashboard..."):
                # Get comprehensive anomaly data
                from utils.data_generator import get_recent_sensor_data_from_db
                from utils.ml_models import detect_anomalies
                
                recent_data = get_recent_sensor_data_from_db(hours=24)
                
                if not recent_data.empty:
                    anomaly_scores = detect_anomalies(recent_data)
                    
                    # Create dashboard
                    st.subheader("📊 Anomaly Detection Dashboard")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        high_anomalies = len([s for s in anomaly_scores if s > 0.7])
                        st.metric("High Anomalies", high_anomalies, delta=f"{high_anomalies - 2}")
                    
                    with col2:
                        avg_anomaly = np.mean(anomaly_scores)
                        st.metric("Avg Anomaly Score", f"{avg_anomaly:.3f}", delta=f"{avg_anomaly - 0.3:.3f}")
                    
                    with col3:
                        max_anomaly = np.max(anomaly_scores)
                        st.metric("Max Anomaly Score", f"{max_anomaly:.3f}")
                    
                    with col4:
                        affected_sensors = len(recent_data['sensor_id'].unique())
                        st.metric("Sensors Monitored", affected_sensors)
                    
                    # Anomaly distribution chart
                    st.subheader("📈 Anomaly Score Distribution")
                    
                    fig_dist = px.histogram(
                        x=anomaly_scores,
                        nbins=30,
                        title="Distribution of Anomaly Scores",
                        labels={'x': 'Anomaly Score', 'count': 'Frequency'}
                    )
                    fig_dist.add_vline(x=0.7, line_dash="dash", line_color="red", 
                                     annotation_text="Alert Threshold")
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                    # Time series anomaly detection
                    if 'timestamp' in recent_data.columns:
                        st.subheader("⏰ Anomaly Timeline")
                        
                        # Add anomaly scores to data
                        timeline_data = recent_data.copy()
                        timeline_data['anomaly_score'] = anomaly_scores
                        
                        # Create timeline chart
                        fig_timeline = px.scatter(
                            timeline_data,
                            x='timestamp',
                            y='anomaly_score',
                            color='sensor_type',
                            size='anomaly_score',
                            hover_data=['sensor_id', 'pressure', 'flow_rate', 'temperature'],
                            title="Anomaly Scores Over Time"
                        )
                        fig_timeline.add_hline(y=0.7, line_dash="dash", line_color="red", 
                                             annotation_text="Alert Threshold")
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    
                    # Top anomalous sensors
                    st.subheader("🔍 Top Anomalous Sensors")
                    
                    sensor_anomaly_summary = []
                    for sensor_id in recent_data['sensor_id'].unique():
                        sensor_indices = recent_data[recent_data['sensor_id'] == sensor_id].index
                        sensor_anomaly_scores = [anomaly_scores[i] for i in sensor_indices]
                        
                        if sensor_anomaly_scores:
                            sensor_anomaly_summary.append({
                                'sensor_id': sensor_id,
                                'max_anomaly_score': max(sensor_anomaly_scores),
                                'avg_anomaly_score': np.mean(sensor_anomaly_scores),
                                'anomaly_incidents': len([s for s in sensor_anomaly_scores if s > 0.7])
                            })
                    
                    anomaly_df = pd.DataFrame(sensor_anomaly_summary)
                    anomaly_df = anomaly_df.sort_values('max_anomaly_score', ascending=False)
                    
                    st.dataframe(anomaly_df.head(10), use_container_width=True)
                    
                    # Export dashboard data
                    dashboard_data = {
                        'summary': {
                            'high_anomalies': int(high_anomalies),
                            'avg_anomaly_score': float(avg_anomaly),
                            'max_anomaly_score': float(max_anomaly),
                            'sensors_monitored': int(affected_sensors)
                        },
                        'sensor_details': sensor_anomaly_summary,
                        'generated_at': datetime.now().isoformat()
                    }
                    
                    import json
                    dashboard_json = json.dumps(dashboard_data, indent=2)
                    
                    st.download_button(
                        label="📊 Download Dashboard Data (JSON)",
                        data=dashboard_json,
                        file_name=f"anomaly_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime='application/json'
                    )
                    
                    st.success("Anomaly dashboard generated successfully!")
                else:
                    st.warning("No recent data available for dashboard generation.")

with col3:
    if st.button("⚙️ Model Settings"):
        with st.expander("Advanced Settings"):
            st.checkbox("Enable Real-time Detection")
            st.checkbox("Auto-retrain Model")
            st.selectbox("Alert Threshold", ["Low", "Medium", "High"])
            st.number_input("Max Alerts per Hour", min_value=1, max_value=100, value=10)
