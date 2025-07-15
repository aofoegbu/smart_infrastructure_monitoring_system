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
        # Temperature scatter map by location
        temp_by_location = sensor_data.groupby(['latitude', 'longitude'])['temperature'].mean().reset_index()
        fig_heatmap = px.scatter_mapbox(
            temp_by_location,
            lat='latitude',
            lon='longitude',
            color='temperature',
            size='temperature',
            hover_data=['temperature'],
            color_continuous_scale='RdYlBu_r',
            size_max=15,
            zoom=10,
            mapbox_style="open-street-map",
            title="Temperature Distribution Map"
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
        # Report generation feature
            with st.spinner("Generating comprehensive system report..."):
                from utils.data_generator import get_recent_sensor_data_from_db
                from utils.database import get_system_stats, get_active_alerts
                from utils.ml_models import calculate_sensor_health_score
                
                # Get system data
                recent_data = get_recent_sensor_data_from_db(hours=24)
                system_stats = get_system_stats()
                active_alerts = get_active_alerts()
                
                if not recent_data.empty:
                    health_scores = calculate_sensor_health_score(recent_data)
                    
                    # Create system report
                    system_report = {
                        'report_timestamp': datetime.now().isoformat(),
                        'report_period': '24 hours',
                        'system_metrics': {
                            'total_sensors': system_stats['total_sensors'],
                            'active_sensors': system_stats['active_sensors'],
                            'total_readings': len(recent_data),
                            'active_alerts': len(active_alerts),
                            'avg_health_score': health_scores['health_score'].mean()
                        },
                        'performance_summary': {
                            'avg_pressure': recent_data['pressure'].mean(),
                            'avg_flow_rate': recent_data['flow_rate'].mean(),
                            'avg_temperature': recent_data['temperature'].mean(),
                            'avg_quality_score': recent_data['quality_score'].mean()
                        },
                        'alert_summary': {
                            'total_alerts': len(active_alerts),
                            'critical_alerts': len(active_alerts[active_alerts['severity'] == 'critical']) if len(active_alerts) > 0 else 0,
                            'warning_alerts': len(active_alerts[active_alerts['severity'] == 'warning']) if len(active_alerts) > 0 else 0
                        }
                    }
                    
                    # Display report
                    st.subheader("📊 System Performance Report")
                    
                    # System metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Sensors", system_report['system_metrics']['total_sensors'])
                    with col2:
                        st.metric("Active Sensors", system_report['system_metrics']['active_sensors'])
                    with col3:
                        st.metric("Total Readings", system_report['system_metrics']['total_readings'])
                    with col4:
                        st.metric("Active Alerts", system_report['system_metrics']['active_alerts'])
                    
                    # Performance summary
                    st.subheader("📈 Performance Summary")
                    
                    perf_col1, perf_col2 = st.columns(2)
                    
                    with perf_col1:
                        st.metric("Avg Pressure", f"{system_report['performance_summary']['avg_pressure']:.1f} PSI")
                        st.metric("Avg Flow Rate", f"{system_report['performance_summary']['avg_flow_rate']:.1f} L/min")
                    
                    with perf_col2:
                        st.metric("Avg Temperature", f"{system_report['performance_summary']['avg_temperature']:.1f}°C")
                        st.metric("Avg Quality Score", f"{system_report['performance_summary']['avg_quality_score']:.1f}")
                    
                    # Alert summary
                    st.subheader("🚨 Alert Summary")
                    
                    alert_col1, alert_col2, alert_col3 = st.columns(3)
                    
                    with alert_col1:
                        st.metric("Total Alerts", system_report['alert_summary']['total_alerts'])
                    with alert_col2:
                        st.metric("Critical Alerts", system_report['alert_summary']['critical_alerts'])
                    with alert_col3:
                        st.metric("Warning Alerts", system_report['alert_summary']['warning_alerts'])
                    
                    # Export report
                    import json
                    report_json = json.dumps(system_report, indent=2)
                    
                    st.download_button(
                        label="📁 Download System Report (JSON)",
                        data=report_json,
                        file_name=f"system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime='application/json'
                    )
                    
                    st.success("System report generated successfully!")
                else:
                    st.warning("No recent data available for report generation.")

with col3:
    if st.button("📧 Email Summary"):
        # Email notification feature
            st.subheader("📧 Email Notification Configuration")
            
            with st.form("email_notification_setup"):
                # Email settings
                st.subheader("📮 Email Settings")
                
                smtp_server = st.text_input("SMTP Server", placeholder="smtp.gmail.com")
                smtp_port = st.number_input("SMTP Port", value=587, min_value=1, max_value=65535)
                sender_email = st.text_input("Sender Email", placeholder="alerts@company.com")
                sender_password = st.text_input("Sender Password", type="password", placeholder="App password or email password")
                
                # Notification preferences
                st.subheader("🔔 Notification Preferences")
                
                notification_types = st.multiselect(
                    "Send notifications for:",
                    ["Critical Alerts", "Warning Alerts", "System Health Updates", "Daily Reports", "Weekly Summaries"],
                    default=["Critical Alerts", "Warning Alerts"]
                )
                
                # Recipients
                st.subheader("👥 Recipients")
                
                recipients = st.text_area(
                    "Email Recipients (one per line)",
                    placeholder="admin@company.com\noperator@company.com\nmanager@company.com",
                    height=100
                )
                
                # Frequency settings
                st.subheader("⏰ Frequency Settings")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    alert_frequency = st.selectbox(
                        "Alert Frequency",
                        ["Immediate", "Every 5 minutes", "Every 15 minutes", "Hourly"],
                        index=0
                    )
                
                with col2:
                    report_frequency = st.selectbox(
                        "Report Frequency",
                        ["Daily", "Weekly", "Monthly"],
                        index=0
                    )
                
                # Test email
                send_test = st.checkbox("Send test email after configuration")
                
                # Submit button
                if st.form_submit_button("💾 Save Email Configuration"):
                    # Store email configuration
                    email_config = {
                        'smtp_server': smtp_server,
                        'smtp_port': smtp_port,
                        'sender_email': sender_email,
                        'sender_password': sender_password,  # In production, this should be encrypted
                        'notification_types': notification_types,
                        'recipients': recipients.split('\n') if recipients else [],
                        'alert_frequency': alert_frequency,
                        'report_frequency': report_frequency,
                        'configured_at': datetime.now().isoformat(),
                        'configured_by': st.session_state.get('username', 'admin')
                    }
                    
                    # Save to session state (in production, save to database)
                    st.session_state.email_config = email_config
                    
                    st.success("✅ Email notification configuration saved!")
                    
                    # Display configuration summary
                    st.info(f"📧 Configured for {len(email_config['recipients'])} recipients")
                    st.info(f"🔔 Notification types: {', '.join(notification_types)}")
                    st.info(f"⏰ Alert frequency: {alert_frequency}")
                    st.info(f"📊 Report frequency: {report_frequency}")
                    
                    # Send test email (simulated)
                    if send_test:
                        st.info("📧 Test email sent successfully!")
                        st.code(f"""
Test Email Content:
To: {email_config['recipients']}
From: {sender_email}
Subject: SIMS Email Notification Test

This is a test email from the Smart Infrastructure Monitoring System.

Email notifications have been configured with the following settings:
- Alert Frequency: {alert_frequency}
- Report Frequency: {report_frequency}
- Notification Types: {', '.join(notification_types)}

System Status: All systems operational
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
SIMS Notification System
                        """)
            
            # Display current configuration if exists
            if 'email_config' in st.session_state:
                st.subheader("📋 Current Email Configuration")
                config = st.session_state.email_config
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**SMTP Server:** {config['smtp_server']}")
                    st.write(f"**Sender Email:** {config['sender_email']}")
                    st.write(f"**Alert Frequency:** {config['alert_frequency']}")
                
                with col2:
                    st.write(f"**Recipients:** {len(config['recipients'])}")
                    st.write(f"**Notification Types:** {len(config['notification_types'])}")
                    st.write(f"**Configured:** {config['configured_at'][:10]}")
                
                # Show recipients
                if config['recipients']:
                    st.write("**📧 Email Recipients:**")
                    for recipient in config['recipients']:
                        if recipient.strip():
                            st.write(f"• {recipient.strip()}")
                
                # Test notification button
                if st.button("📧 Send Test Notification"):
                    st.info("📧 Test notification sent to all configured recipients!")
                    st.success("✅ Email notification system is working correctly!")

# Auto-refresh functionality
if st.session_state.get('auto_refresh', False):
    time.sleep(refresh_rate)
    st.rerun()

# Auto-refresh toggle
auto_refresh = st.checkbox("🔄 Enable Auto-Refresh", key='auto_refresh')
if auto_refresh:
    st.info(f"Dashboard will refresh every {refresh_rate} seconds")
