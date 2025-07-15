import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import psutil
import time
from utils.auth import check_authentication
from utils.data_generator import generate_real_time_data

# Authentication check
if not check_authentication():
    st.stop()

st.title("⚡ System Health")
st.markdown("Monitor pipeline status, performance metrics, and system infrastructure")

# System health overview
col1, col2, col3, col4 = st.columns(4)

# Simulate system metrics
cpu_usage = np.random.uniform(15, 35)
memory_usage = np.random.uniform(45, 75)
disk_usage = np.random.uniform(25, 60)
network_latency = np.random.uniform(5, 25)

with col1:
    st.metric("CPU Usage", f"{cpu_usage:.1f}%", f"{cpu_usage - 25:.1f}%")

with col2:
    st.metric("Memory Usage", f"{memory_usage:.1f}%", f"{memory_usage - 60:.1f}%")

with col3:
    st.metric("Disk Usage", f"{disk_usage:.1f}%", f"{disk_usage - 40:.1f}%")

with col4:
    st.metric("Network Latency", f"{network_latency:.1f}ms", f"{network_latency - 15:.1f}ms")

st.markdown("---")

# Main system health tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🖥️ Infrastructure", 
    "🔄 Data Pipeline", 
    "📊 Performance", 
    "🚨 Alerts & Logs", 
    "🔧 Maintenance"
])

with tab1:
    st.markdown("### Infrastructure Monitoring")
    
    # System components status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**System Components**")
        
        components = pd.DataFrame({
            'Component': ['Web Server', 'Database', 'Message Queue', 'ML Service', 'File Storage', 'Load Balancer'],
            'Status': ['Healthy', 'Healthy', 'Warning', 'Healthy', 'Healthy', 'Healthy'],
            'CPU (%)': [12.3, 23.7, 45.2, 67.8, 8.9, 15.4],
            'Memory (%)': [34.5, 56.2, 78.9, 82.1, 23.4, 29.7],
            'Uptime': ['15d 8h', '15d 8h', '12d 4h', '15d 8h', '15d 8h', '15d 8h']
        })
        
        # Color code status
        def color_status(val):
            if val == 'Healthy':
                return 'background-color: #90EE90'
            elif val == 'Warning':
                return 'background-color: #FFE4B5'
            else:
                return 'background-color: #FFB6C1'
        
        st.dataframe(
            components.style.applymap(color_status, subset=['Status']),
            use_container_width=True
        )
    
    with col2:
        # Component resource usage
        fig_resources = make_subplots(
            rows=2, cols=1,
            subplot_titles=('CPU Usage by Component', 'Memory Usage by Component'),
            vertical_spacing=0.1
        )
        
        fig_resources.add_trace(
            go.Bar(x=components['Component'], y=components['CPU (%)'], name='CPU'),
            row=1, col=1
        )
        
        fig_resources.add_trace(
            go.Bar(x=components['Component'], y=components['Memory (%)'], name='Memory'),
            row=2, col=1
        )
        
        fig_resources.update_layout(height=500, showlegend=False)
        fig_resources.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig_resources, use_container_width=True)
    
    # Network monitoring
    st.markdown("### Network Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Network throughput over time
        time_range = pd.date_range(start=datetime.now() - timedelta(hours=1), end=datetime.now(), freq='5T')
        network_data = pd.DataFrame({
            'Timestamp': time_range,
            'Inbound (Mbps)': np.random.uniform(50, 150, len(time_range)),
            'Outbound (Mbps)': np.random.uniform(30, 100, len(time_range))
        })
        
        fig_network = px.line(
            network_data.melt(id_vars=['Timestamp'], var_name='Direction', value_name='Throughput'),
            x='Timestamp',
            y='Throughput',
            color='Direction',
            title="Network Throughput"
        )
        st.plotly_chart(fig_network, use_container_width=True)
    
    with col2:
        # Connection status
        connections = pd.DataFrame({
            'Service': ['Database', 'External API', 'Message Queue', 'File Storage', 'ML Service'],
            'Status': ['Connected', 'Connected', 'Timeout', 'Connected', 'Connected'],
            'Response Time (ms)': [12, 145, 5000, 23, 234],
            'Availability (%)': [99.9, 98.5, 85.2, 99.7, 97.8]
        })
        
        st.dataframe(connections, use_container_width=True)
        
        # Response time chart
        fig_response = px.bar(
            connections,
            x='Service',
            y='Response Time (ms)',
            title="Service Response Times",
            color='Response Time (ms)',
            color_continuous_scale='RdYlGn_r'
        )
        fig_response.update_layout(height=300)
        st.plotly_chart(fig_response, use_container_width=True)

with tab2:
    st.markdown("### Data Pipeline Health")
    
    # Pipeline overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Pipelines", "8", "→ 0")
    
    with col2:
        st.metric("Records Processed", "2.4M", "↑ 125K")
    
    with col3:
        st.metric("Pipeline Success Rate", "98.7%", "↑ 0.3%")
    
    with col4:
        st.metric("Avg Processing Time", "1.2s", "↓ 0.1s")
    
    # Pipeline status
    st.markdown("### Pipeline Status")
    
    pipeline_data = pd.DataFrame({
        'Pipeline': ['Sensor Data Ingestion', 'Data Validation', 'Data Transformation', 'ML Processing', 'Data Export', 'Backup Process'],
        'Status': ['Running', 'Running', 'Running', 'Running', 'Completed', 'Running'],
        'Last Run': ['2 mins ago', '1 min ago', '3 mins ago', '5 mins ago', '1 hour ago', '30 mins ago'],
        'Duration': ['45s', '12s', '67s', '234s', '23s', '456s'],
        'Records': [15234, 15234, 15201, 15201, 15201, 15201],
        'Success Rate (%)': [99.8, 100.0, 99.7, 98.9, 100.0, 99.9]
    })
    
    st.dataframe(pipeline_data, use_container_width=True)
    
    # Pipeline performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Processing time trends
        time_data = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
        processing_times = pd.DataFrame({
            'Hour': time_data,
            'Ingestion': np.random.uniform(40, 60, len(time_data)),
            'Validation': np.random.uniform(10, 20, len(time_data)),
            'Transformation': np.random.uniform(50, 80, len(time_data)),
            'ML Processing': np.random.uniform(200, 300, len(time_data))
        })
        
        fig_processing = px.line(
            processing_times.melt(id_vars=['Hour'], var_name='Pipeline', value_name='Time (s)'),
            x='Hour',
            y='Time (s)',
            color='Pipeline',
            title="Pipeline Processing Times (24h)"
        )
        st.plotly_chart(fig_processing, use_container_width=True)
    
    with col2:
        # Data volume processing
        volume_data = pd.DataFrame({
            'Pipeline': ['Ingestion', 'Validation', 'Transform', 'ML Process', 'Export'],
            'Records/Hour': [125000, 124500, 124200, 123800, 123800],
            'Data Size (GB/Hour)': [1.2, 1.19, 1.18, 1.17, 1.17]
        })
        
        fig_volume = px.bar(
            volume_data,
            x='Pipeline',
            y='Records/Hour',
            title="Data Volume Processing",
            color='Records/Hour',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_volume, use_container_width=True)
    
    # Data quality metrics
    st.markdown("### Data Quality Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Data completeness
        completeness_data = pd.DataFrame({
            'Dataset': ['Sensor Readings', 'Infrastructure', 'Maintenance', 'User Activities'],
            'Completeness (%)': [98.7, 99.2, 96.5, 94.8]
        })
        
        fig_completeness = px.bar(
            completeness_data,
            x='Dataset',
            y='Completeness (%)',
            title="Data Completeness by Dataset",
            color='Completeness (%)',
            color_continuous_scale='RdYlGn'
        )
        fig_completeness.update_layout(height=300)
        st.plotly_chart(fig_completeness, use_container_width=True)
    
    with col2:
        # Error rates
        error_data = pd.DataFrame({
            'Error Type': ['Validation', 'Format', 'Missing', 'Duplicate', 'Range'],
            'Count': [45, 23, 67, 12, 34],
            'Rate (%)': [0.3, 0.15, 0.45, 0.08, 0.23]
        })
        
        fig_errors = px.pie(
            error_data,
            values='Count',
            names='Error Type',
            title="Error Distribution"
        )
        fig_errors.update_layout(height=300)
        st.plotly_chart(fig_errors, use_container_width=True)
    
    with col3:
        # Processing bottlenecks
        bottleneck_data = pd.DataFrame({
            'Stage': ['Input Queue', 'Validation', 'Transform', 'ML Model', 'Output'],
            'Queue Size': [234, 45, 67, 123, 23],
            'Avg Wait (s)': [2.3, 0.8, 1.2, 5.6, 0.5]
        })
        
        fig_bottleneck = px.bar(
            bottleneck_data,
            x='Stage',
            y='Avg Wait (s)',
            title="Processing Bottlenecks",
            color='Avg Wait (s)',
            color_continuous_scale='Reds'
        )
        fig_bottleneck.update_layout(height=300)
        st.plotly_chart(fig_bottleneck, use_container_width=True)

with tab3:
    st.markdown("### Performance Monitoring")
    
    # Performance KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Requests/Second", "1,247", "↑ 56")
    
    with col2:
        st.metric("Response Time", "187ms", "↓ 23ms")
    
    with col3:
        st.metric("Error Rate", "0.12%", "↓ 0.05%")
    
    with col4:
        st.metric("Throughput", "2.3 GB/h", "↑ 0.2 GB/h")
    
    # Performance trends
    st.markdown("### Performance Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time over time
        perf_time = pd.date_range(start=datetime.now() - timedelta(hours=6), end=datetime.now(), freq='15T')
        response_data = pd.DataFrame({
            'Time': perf_time,
            'Response Time (ms)': np.random.uniform(150, 250, len(perf_time)),
            'Requests/Min': np.random.uniform(800, 1500, len(perf_time))
        })
        
        fig_response_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_response_trend.add_trace(
            go.Scatter(x=response_data['Time'], y=response_data['Response Time (ms)'], name="Response Time"),
            secondary_y=False,
        )
        
        fig_response_trend.add_trace(
            go.Scatter(x=response_data['Time'], y=response_data['Requests/Min'], name="Requests/Min"),
            secondary_y=True,
        )
        
        fig_response_trend.update_xaxes(title_text="Time")
        fig_response_trend.update_yaxes(title_text="Response Time (ms)", secondary_y=False)
        fig_response_trend.update_yaxes(title_text="Requests/Min", secondary_y=True)
        fig_response_trend.update_layout(title_text="Performance Trends")
        
        st.plotly_chart(fig_response_trend, use_container_width=True)
    
    with col2:
        # Error rate trends
        error_trend_data = pd.DataFrame({
            'Time': perf_time,
            'Error Rate (%)': np.random.uniform(0.05, 0.3, len(perf_time)),
            'Success Rate (%)': 100 - np.random.uniform(0.05, 0.3, len(perf_time))
        })
        
        fig_error_trend = px.line(
            error_trend_data.melt(id_vars=['Time'], var_name='Metric', value_name='Percentage'),
            x='Time',
            y='Percentage',
            color='Metric',
            title="Success vs Error Rates"
        )
        st.plotly_chart(fig_error_trend, use_container_width=True)
    
    # Resource utilization
    st.markdown("### Resource Utilization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU and Memory trends
        resource_data = pd.DataFrame({
            'Time': perf_time,
            'CPU (%)': np.random.uniform(20, 40, len(perf_time)),
            'Memory (%)': np.random.uniform(50, 70, len(perf_time)),
            'Disk I/O (MB/s)': np.random.uniform(10, 50, len(perf_time))
        })
        
        fig_resources_trend = px.line(
            resource_data.melt(id_vars=['Time'], var_name='Resource', value_name='Usage'),
            x='Time',
            y='Usage',
            color='Resource',
            title="Resource Utilization Trends"
        )
        st.plotly_chart(fig_resources_trend, use_container_width=True)
    
    with col2:
        # Database performance
        db_metrics = pd.DataFrame({
            'Metric': ['Connections', 'Queries/Sec', 'Cache Hit Rate (%)', 'Lock Waits', 'Deadlocks'],
            'Current': [145, 2340, 96.7, 12, 0],
            'Baseline': [120, 2200, 95.5, 15, 1],
            'Status': ['Normal', 'High', 'Good', 'Normal', 'Good']
        })
        
        st.dataframe(db_metrics, use_container_width=True)
        
        # Query performance
        fig_db_perf = px.bar(
            db_metrics.head(3),
            x='Metric',
            y='Current',
            title="Database Performance Metrics",
            color='Current',
            color_continuous_scale='Blues'
        )
        fig_db_perf.update_layout(height=300)
        st.plotly_chart(fig_db_perf, use_container_width=True)

with tab4:
    st.markdown("### Alerts & System Logs")
    
    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Alerts", "3", "↓ 2")
    
    with col2:
        st.metric("Critical Alerts", "0", "↓ 1")
    
    with col3:
        st.metric("Warnings", "3", "→ 0")
    
    with col4:
        st.metric("Info Alerts", "12", "↑ 4")
    
    # Recent alerts
    st.markdown("### Recent Alerts")
    
    alerts_data = pd.DataFrame({
        'Timestamp': [
            datetime.now() - timedelta(minutes=5),
            datetime.now() - timedelta(minutes=15),
            datetime.now() - timedelta(minutes=30),
            datetime.now() - timedelta(hours=1),
            datetime.now() - timedelta(hours=2)
        ],
        'Severity': ['Warning', 'Info', 'Warning', 'Info', 'Critical'],
        'Component': ['Data Pipeline', 'Web Server', 'Database', 'ML Service', 'Network'],
        'Message': [
            'Processing queue size exceeding threshold',
            'High memory usage detected',
            'Slow query performance',
            'Model prediction accuracy below baseline',
            'Network latency spike detected'
        ],
        'Status': ['Active', 'Resolved', 'Active', 'Resolved', 'Resolved']
    })
    
    # Color code alerts by severity
    def color_severity(val):
        if val == 'Critical':
            return 'background-color: #FFB6C1'
        elif val == 'Warning':
            return 'background-color: #FFE4B5'
        else:
            return 'background-color: #E0E0E0'
    
    st.dataframe(
        alerts_data.style.applymap(color_severity, subset=['Severity']),
        use_container_width=True
    )
    
    # System logs
    st.markdown("### System Logs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        log_level = st.selectbox("Log Level", ["All", "Error", "Warning", "Info", "Debug"])
        time_filter = st.selectbox("Time Range", ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours"])
    
    with col2:
        component_filter = st.multiselect(
            "Components",
            ["Web Server", "Database", "Data Pipeline", "ML Service", "Network"],
            default=["Web Server", "Database", "Data Pipeline"]
        )
    
    # Sample log entries
    log_entries = pd.DataFrame({
        'Timestamp': [
            datetime.now() - timedelta(seconds=30),
            datetime.now() - timedelta(minutes=1),
            datetime.now() - timedelta(minutes=2),
            datetime.now() - timedelta(minutes=3),
            datetime.now() - timedelta(minutes=5)
        ],
        'Level': ['INFO', 'WARNING', 'INFO', 'ERROR', 'INFO'],
        'Component': ['Data Pipeline', 'Database', 'Web Server', 'ML Service', 'Data Pipeline'],
        'Message': [
            'Data processing completed successfully - 15,234 records',
            'Database connection pool nearly exhausted - 95% utilization',
            'User authentication successful - user: jane.smith',
            'ML model prediction failed - invalid input data format',
            'Starting data validation process'
        ]
    })
    
    st.dataframe(log_entries, use_container_width=True)
    
    # Log analysis
    st.markdown("### Log Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Log level distribution
        log_level_counts = log_entries['Level'].value_counts()
        fig_log_levels = px.pie(
            values=log_level_counts.values,
            names=log_level_counts.index,
            title="Log Level Distribution",
            color_discrete_map={'ERROR': 'red', 'WARNING': 'orange', 'INFO': 'blue', 'DEBUG': 'green'}
        )
        st.plotly_chart(fig_log_levels, use_container_width=True)
    
    with col2:
        # Component activity
        component_counts = log_entries['Component'].value_counts()
        fig_components = px.bar(
            x=component_counts.index,
            y=component_counts.values,
            title="Activity by Component",
            color=component_counts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_components, use_container_width=True)

with tab5:
    st.markdown("### System Maintenance")
    
    # Maintenance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Scheduled Tasks", "12", "→ 0")
    
    with col2:
        st.metric("Last Backup", "2h ago", "✅")
    
    with col3:
        st.metric("System Uptime", "15d 8h", "↑")
    
    with col4:
        st.metric("Health Score", "94.2%", "↑ 1.1%")
    
    # Scheduled maintenance
    st.markdown("### Scheduled Maintenance")
    
    maintenance_data = pd.DataFrame({
        'Task': ['Database Backup', 'Log Rotation', 'Cache Cleanup', 'Model Retraining', 'System Update'],
        'Type': ['Backup', 'Cleanup', 'Cleanup', 'ML', 'Update'],
        'Schedule': ['Daily 2:00 AM', 'Weekly Sunday', 'Daily 1:00 AM', 'Weekly Friday', 'Monthly'],
        'Last Run': ['2h ago', '2d ago', '3h ago', '3d ago', '15d ago'],
        'Next Run': ['22h', '5d', '21h', '4d', '15d'],
        'Status': ['Success', 'Success', 'Success', 'Success', 'Pending']
    })
    
    st.dataframe(maintenance_data, use_container_width=True)
    
    # System diagnostics
    st.markdown("### System Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Health Checks**")
        
        health_checks = pd.DataFrame({
            'Check': ['Database Connectivity', 'API Endpoints', 'File System', 'Network Connectivity', 'Service Dependencies'],
            'Status': ['✅ Healthy', '✅ Healthy', '✅ Healthy', '⚠️ Warning', '✅ Healthy'],
            'Last Check': ['1 min ago', '1 min ago', '5 min ago', '1 min ago', '2 min ago'],
            'Response Time': ['12ms', '145ms', 'N/A', '245ms', '67ms']
        })
        
        st.dataframe(health_checks, use_container_width=True)
    
    with col2:
        # Performance optimization suggestions
        st.markdown("**Optimization Suggestions**")
        
        suggestions = [
            "Consider increasing memory allocation for ML service",
            "Database index optimization recommended",
            "Enable compression for large data transfers",
            "Review and clean up old log files",
            "Update SSL certificates expiring in 30 days"
        ]
        
        for i, suggestion in enumerate(suggestions, 1):
            st.info(f"{i}. {suggestion}")
    
    # Maintenance actions
    st.markdown("### Maintenance Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Run System Diagnostics"):
            st.info("System diagnostics initiated...")
            # Simulate diagnostic progress
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            st.success("✅ Diagnostics completed - No issues found")
    
    with col2:
        if st.button("🧹 Clean System Cache"):
            st.info("Cache cleanup initiated...")
            st.success("✅ Cache cleanup completed - 2.3 GB freed")
    
    with col3:
        if st.button("📊 Generate Health Report"):
            st.info("Generating comprehensive health report...")
            
            # Simulate report data
            health_report = {
                "report_id": "HEALTH-2023-07-15-001",
                "timestamp": datetime.now().isoformat(),
                "overall_health": "94.2%",
                "components_checked": 15,
                "issues_found": 1,
                "recommendations": 5
            }
            
            with st.expander("View Health Report Summary"):
                st.json(health_report)

# Real-time monitoring toggle
st.markdown("---")
auto_refresh = st.checkbox("🔄 Enable Real-time Monitoring (refresh every 30 seconds)")

if auto_refresh:
    st.info("Real-time monitoring enabled - Dashboard will refresh automatically")
    time.sleep(30)
    st.rerun()
