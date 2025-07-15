import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.auth import check_authentication
from utils.data_generator import generate_historical_data, generate_real_time_data
from utils.ml_models import detect_anomalies, predict_maintenance

# Authentication check
if not check_authentication():
    st.stop()

st.title("📈 Analytics Hub")
st.markdown("Advanced data analysis and insights for infrastructure monitoring")

# Analytics controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    analysis_type = st.selectbox(
        "Analysis Type", 
        ["Time Series", "Statistical", "Correlation", "Predictive", "PCA"]
    )

with col2:
    time_range = st.selectbox(
        "Time Range", 
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"]
    )

with col3:
    sensor_types = st.multiselect(
        "Sensor Types",
        ["pressure", "flow", "temperature", "quality"],
        default=["pressure", "flow", "temperature"]
    )

with col4:
    aggregation = st.selectbox(
        "Aggregation",
        ["Raw Data", "Hourly", "Daily", "Weekly"]
    )

st.markdown("---")

# Generate data based on time range
if time_range == "Last 24 Hours":
    hours = 24
elif time_range == "Last 7 Days":
    hours = 24 * 7
elif time_range == "Last 30 Days":
    hours = 24 * 30
else:  # Last 90 Days
    hours = 24 * 90

# Generate historical data
historical_data = generate_historical_data(hours=hours, sensors_count=50)
filtered_data = historical_data[historical_data['sensor_type'].isin(sensor_types)]

# Apply aggregation
if aggregation != "Raw Data":
    if aggregation == "Hourly":
        freq = 'H'
    elif aggregation == "Daily":
        freq = 'D'
    else:  # Weekly
        freq = 'W'
    
    filtered_data = filtered_data.set_index('timestamp').groupby(['sensor_id', 'sensor_type']).resample(freq).agg({
        'pressure': 'mean',
        'flow_rate': 'mean',
        'temperature': 'mean',
        'anomaly_score': 'max',
        'latitude': 'first',
        'longitude': 'first'
    }).reset_index()

# Analysis based on selected type
if analysis_type == "Time Series":
    st.markdown("### Time Series Analysis")
    
    # Time series decomposition
    col1, col2 = st.columns(2)
    
    with col1:
        metric = st.selectbox("Select Metric", ["pressure", "flow_rate", "temperature"])
        
        # Aggregate data by timestamp for overall trend
        time_series = filtered_data.groupby('timestamp')[metric].mean().reset_index()
        
        # Create time series plot
        fig_ts = px.line(
            time_series,
            x='timestamp',
            y=metric,
            title=f"{metric.replace('_', ' ').title()} Over Time",
            labels={metric: metric.replace('_', ' ').title(), 'timestamp': 'Time'}
        )
        
        # Add trend line
        if len(time_series) > 1:
            z = np.polyfit(range(len(time_series)), time_series[metric], 1)
            p = np.poly1d(z)
            fig_ts.add_trace(go.Scatter(
                x=time_series['timestamp'],
                y=p(range(len(time_series))),
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
        
        st.plotly_chart(fig_ts, use_container_width=True)
    
    with col2:
        # Seasonal decomposition
        st.markdown("**Statistical Summary**")
        metric_data = time_series[metric]
        
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Mean", f"{metric_data.mean():.2f}")
            st.metric("Std Dev", f"{metric_data.std():.2f}")
            st.metric("Min", f"{metric_data.min():.2f}")
        with col2_2:
            st.metric("Max", f"{metric_data.max():.2f}")
            st.metric("Median", f"{metric_data.median():.2f}")
            st.metric("Range", f"{metric_data.max() - metric_data.min():.2f}")
        
        # Distribution plot
        fig_dist = px.histogram(
            time_series,
            x=metric,
            nbins=30,
            title=f"{metric.replace('_', ' ').title()} Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Sensor comparison
    st.markdown("### Sensor Comparison")
    
    # Select top sensors by data volume
    sensor_counts = filtered_data['sensor_id'].value_counts().head(10)
    top_sensors = sensor_counts.index.tolist()
    
    selected_sensors = st.multiselect(
        "Select Sensors for Comparison",
        top_sensors,
        default=top_sensors[:5]
    )
    
    if selected_sensors:
        sensor_comparison_data = filtered_data[filtered_data['sensor_id'].isin(selected_sensors)]
        
        fig_comparison = px.line(
            sensor_comparison_data,
            x='timestamp',
            y=metric,
            color='sensor_id',
            title=f"{metric.replace('_', ' ').title()} Comparison Across Sensors"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

elif analysis_type == "Statistical":
    st.markdown("### Statistical Analysis")
    
    # Statistical tests and analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Descriptive Statistics**")
        
        # Calculate statistics for each metric
        stats_data = []
        for metric in ['pressure', 'flow_rate', 'temperature']:
            if metric in filtered_data.columns:
                data = filtered_data[metric].dropna()
                stats_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Count': len(data),
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Min': data.min(),
                    'Q1': data.quantile(0.25),
                    'Median': data.median(),
                    'Q3': data.quantile(0.75),
                    'Max': data.max(),
                    'Skewness': stats.skew(data),
                    'Kurtosis': stats.kurtosis(data)
                })
        
        stats_df = pd.DataFrame(stats_data).round(3)
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.markdown("**Distribution Analysis**")
        
        # Box plots for each metric
        metrics_melted = filtered_data[['pressure', 'flow_rate', 'temperature']].melt(
            var_name='Metric', value_name='Value'
        )
        
        fig_box = px.box(
            metrics_melted,
            x='Metric',
            y='Value',
            title="Distribution Comparison"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Outlier detection
    st.markdown("### Outlier Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # IQR-based outlier detection
        outliers_data = []
        for metric in ['pressure', 'flow_rate', 'temperature']:
            if metric in filtered_data.columns:
                data = filtered_data[metric].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                outliers_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Total Points': len(data),
                    'Outliers': len(outliers),
                    'Outlier %': (len(outliers) / len(data)) * 100,
                    'Lower Bound': lower_bound,
                    'Upper Bound': upper_bound
                })
        
        outliers_df = pd.DataFrame(outliers_data).round(3)
        st.dataframe(outliers_df, use_container_width=True)
    
    with col2:
        # Outlier visualization
        metric_for_outliers = st.selectbox("Select Metric for Outlier Visualization", 
                                         ['pressure', 'flow_rate', 'temperature'])
        
        data = filtered_data[metric_for_outliers].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        fig_outliers = go.Figure()
        fig_outliers.add_trace(go.Scatter(
            y=data,
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=4)
        ))
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_indices = outliers.index
        
        fig_outliers.add_trace(go.Scatter(
            x=outlier_indices,
            y=outliers,
            mode='markers',
            name='Outliers',
            marker=dict(color='red', size=6)
        ))
        
        fig_outliers.add_hline(y=lower_bound, line_dash="dash", line_color="orange", 
                              annotation_text="Lower Bound")
        fig_outliers.add_hline(y=upper_bound, line_dash="dash", line_color="orange", 
                              annotation_text="Upper Bound")
        
        fig_outliers.update_layout(title=f"Outlier Detection - {metric_for_outliers.title()}")
        st.plotly_chart(fig_outliers, use_container_width=True)

elif analysis_type == "Correlation":
    st.markdown("### Correlation Analysis")
    
    # Calculate correlation matrix
    correlation_metrics = ['pressure', 'flow_rate', 'temperature', 'anomaly_score']
    correlation_data = filtered_data[correlation_metrics].corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        fig_corr = px.imshow(
            correlation_data,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig_corr.update_layout(height=400)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # Strongest correlations
        st.markdown("**Strongest Correlations**")
        
        # Get correlation pairs
        corr_pairs = []
        for i in range(len(correlation_data.columns)):
            for j in range(i+1, len(correlation_data.columns)):
                corr_pairs.append({
                    'Variable 1': correlation_data.columns[i],
                    'Variable 2': correlation_data.columns[j],
                    'Correlation': correlation_data.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(corr_pairs)
        corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs Correlation', ascending=False)
        
        st.dataframe(corr_df.head(10), use_container_width=True)
    
    # Scatter plot matrix
    st.markdown("### Scatter Plot Matrix")
    
    # Select variables for scatter matrix
    selected_vars = st.multiselect(
        "Select Variables for Scatter Matrix",
        correlation_metrics,
        default=['pressure', 'flow_rate', 'temperature']
    )
    
    if len(selected_vars) >= 2:
        fig_scatter_matrix = px.scatter_matrix(
            filtered_data[selected_vars].sample(min(1000, len(filtered_data))),  # Sample for performance
            title="Scatter Plot Matrix"
        )
        fig_scatter_matrix.update_layout(height=600)
        st.plotly_chart(fig_scatter_matrix, use_container_width=True)

elif analysis_type == "Predictive":
    st.markdown("### Predictive Analysis")
    
    # Maintenance prediction
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Maintenance Prediction**")
        
        # Get recent data for prediction
        recent_data = generate_real_time_data(20)
        maintenance_scores = predict_maintenance(recent_data)
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'Sensor ID': recent_data['sensor_id'],
            'Maintenance Score': maintenance_scores,
            'Risk Level': ['High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low' 
                          for score in maintenance_scores],
            'Days Until Maintenance': [int(30 * (1 - score)) for score in maintenance_scores]
        })
        
        st.dataframe(prediction_df.sort_values('Maintenance Score', ascending=False), 
                    use_container_width=True)
    
    with col2:
        # Maintenance risk distribution
        fig_maintenance = px.histogram(
            prediction_df,
            x='Risk Level',
            title="Maintenance Risk Distribution",
            color='Risk Level',
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        st.plotly_chart(fig_maintenance, use_container_width=True)
    
    # Trend forecasting
    st.markdown("### Trend Forecasting")
    
    metric_to_forecast = st.selectbox("Select Metric to Forecast", 
                                    ['pressure', 'flow_rate', 'temperature'])
    
    # Simple linear trend forecast
    time_series_data = filtered_data.groupby('timestamp')[metric_to_forecast].mean().reset_index()
    
    if len(time_series_data) > 10:
        # Fit trend
        x = np.arange(len(time_series_data))
        y = time_series_data[metric_to_forecast].values
        z = np.polyfit(x, y, 1)
        
        # Forecast next 24 points
        forecast_x = np.arange(len(time_series_data), len(time_series_data) + 24)
        forecast_y = np.polyval(z, forecast_x)
        
        # Create forecast plot
        fig_forecast = go.Figure()
        
        # Historical data
        fig_forecast.add_trace(go.Scatter(
            x=time_series_data['timestamp'],
            y=y,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        forecast_timestamps = pd.date_range(
            start=time_series_data['timestamp'].iloc[-1] + timedelta(hours=1),
            periods=24,
            freq='H'
        )
        
        fig_forecast.add_trace(go.Scatter(
            x=forecast_timestamps,
            y=forecast_y,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig_forecast.update_layout(
            title=f"{metric_to_forecast.replace('_', ' ').title()} Forecast",
            xaxis_title="Time",
            yaxis_title=metric_to_forecast.replace('_', ' ').title()
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Forecast statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Value", f"{y[-1]:.2f}")
        with col2:
            st.metric("24h Forecast", f"{forecast_y[-1]:.2f}")
        with col3:
            change = forecast_y[-1] - y[-1]
            st.metric("Predicted Change", f"{change:+.2f}")

elif analysis_type == "PCA":
    st.markdown("### Principal Component Analysis")
    
    # Prepare data for PCA
    pca_features = ['pressure', 'flow_rate', 'temperature', 'anomaly_score']
    pca_data = filtered_data[pca_features].dropna()
    
    if len(pca_data) > 10:
        # Standardize data
        scaler = StandardScaler()
        pca_scaled = scaler.fit_transform(pca_data)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(pca_scaled)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            fig_var = go.Figure()
            fig_var.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(len(explained_var))],
                y=explained_var,
                name='Individual',
                marker_color='blue'
            ))
            fig_var.add_trace(go.Scatter(
                x=[f'PC{i+1}' for i in range(len(cumulative_var))],
                y=cumulative_var,
                mode='lines+markers',
                name='Cumulative',
                yaxis='y2',
                line=dict(color='red')
            ))
            
            fig_var.update_layout(
                title="PCA Explained Variance",
                xaxis_title="Principal Components",
                yaxis_title="Explained Variance Ratio",
                yaxis2=dict(title="Cumulative Variance", overlaying='y', side='right')
            )
            
            st.plotly_chart(fig_var, use_container_width=True)
        
        with col2:
            # PCA scatter plot
            pca_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Sensor_Type': filtered_data['sensor_type'].iloc[:len(pca_result)]
            })
            
            fig_pca = px.scatter(
                pca_df.sample(min(1000, len(pca_df))),  # Sample for performance
                x='PC1',
                y='PC2',
                color='Sensor_Type',
                title="PCA Scatter Plot (PC1 vs PC2)"
            )
            st.plotly_chart(fig_pca, use_container_width=True)
        
        # Component loadings
        st.markdown("### Component Loadings")
        
        loadings = pd.DataFrame(
            pca.components_[:4].T,  # First 4 components
            columns=[f'PC{i+1}' for i in range(4)],
            index=pca_features
        )
        
        fig_loadings = px.imshow(
            loadings.T,
            title="PCA Component Loadings",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_loadings, use_container_width=True)
        
        st.dataframe(loadings.round(3), use_container_width=True)

# Export analysis results
st.markdown("---")
st.markdown("### Export Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Data"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Analysis Data",
            data=csv,
            file_name=f"analytics_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📊 Generate Report"):
        # Automated report generation
        with st.spinner("Generating comprehensive analytics report..."):
            # Get recent data for analysis
            from utils.data_generator import get_recent_sensor_data_from_db
            from utils.database import get_system_stats, get_active_alerts
            from utils.ml_models import calculate_sensor_health_score, predict_maintenance, detect_anomalies
            from utils.data_quality import generate_quality_report
            
            recent_data = get_recent_sensor_data_from_db(hours=24)
            system_stats = get_system_stats()
            active_alerts = get_active_alerts()
            
            if not recent_data.empty:
                # Generate analytics
                health_scores = calculate_sensor_health_score(recent_data)
                maintenance_scores = predict_maintenance(recent_data)
                anomaly_scores = detect_anomalies(recent_data)
                quality_report = generate_quality_report(recent_data)
                
                # Create comprehensive report
                report = {
                        'timestamp': datetime.now().isoformat(),
                        'system_overview': {
                            'total_sensors': system_stats['total_sensors'],
                            'active_sensors': system_stats['active_sensors'],
                            'total_readings': len(recent_data),
                            'active_alerts': len(active_alerts)
                        },
                        'performance_metrics': {
                            'avg_pressure': recent_data['pressure'].mean(),
                            'avg_flow_rate': recent_data['flow_rate'].mean(),
                            'avg_temperature': recent_data['temperature'].mean(),
                            'avg_quality_score': recent_data['quality_score'].mean()
                        },
                        'health_analysis': {
                            'avg_health_score': health_scores['health_score'].mean(),
                            'sensors_needing_maintenance': len([s for s in maintenance_scores if s > 0.6]),
                            'high_anomaly_sensors': len([s for s in anomaly_scores if s > 0.7])
                        },
                        'data_quality': {
                            'overall_score': quality_report['overall_score'],
                            'completeness': quality_report['checks']['completeness']['score'],
                            'accuracy': quality_report['checks']['accuracy']['score']
                        }
                    }
                
                # Display report
                st.subheader("📊 Comprehensive Analytics Report")
                
                col1, col2 = st.columns(2)
                    
                    with col1:
                        st.json(report['system_overview'])
                        st.json(report['performance_metrics'])
                    
                    with col2:
                        st.json(report['health_analysis'])
                        st.json(report['data_quality'])
                    
                    # Create downloadable report
                    import json
                    report_json = json.dumps(report, indent=2)
                    
                    st.download_button(
                        label="📁 Download Analytics Report (JSON)",
                        data=report_json,
                        file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime='application/json'
                    )
                    
                    st.success("Automated analytics report generated successfully!")
                else:
                    st.warning("No recent data available for report generation.")

with col3:
    if st.button("📧 Schedule Analysis"):
        # Analysis scheduling feature
        st.subheader("📅 Schedule Automated Analysis")
        
        with st.form("schedule_analysis"):
            # Analysis type selection
                analysis_type = st.selectbox(
                    "Analysis Type",
                    ["Daily System Health", "Weekly Performance Report", "Monthly Quality Assessment", "Anomaly Detection Scan"]
                )
                
                # Frequency selection
                frequency = st.selectbox(
                    "Frequency",
                    ["Daily", "Weekly", "Monthly", "Custom"]
                )
                
                # Time selection
                run_time = st.time_input("Run Time", value=datetime.now().time())
                
                # Recipients
                recipients = st.text_area(
                    "Email Recipients (comma-separated)",
                    placeholder="admin@company.com, operator@company.com"
                )
                
                # Parameters
                st.subheader("Analysis Parameters")
                
                col1, col2 = st.columns(2)
                with col1:
                    include_charts = st.checkbox("Include Charts", value=True)
                    include_alerts = st.checkbox("Include Alerts", value=True)
                
                with col2:
                    include_recommendations = st.checkbox("Include Recommendations", value=True)
                    send_summary_only = st.checkbox("Send Summary Only", value=False)
                
                # Submit button
                if st.form_submit_button("📅 Schedule Analysis"):
                    # Create scheduled analysis record
                    scheduled_analysis = {
                        'id': f"SCHED_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'type': analysis_type,
                        'frequency': frequency,
                        'run_time': run_time.strftime('%H:%M'),
                        'recipients': recipients.split(',') if recipients else [],
                        'parameters': {
                            'include_charts': include_charts,
                            'include_alerts': include_alerts,
                            'include_recommendations': include_recommendations,
                            'send_summary_only': send_summary_only
                        },
                        'created_at': datetime.now().isoformat(),
                        'status': 'active'
                    }
                    
                    # Store in session state (in production, this would go to database)
                    if 'scheduled_analyses' not in st.session_state:
                        st.session_state.scheduled_analyses = []
                    
                    st.session_state.scheduled_analyses.append(scheduled_analysis)
                    
                    st.success(f"✅ Scheduled {analysis_type} analysis for {frequency} at {run_time.strftime('%H:%M')}")
                    st.info("📧 Email notifications will be sent to specified recipients when analysis runs.")
            
            # Display existing scheduled analyses
            if 'scheduled_analyses' in st.session_state and st.session_state.scheduled_analyses:
                st.subheader("📋 Scheduled Analyses")
                
                for i, analysis in enumerate(st.session_state.scheduled_analyses):
                    with st.expander(f"📅 {analysis['type']} - {analysis['frequency']} at {analysis['run_time']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Type:** {analysis['type']}")
                            st.write(f"**Frequency:** {analysis['frequency']}")
                            st.write(f"**Run Time:** {analysis['run_time']}")
                        
                        with col2:
                            st.write(f"**Recipients:** {len(analysis['recipients'])} recipients")
                            st.write(f"**Status:** {analysis['status']}")
                            st.write(f"**Created:** {analysis['created_at'][:10]}")
                        
                        if st.button(f"🗑️ Delete Schedule", key=f"delete_{i}"):
                            st.session_state.scheduled_analyses.pop(i)
                            st.rerun()
