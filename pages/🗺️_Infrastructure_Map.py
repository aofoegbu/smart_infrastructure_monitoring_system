import streamlit as st
import pandas as pd
import folium
from folium import plugins
import plotly.express as px
import numpy as np
from streamlit_folium import st_folium
from utils.auth import check_authentication
from utils.data_generator import generate_real_time_data, load_sensor_metadata

# Authentication check
if not check_authentication():
    st.stop()

st.title("🗺️ Infrastructure Map")
st.markdown("Interactive visualization of sensor networks and infrastructure assets")

# Map controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    map_view = st.selectbox("Map View", ["Satellite", "Terrain", "Street", "Hybrid"])

with col2:
    sensor_filter = st.multiselect(
        "Sensor Types", 
        ["pressure", "flow", "temperature", "quality"], 
        default=["pressure", "flow", "temperature", "quality"]
    )

with col3:
    status_filter = st.multiselect(
        "Status Filter", 
        ["Online", "Offline", "Alert", "Maintenance"], 
        default=["Online", "Alert"]
    )

with col4:
    show_heatmap = st.checkbox("Show Heatmap", value=True)

st.markdown("---")

# Generate sensor data and metadata
sensor_data = generate_real_time_data(100)
sensor_metadata = load_sensor_metadata()

# Filter data based on selections
filtered_data = sensor_data[sensor_data['sensor_type'].isin(sensor_filter)]

# Create the map
col1, col2 = st.columns([3, 1])

with col1:
    # Initialize map
    center_lat = filtered_data['latitude'].mean()
    center_lon = filtered_data['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add different tile layers based on selection
    if map_view == "Satellite":
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
    elif map_view == "Terrain":
        folium.TileLayer(
            tiles='https://stamen-tiles-{s}.a.ssl.fastly.net/terrain/{z}/{x}/{y}{r}.png',
            attr='Map tiles by Stamen Design',
            name='Terrain',
            overlay=False,
            control=True
        ).add_to(m)
    
    # Color mapping for sensor types
    color_map = {
        'pressure': 'blue',
        'flow': 'green', 
        'temperature': 'red',
        'quality': 'purple'
    }
    
    # Add sensor markers
    for _, sensor in filtered_data.iterrows():
        # Determine marker color based on status
        if sensor['anomaly_score'] > 0.7:
            marker_color = 'red'
            status = 'Alert'
        elif sensor['anomaly_score'] > 0.5:
            marker_color = 'orange'
            status = 'Warning'
        else:
            marker_color = color_map.get(sensor['sensor_type'], 'blue')
            status = 'Normal'
        
        # Filter by status if specified
        if status_filter and status not in status_filter and 'Online' not in status_filter:
            continue
            
        # Create popup content
        popup_content = f"""
        <b>Sensor ID:</b> {sensor['sensor_id']}<br>
        <b>Type:</b> {sensor['sensor_type']}<br>
        <b>Status:</b> {status}<br>
        <b>Pressure:</b> {sensor['pressure']:.1f} PSI<br>
        <b>Flow Rate:</b> {sensor['flow_rate']:.1f} L/min<br>
        <b>Temperature:</b> {sensor['temperature']:.1f}°C<br>
        <b>Anomaly Score:</b> {sensor['anomaly_score']:.2f}<br>
        <b>Last Update:</b> {sensor['timestamp'].strftime('%H:%M:%S')}
        """
        
        # Add marker
        folium.CircleMarker(
            location=[sensor['latitude'], sensor['longitude']],
            radius=8 if status == 'Alert' else 5,
            popup=folium.Popup(popup_content, max_width=300),
            color='black',
            weight=1,
            fillColor=marker_color,
            fillOpacity=0.8,
            tooltip=f"Sensor {sensor['sensor_id']} ({sensor['sensor_type']})"
        ).add_to(m)
    
    # Add heatmap if requested
    if show_heatmap:
        heat_data = [[row['latitude'], row['longitude'], row['anomaly_score']] 
                    for _, row in filtered_data.iterrows()]
        
        if heat_data:
            plugins.HeatMap(
                heat_data,
                name='Anomaly Heatmap',
                radius=15,
                blur=10,
                max_zoom=1,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1.0: 'red'}
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display map
    map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])

with col2:
    st.markdown("### Map Legend")
    
    # Sensor type legend
    st.markdown("**Sensor Types:**")
    for sensor_type, color in color_map.items():
        if sensor_type in sensor_filter:
            st.markdown(f"🔵 {sensor_type.title()}", unsafe_allow_html=True)
    
    st.markdown("**Status Indicators:**")
    st.markdown("🔴 Alert (Anomaly > 0.7)")
    st.markdown("🟠 Warning (Anomaly > 0.5)")
    st.markdown("🔵 Normal")
    
    # Statistics
    st.markdown("### Map Statistics")
    total_sensors = len(filtered_data)
    alert_sensors = len(filtered_data[filtered_data['anomaly_score'] > 0.7])
    warning_sensors = len(filtered_data[filtered_data['anomaly_score'] > 0.5])
    normal_sensors = total_sensors - alert_sensors - warning_sensors
    
    st.metric("Total Sensors", total_sensors)
    st.metric("Alert Sensors", alert_sensors)
    st.metric("Warning Sensors", warning_sensors)
    st.metric("Normal Sensors", normal_sensors)

# Detailed sensor information
st.markdown("---")
st.markdown("### Sensor Details")

if map_data['last_object_clicked']:
    clicked_lat = map_data['last_object_clicked']['lat']
    clicked_lng = map_data['last_object_clicked']['lng']
    
    # Find the closest sensor to the clicked location
    distances = np.sqrt((filtered_data['latitude'] - clicked_lat)**2 + 
                       (filtered_data['longitude'] - clicked_lng)**2)
    closest_sensor_idx = distances.idxmin()
    closest_sensor = filtered_data.loc[closest_sensor_idx]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Sensor Information**")
        st.write(f"**ID:** {closest_sensor['sensor_id']}")
        st.write(f"**Type:** {closest_sensor['sensor_type']}")
        st.write(f"**Location:** {closest_sensor['latitude']:.4f}, {closest_sensor['longitude']:.4f}")
        st.write(f"**Last Update:** {closest_sensor['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    with col2:
        st.markdown("**Current Readings**")
        st.metric("Pressure", f"{closest_sensor['pressure']:.1f} PSI")
        st.metric("Flow Rate", f"{closest_sensor['flow_rate']:.1f} L/min")
        st.metric("Temperature", f"{closest_sensor['temperature']:.1f}°C")
    
    with col3:
        st.markdown("**Analysis**")
        st.metric("Anomaly Score", f"{closest_sensor['anomaly_score']:.2f}")
        if closest_sensor['anomaly_score'] > 0.7:
            st.error("🚨 High Anomaly Detected!")
        elif closest_sensor['anomaly_score'] > 0.5:
            st.warning("⚠️ Moderate Anomaly")
        else:
            st.success("✅ Normal Operation")

# Infrastructure zones analysis
st.markdown("---")
st.markdown("### Zone Analysis")

# Create zones based on geographical clustering
from sklearn.cluster import KMeans

if len(filtered_data) > 5:
    # Perform clustering to identify zones
    coords = filtered_data[['latitude', 'longitude']].values
    n_clusters = min(5, len(filtered_data) // 10)  # Reasonable number of clusters
    
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        filtered_data['zone'] = kmeans.fit_predict(coords)
        
        # Zone statistics
        zone_stats = filtered_data.groupby('zone').agg({
            'anomaly_score': ['mean', 'max', 'count'],
            'pressure': 'mean',
            'flow_rate': 'mean',
            'temperature': 'mean'
        }).round(2)
        
        zone_stats.columns = ['Avg_Anomaly', 'Max_Anomaly', 'Sensor_Count', 'Avg_Pressure', 'Avg_Flow', 'Avg_Temp']
        zone_stats = zone_stats.reset_index()
        zone_stats['zone'] = zone_stats['zone'].apply(lambda x: f"Zone {x+1}")
        
        st.dataframe(zone_stats, use_container_width=True)
        
        # Zone comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            fig_zone_anomaly = px.bar(
                zone_stats,
                x='zone',
                y='Avg_Anomaly',
                title="Average Anomaly Score by Zone",
                color='Avg_Anomaly',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig_zone_anomaly, use_container_width=True)
        
        with col2:
            fig_zone_sensors = px.bar(
                zone_stats,
                x='zone',
                y='Sensor_Count',
                title="Sensor Count by Zone",
                color='Sensor_Count',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_zone_sensors, use_container_width=True)

# Export map data
st.markdown("---")
st.markdown("### Export Options")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Export Sensor Data"):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"map_sensor_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🗺️ Export Map Image"):
        # Map image export functionality
        import base64
        import io
        from PIL import Image
        
        # Create a map export
        export_map = folium.Map(
            location=[base_lat, base_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Add all sensors to export map
        for _, sensor in sensors_df.iterrows():
            status_color = 'green' if sensor['status'] == 'Active' else 'red'
            folium.CircleMarker(
                location=[sensor['latitude'], sensor['longitude']],
                radius=8,
                popup=f"Sensor: {sensor['sensor_id']}<br>Type: {sensor['sensor_type']}<br>Status: {sensor['status']}",
                color=status_color,
                fill=True,
                fillOpacity=0.7
            ).add_to(export_map)
        
        # Save map as HTML and display download link
        map_html = export_map._repr_html_()
        b64_html = base64.b64encode(map_html.encode()).decode()
        
        st.download_button(
            label="📁 Download Map (HTML)",
            data=map_html,
            file_name=f"infrastructure_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime='text/html'
        )
        st.success("Map export ready for download!")

with col3:
    if st.button("📊 Generate Zone Report"):
        # Zone analysis report generation
        with st.spinner("Analyzing zones..."):
            # Analyze sensors by geographical zones
            from utils.data_generator import get_recent_sensor_data_from_db
            recent_data = get_recent_sensor_data_from_db(hours=24)
            
            if not recent_data.empty:
                # Create zone analysis
                zone_analysis = {}
                
                # Define zones based on latitude/longitude ranges
                zones = {
                    'North': {'lat_min': base_lat + 0.02, 'lat_max': base_lat + 0.1, 'lon_min': base_lon - 0.1, 'lon_max': base_lon + 0.1},
                    'South': {'lat_min': base_lat - 0.1, 'lat_max': base_lat - 0.02, 'lon_min': base_lon - 0.1, 'lon_max': base_lon + 0.1},
                    'East': {'lat_min': base_lat - 0.02, 'lat_max': base_lat + 0.02, 'lon_min': base_lon + 0.02, 'lon_max': base_lon + 0.1},
                    'West': {'lat_min': base_lat - 0.02, 'lat_max': base_lat + 0.02, 'lon_min': base_lon - 0.1, 'lon_max': base_lon - 0.02},
                    'Central': {'lat_min': base_lat - 0.02, 'lat_max': base_lat + 0.02, 'lon_min': base_lon - 0.02, 'lon_max': base_lon + 0.02}
                }
                
                for zone_name, zone_bounds in zones.items():
                    zone_sensors = sensors_df[
                        (sensors_df['latitude'] >= zone_bounds['lat_min']) &
                        (sensors_df['latitude'] <= zone_bounds['lat_max']) &
                        (sensors_df['longitude'] >= zone_bounds['lon_min']) &
                        (sensors_df['longitude'] <= zone_bounds['lon_max'])
                    ]
                    
                    if not zone_sensors.empty:
                        zone_data = recent_data[recent_data['sensor_id'].isin(zone_sensors['sensor_id'])]
                        
                        if not zone_data.empty:
                            zone_analysis[zone_name] = {
                                'sensor_count': len(zone_sensors),
                                'avg_pressure': zone_data['pressure'].mean(),
                                'avg_flow': zone_data['flow_rate'].mean(),
                                'avg_temperature': zone_data['temperature'].mean(),
                                'avg_quality': zone_data['quality_score'].mean(),
                                'anomaly_count': len(zone_data[zone_data['anomaly_score'] > 0.7])
                            }
                    
                    # Display zone analysis
                    st.subheader("📍 Zone Analysis Report")
                    
                    for zone_name, data in zone_analysis.items():
                        with st.expander(f"🗺️ {zone_name} Zone Analysis"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Sensors", data['sensor_count'])
                                st.metric("Avg Pressure", f"{data['avg_pressure']:.1f} PSI")
                            
                            with col2:
                                st.metric("Avg Flow", f"{data['avg_flow']:.1f} L/min")
                                st.metric("Avg Temperature", f"{data['avg_temperature']:.1f}°C")
                            
                            with col3:
                                st.metric("Avg Quality", f"{data['avg_quality']:.1f}")
                                st.metric("Anomalies", data['anomaly_count'])
                    
                    # Generate downloadable report
                    report_data = []
                    for zone_name, data in zone_analysis.items():
                        report_data.append({
                            'Zone': zone_name,
                            'Sensors': data['sensor_count'],
                            'Avg_Pressure_PSI': round(data['avg_pressure'], 2),
                            'Avg_Flow_L_min': round(data['avg_flow'], 2),
                            'Avg_Temperature_C': round(data['avg_temperature'], 2),
                            'Avg_Quality_Score': round(data['avg_quality'], 2),
                            'Anomaly_Count': data['anomaly_count']
                        })
                    
                    import csv
                    import io
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=report_data[0].keys())
                    writer.writeheader()
                    writer.writerows(report_data)
                    
                    st.download_button(
                        label="📊 Download Zone Analysis Report (CSV)",
                        data=output.getvalue(),
                        file_name=f"zone_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                    
                    st.success("Zone analysis report generated successfully!")
                else:
                    st.warning("No recent sensor data available for zone analysis.")
