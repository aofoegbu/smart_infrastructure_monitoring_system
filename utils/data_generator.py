import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
from .database import get_sensors, get_sensor_readings, store_sensor_reading, get_infrastructure_assets

def generate_real_time_data(num_sensors=50):
    """
    Generate realistic real-time sensor data for infrastructure monitoring
    """
    sensor_types = ['pressure', 'flow', 'temperature', 'quality']
    
    # Base coordinates for a city infrastructure network
    base_lat = 40.7128  # New York City coordinates
    base_lon = -74.0060
    
    data = []
    
    for i in range(num_sensors):
        sensor_id = f"SENSOR_{i+1:03d}"
        sensor_type = random.choice(sensor_types)
        
        # Generate realistic coordinates within a city area
        lat_offset = np.random.normal(0, 0.05)  # ~5km radius
        lon_offset = np.random.normal(0, 0.05)
        latitude = base_lat + lat_offset
        longitude = base_lon + lon_offset
        
        # Generate realistic sensor readings based on type
        if sensor_type == 'pressure':
            # Pressure in PSI (20-80 range with normal around 45)
            pressure = np.random.normal(45, 10)
            pressure = max(20, min(80, pressure))
            flow_rate = np.random.normal(25, 5)  # L/min
            temperature = np.random.normal(22, 3)  # Celsius
        elif sensor_type == 'flow':
            pressure = np.random.normal(50, 8)
            flow_rate = np.random.normal(30, 8)
            temperature = np.random.normal(20, 4)
        elif sensor_type == 'temperature':
            pressure = np.random.normal(40, 6)
            flow_rate = np.random.normal(20, 6)
            temperature = np.random.normal(25, 5)
        else:  # quality
            pressure = np.random.normal(42, 7)
            flow_rate = np.random.normal(22, 4)
            temperature = np.random.normal(21, 3)
        
        # Ensure positive values
        pressure = max(0, pressure)
        flow_rate = max(0, flow_rate)
        
        # Generate anomaly score (higher means more anomalous)
        # Most sensors should be normal (low anomaly score)
        if random.random() < 0.1:  # 10% chance of anomaly
            anomaly_score = np.random.uniform(0.6, 1.0)
        else:
            anomaly_score = np.random.uniform(0.0, 0.4)
        
        data.append({
            'sensor_id': sensor_id,
            'sensor_type': sensor_type,
            'timestamp': datetime.now() - timedelta(minutes=random.randint(0, 60)),
            'latitude': latitude,
            'longitude': longitude,
            'pressure': pressure,
            'flow_rate': flow_rate,
            'temperature': temperature,
            'anomaly_score': anomaly_score
        })
    
    return pd.DataFrame(data)

def generate_historical_data(hours=24, sensors_count=50):
    """
    Generate historical sensor data for time series analysis
    """
    sensor_types = ['pressure', 'flow', 'temperature', 'quality']
    base_lat = 40.7128
    base_lon = -74.0060
    
    # Create time series
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    time_points = pd.date_range(start=start_time, end=end_time, freq='15T')  # 15-minute intervals
    
    data = []
    
    # Generate sensor metadata
    sensors = []
    for i in range(sensors_count):
        sensors.append({
            'sensor_id': f"SENSOR_{i+1:03d}",
            'sensor_type': random.choice(sensor_types),
            'latitude': base_lat + np.random.normal(0, 0.05),
            'longitude': base_lon + np.random.normal(0, 0.05)
        })
    
    for sensor in sensors:
        # Generate base patterns for each sensor
        base_pressure = np.random.uniform(35, 55)
        base_flow = np.random.uniform(15, 35)
        base_temp = np.random.uniform(18, 26)
        
        for timestamp in time_points:
            # Add daily patterns (lower values at night)
            hour = timestamp.hour
            daily_factor = 0.8 + 0.4 * np.sin((hour - 6) * np.pi / 12)
            
            # Add some random noise
            noise_factor = np.random.normal(1.0, 0.1)
            
            pressure = base_pressure * daily_factor * noise_factor
            flow_rate = base_flow * daily_factor * noise_factor
            temperature = base_temp + np.random.normal(0, 2)
            
            # Ensure positive values
            pressure = max(0, pressure)
            flow_rate = max(0, flow_rate)
            
            # Generate anomaly score
            if random.random() < 0.05:  # 5% chance of anomaly in historical data
                anomaly_score = np.random.uniform(0.6, 1.0)
                # Anomalous readings
                if random.random() < 0.5:
                    pressure *= np.random.uniform(1.5, 2.0)
                else:
                    pressure *= np.random.uniform(0.3, 0.6)
            else:
                anomaly_score = np.random.uniform(0.0, 0.3)
            
            data.append({
                'sensor_id': sensor['sensor_id'],
                'sensor_type': sensor['sensor_type'],
                'timestamp': timestamp,
                'latitude': sensor['latitude'],
                'longitude': sensor['longitude'],
                'pressure': pressure,
                'flow_rate': flow_rate,
                'temperature': temperature,
                'anomaly_score': anomaly_score
            })
    
    return pd.DataFrame(data)

def load_sensor_metadata():
    """
    Load sensor metadata from database
    """
    try:
        # Try to load from database first
        sensors_df = get_sensors()
        if not sensors_df.empty:
            return sensors_df
        else:
            # Fallback to CSV file if database is empty
            if os.path.exists('data/sensors.csv'):
                return pd.read_csv('data/sensors.csv')
            else:
                # Generate if neither database nor file exists
                return generate_sensor_metadata()
    except Exception as e:
        print(f"Error loading sensor metadata: {e}")
        return generate_sensor_metadata()

def generate_sensor_metadata():
    """
    Generate sensor metadata
    """
    sensor_types = ['pressure', 'flow', 'temperature', 'quality']
    locations = [
        'Water Treatment Plant A', 'Distribution Center B', 'Pumping Station C',
        'Reservoir D', 'Main Pipeline E', 'Secondary Network F',
        'Industrial Zone G', 'Residential Area H', 'Commercial District I',
        'Emergency Backup J'
    ]
    
    metadata = []
    for i in range(50):
        metadata.append({
            'sensor_id': f"SENSOR_{i+1:03d}",
            'sensor_type': random.choice(sensor_types),
            'location': random.choice(locations),
            'installation_date': (datetime.now() - timedelta(days=random.randint(30, 1095))).strftime('%Y-%m-%d'),
            'maintenance_interval': random.choice([30, 60, 90, 180]),
            'manufacturer': random.choice(['SensorTech', 'AquaMonitor', 'FlowControl', 'TempSense']),
            'model': f"Model-{random.randint(100, 999)}",
            'status': random.choice(['Active', 'Active', 'Active', 'Maintenance']),
            'calibration_date': (datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(metadata)

def load_infrastructure_assets():
    """
    Load infrastructure assets data from database
    """
    try:
        # Try to load from database first
        assets_df = get_infrastructure_assets()
        if not assets_df.empty:
            return assets_df
        else:
            # Fallback to CSV file if database is empty
            if os.path.exists('data/infrastructure_assets.csv'):
                return pd.read_csv('data/infrastructure_assets.csv')
            else:
                # Generate if neither database nor file exists
                return generate_infrastructure_assets()
    except Exception as e:
        print(f"Error loading infrastructure assets: {e}")
        return generate_infrastructure_assets()

def generate_infrastructure_assets():
    """
    Generate infrastructure assets data
    """
    asset_types = ['Pipeline', 'Pump Station', 'Treatment Plant', 'Storage Tank', 'Valve', 'Meter']
    
    assets = []
    base_lat = 40.7128
    base_lon = -74.0060
    
    for i in range(25):
        asset_id = f"ASSET_{i+1:03d}"
        asset_type = random.choice(asset_types)
        
        assets.append({
            'asset_id': asset_id,
            'asset_type': asset_type,
            'name': f"{asset_type} {i+1}",
            'latitude': base_lat + np.random.normal(0, 0.08),
            'longitude': base_lon + np.random.normal(0, 0.08),
            'installation_year': random.randint(1995, 2020),
            'capacity': random.randint(100, 5000),
            'material': random.choice(['Steel', 'PVC', 'Cast Iron', 'Concrete']),
            'condition': random.choice(['Excellent', 'Good', 'Fair', 'Poor']),
            'last_inspection': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'next_maintenance': (datetime.now() + timedelta(days=random.randint(30, 180))).strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(assets)

def generate_and_store_real_time_data(num_sensors=10):
    """
    Generate real-time sensor data and store it to database
    """
    try:
        # Load sensor metadata from database
        sensors_metadata = load_sensor_metadata()
        
        if sensors_metadata.empty:
            print("No sensors found in database")
            return pd.DataFrame()
        
        # Select random sensors for this batch
        available_sensors = len(sensors_metadata)
        num_sensors = min(num_sensors, available_sensors)
        selected_sensors = sensors_metadata.sample(n=num_sensors) if num_sensors < available_sensors else sensors_metadata
        
        current_time = datetime.now()
        stored_count = 0
        
        for _, sensor in selected_sensors.iterrows():
            # Generate realistic values based on sensor type and time of day
            hour = current_time.hour
            base_values = {
                'pressure': 45 + 15 * np.sin(hour * np.pi / 12),
                'flow': 25 + 10 * np.sin(hour * np.pi / 12),
                'temperature': 20 + 8 * np.sin((hour - 6) * np.pi / 12),
                'quality': 8.5 + 1.5 * np.random.normal(0, 0.1)
            }
            
            # Add sensor-specific variations
            pressure = base_values['pressure'] + np.random.normal(0, 2)
            flow_rate = base_values['flow'] + np.random.normal(0, 3)
            temperature = base_values['temperature'] + np.random.normal(0, 1.5)
            quality_score = max(0, min(10, base_values['quality'] + np.random.normal(0, 0.5)))
            
            # Ensure realistic bounds
            pressure = max(0, pressure)
            flow_rate = max(0, flow_rate)
            
            # Generate anomaly detection scores
            anomaly_score = 0.0
            is_anomaly = False
            
            # 3% chance of anomaly
            if random.random() < 0.03:
                anomaly_score = np.random.uniform(0.7, 1.0)
                is_anomaly = True
                # Create anomalous readings
                if sensor['sensor_type'] == 'pressure':
                    pressure *= np.random.choice([0.3, 2.5])
                elif sensor['sensor_type'] == 'flow':
                    flow_rate *= np.random.choice([0.2, 3.0])
                elif sensor['sensor_type'] == 'temperature':
                    temperature += np.random.choice([-15, 25])
                elif sensor['sensor_type'] == 'quality':
                    quality_score *= np.random.uniform(0.3, 0.7)
            else:
                anomaly_score = np.random.uniform(0.0, 0.4)
            
            # Add some correlation between parameters for realism
            if pressure > 60:
                flow_rate *= 1.2
                temperature += 2
            elif pressure < 20:
                flow_rate *= 0.8
            
            reading_data = {
                'timestamp': current_time,
                'pressure': float(round(pressure, 2)),
                'flow_rate': float(round(flow_rate, 2)),
                'temperature': float(round(temperature, 2)),
                'quality_score': float(round(quality_score, 1)),
                'anomaly_score': float(round(anomaly_score, 3)),
                'is_anomaly': bool(is_anomaly)
            }
            
            # Store to database
            if store_sensor_reading(sensor['sensor_id'], reading_data):
                stored_count += 1
        
        print(f"Stored {stored_count} sensor readings to database")
        return True
        
    except Exception as e:
        print(f"Error generating and storing real-time data: {e}")
        return False

def get_recent_sensor_data_from_db(hours=1, limit=100):
    """
    Get recent sensor data from database
    """
    try:
        # Get sensor readings from database
        readings_df = get_sensor_readings(hours=hours)
        
        if readings_df.empty:
            return pd.DataFrame()
        
        # Get sensor metadata
        sensors_df = get_sensors()
        
        # Merge readings with sensor metadata
        if not sensors_df.empty:
            merged_df = readings_df.merge(
                sensors_df[['sensor_id', 'sensor_type', 'location', 'latitude', 'longitude', 'status', 'manufacturer', 'model']], 
                on='sensor_id', 
                how='left'
            )
            return merged_df.head(limit)
        
        return readings_df.head(limit)
        
    except Exception as e:
        print(f"Error getting recent sensor data: {e}")
        return pd.DataFrame()

def simulate_sensor_failure(sensor_data, failure_rate=0.02):
    """
    Simulate sensor failures in the data
    """
    failed_sensors = sensor_data.sample(frac=failure_rate).copy()
    failed_sensors['pressure'] = np.nan
    failed_sensors['flow_rate'] = np.nan
    failed_sensors['temperature'] = np.nan
    failed_sensors['anomaly_score'] = 1.0
    
    # Update original data
    sensor_data.update(failed_sensors)
    return sensor_data

def add_seasonal_patterns(data):
    """
    Add seasonal patterns to sensor data
    """
    data = data.copy()
    
    # Add seasonal temperature variations
    day_of_year = data['timestamp'].dt.dayofyear
    seasonal_temp_adjustment = 5 * np.sin(2 * np.pi * day_of_year / 365)
    data['temperature'] += seasonal_temp_adjustment
    
    # Add seasonal pressure variations (higher in winter due to increased demand)
    seasonal_pressure_adjustment = 3 * np.cos(2 * np.pi * day_of_year / 365)
    data['pressure'] += seasonal_pressure_adjustment
    
    return data

def generate_maintenance_data():
    """
    Generate maintenance records data
    """
    maintenance_types = ['Preventive', 'Corrective', 'Emergency', 'Calibration']
    
    records = []
    for i in range(100):
        record_id = f"MAINT_{i+1:03d}"
        sensor_id = f"SENSOR_{random.randint(1, 50):03d}"
        
        records.append({
            'record_id': record_id,
            'sensor_id': sensor_id,
            'maintenance_type': random.choice(maintenance_types),
            'date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
            'duration_hours': random.randint(1, 8),
            'cost': random.randint(100, 2000),
            'technician': f"Tech_{random.randint(1, 10)}",
            'description': f"Routine maintenance on {sensor_id}",
            'parts_replaced': random.choice([True, False, False, False]),  # 25% chance
            'status': random.choice(['Completed', 'Completed', 'Completed', 'Pending'])
        })
    
    return pd.DataFrame(records)
