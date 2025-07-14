import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import os
from datetime import datetime, timedelta

class AnomalyDetector:
    """
    Machine Learning based anomaly detection for sensor data
    """
    
    def __init__(self, method='isolation_forest'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, data, features=['pressure', 'flow_rate', 'temperature']):
        """
        Train the anomaly detection model
        """
        X = data[features].dropna()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        return self
    
    def predict(self, data, features=['pressure', 'flow_rate', 'temperature']):
        """
        Predict anomalies in new data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = data[features].dropna()
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'isolation_forest':
            anomaly_labels = self.model.predict(X_scaled)
            anomaly_scores = self.model.decision_function(X_scaled)
            # Convert to 0-1 scale (higher = more anomalous)
            anomaly_scores = (anomaly_scores.max() - anomaly_scores) / (anomaly_scores.max() - anomaly_scores.min())
        elif self.method == 'dbscan':
            cluster_labels = self.model.fit_predict(X_scaled)
            anomaly_labels = np.where(cluster_labels == -1, -1, 1)
            anomaly_scores = np.where(cluster_labels == -1, 0.8, 0.2)
        
        return anomaly_labels, anomaly_scores
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'method': self.method
            }, filepath)
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data['model']
            self.scaler = data['scaler']
            self.method = data['method']
            self.is_trained = True

def detect_anomalies(sensor_data, method='isolation_forest', contamination=0.1):
    """
    Detect anomalies in sensor data using various methods
    """
    features = ['pressure', 'flow_rate', 'temperature']
    
    # Prepare data
    X = sensor_data[features].dropna()
    if len(X) < 10:
        # Not enough data for anomaly detection
        return np.zeros(len(sensor_data))
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'isolation_forest':
        model = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = model.fit_predict(X_scaled)
        anomaly_scores = model.decision_function(X_scaled)
        # Normalize scores to 0-1
        anomaly_scores = (anomaly_scores.max() - anomaly_scores) / (anomaly_scores.max() - anomaly_scores.min())
    
    elif method == 'statistical':
        # Z-score based anomaly detection
        z_scores = np.abs((X - X.mean()) / X.std())
        max_z_scores = z_scores.max(axis=1)
        threshold = 3.0
        anomaly_scores = np.minimum(max_z_scores / threshold, 1.0)
        anomaly_labels = np.where(max_z_scores > threshold, -1, 1)
    
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=0.3, min_samples=5)
        cluster_labels = dbscan.fit_predict(X_scaled)
        anomaly_labels = np.where(cluster_labels == -1, -1, 1)
        anomaly_scores = np.where(cluster_labels == -1, 0.8, 0.2)
    
    else:
        # Default to simple statistical method
        z_scores = np.abs((X - X.mean()) / X.std())
        max_z_scores = z_scores.max(axis=1)
        anomaly_scores = np.minimum(max_z_scores / 3.0, 1.0)
        anomaly_labels = np.where(max_z_scores > 2.5, -1, 1)
    
    # Handle NaN values in original data
    full_anomaly_scores = np.full(len(sensor_data), 0.0)
    full_anomaly_scores[sensor_data[features].dropna().index] = anomaly_scores
    
    return full_anomaly_scores

def predict_maintenance(sensor_data):
    """
    Predict maintenance needs based on sensor patterns
    """
    features = ['pressure', 'flow_rate', 'temperature', 'anomaly_score']
    
    # Calculate maintenance scores based on multiple factors
    maintenance_scores = []
    
    for _, sensor in sensor_data.iterrows():
        score = 0.0
        
        # High anomaly score increases maintenance need
        score += sensor['anomaly_score'] * 0.4
        
        # Extreme values indicate potential issues
        if sensor['pressure'] > 70 or sensor['pressure'] < 25:
            score += 0.3
        
        if sensor['flow_rate'] > 40 or sensor['flow_rate'] < 10:
            score += 0.2
        
        if sensor['temperature'] > 30 or sensor['temperature'] < 15:
            score += 0.1
        
        # Random factor for demonstration
        score += np.random.uniform(0, 0.1)
        
        # Ensure score is between 0 and 1
        score = min(1.0, max(0.0, score))
        maintenance_scores.append(score)
    
    return np.array(maintenance_scores)

def train_anomaly_model(historical_data):
    """
    Train an anomaly detection model on historical data
    """
    detector = AnomalyDetector('isolation_forest')
    detector.train(historical_data)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    detector.save_model('models/anomaly_detector.pkl')
    
    return detector

def load_anomaly_model():
    """
    Load a pre-trained anomaly detection model
    """
    detector = AnomalyDetector()
    try:
        detector.load_model('models/anomaly_detector.pkl')
    except:
        # If no model exists, return untrained detector
        pass
    
    return detector

def calculate_sensor_health_score(sensor_data):
    """
    Calculate overall health score for sensors
    """
    health_scores = []
    
    for sensor_id in sensor_data['sensor_id'].unique():
        sensor_subset = sensor_data[sensor_data['sensor_id'] == sensor_id]
        
        # Base health score
        health = 100.0
        
        # Reduce health based on anomaly scores
        avg_anomaly = sensor_subset['anomaly_score'].mean()
        health -= avg_anomaly * 30
        
        # Check for data completeness
        completeness = sensor_subset[['pressure', 'flow_rate', 'temperature']].notna().all(axis=1).mean()
        health *= completeness
        
        # Check for stability (low variance is good)
        pressure_cv = sensor_subset['pressure'].std() / sensor_subset['pressure'].mean()
        if pressure_cv > 0.2:  # High coefficient of variation
            health -= 10
        
        health_scores.append({
            'sensor_id': sensor_id,
            'health_score': max(0, min(100, health))
        })
    
    return pd.DataFrame(health_scores)

def perform_predictive_analytics(sensor_data):
    """
    Perform predictive analytics on sensor data
    """
    results = {}
    
    # Time series forecasting (simplified)
    if 'timestamp' in sensor_data.columns:
        # Group by sensor and calculate trends
        trends = {}
        for sensor_id in sensor_data['sensor_id'].unique():
            sensor_subset = sensor_data[sensor_data['sensor_id'] == sensor_id].copy()
            sensor_subset = sensor_subset.sort_values('timestamp')
            
            if len(sensor_subset) > 5:
                # Simple linear trend
                x = np.arange(len(sensor_subset))
                pressure_trend = np.polyfit(x, sensor_subset['pressure'], 1)[0]
                flow_trend = np.polyfit(x, sensor_subset['flow_rate'], 1)[0]
                
                trends[sensor_id] = {
                    'pressure_trend': pressure_trend,
                    'flow_trend': flow_trend
                }
        
        results['trends'] = trends
    
    # Failure prediction
    failure_probabilities = predict_maintenance(sensor_data)
    results['failure_probabilities'] = failure_probabilities
    
    # Correlation analysis
    features = ['pressure', 'flow_rate', 'temperature']
    correlation_matrix = sensor_data[features].corr()
    results['correlations'] = correlation_matrix
    
    return results

class PredictiveMaintenanceModel:
    """
    Advanced predictive maintenance model
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def create_features(self, data):
        """
        Create features for predictive maintenance
        """
        features = []
        
        # Statistical features
        features.extend([
            data['pressure'].mean(),
            data['pressure'].std(),
            data['flow_rate'].mean(),
            data['flow_rate'].std(),
            data['temperature'].mean(),
            data['temperature'].std(),
            data['anomaly_score'].mean(),
            data['anomaly_score'].max()
        ])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, historical_data, maintenance_records):
        """
        Train the predictive maintenance model
        """
        # This would normally use actual maintenance records
        # For demonstration, we'll create synthetic training data
        
        X = []
        y = []
        
        for sensor_id in historical_data['sensor_id'].unique():
            sensor_data = historical_data[historical_data['sensor_id'] == sensor_id]
            features = self.create_features(sensor_data)
            
            # Synthetic label (1 = needs maintenance, 0 = doesn't need)
            needs_maintenance = np.random.choice([0, 1], p=[0.8, 0.2])
            
            X.append(features.flatten())
            y.append(needs_maintenance)
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return self
    
    def predict_maintenance_need(self, sensor_data):
        """
        Predict if maintenance is needed
        """
        if not self.is_trained:
            # Return random predictions if not trained
            return np.random.uniform(0, 1)
        
        features = self.create_features(sensor_data)
        features_scaled = self.scaler.transform(features)
        
        probability = self.model.predict_proba(features_scaled)[0][1]
        return probability
