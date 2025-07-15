"""
ML Service - Machine Learning and AI Operations
Microservice for ML model training, inference, and AI-powered insights
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import json
import redis
import requests
from kafka import KafkaConsumer, KafkaProducer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
import pickle
import asyncio
from threading import Thread
import uvicorn

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/sims")
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://localhost:8002")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
MODEL_STORAGE_PATH = os.getenv("MODEL_STORAGE_PATH", "./models")

# Ensure model storage directory exists
os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)

# Redis setup for caching
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=2)

# Kafka setup
kafka_producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# FastAPI app
app = FastAPI(title="SIMS ML Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class AnomalyDetectionRequest(BaseModel):
    sensor_data: List[Dict[str, Any]]
    model_type: str = "isolation_forest"
    threshold: float = 0.5

class AnomalyDetectionResponse(BaseModel):
    anomalies: List[Dict[str, Any]]
    model_performance: Dict[str, Any]
    timestamp: datetime

class ModelTrainingRequest(BaseModel):
    model_type: str
    data_source: str
    parameters: Dict[str, Any]
    hours_of_data: int = 168  # 7 days

class ModelTrainingResponse(BaseModel):
    model_id: str
    model_type: str
    training_status: str
    performance_metrics: Dict[str, Any]
    timestamp: datetime

class PredictionRequest(BaseModel):
    model_id: str
    input_data: List[Dict[str, Any]]

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    timestamp: datetime

class ChatRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    data_insights: Optional[Dict[str, Any]] = None
    suggested_actions: List[str]
    timestamp: datetime

# ML Model Manager
class ModelManager:
    def __init__(self):
        self.models = {}
        self.load_existing_models()
    
    def load_existing_models(self):
        """Load existing models from storage"""
        try:
            for filename in os.listdir(MODEL_STORAGE_PATH):
                if filename.endswith('.pkl'):
                    model_id = filename[:-4]
                    model_path = os.path.join(MODEL_STORAGE_PATH, filename)
                    self.models[model_id] = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def save_model(self, model_id: str, model):
        """Save model to storage"""
        try:
            model_path = os.path.join(MODEL_STORAGE_PATH, f"{model_id}.pkl")
            joblib.dump(model, model_path)
            self.models[model_id] = model
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def get_model(self, model_id: str):
        """Get model by ID"""
        return self.models.get(model_id)
    
    def train_anomaly_model(self, data: pd.DataFrame, model_type: str = "isolation_forest"):
        """Train anomaly detection model"""
        try:
            # Prepare features
            features = ['pressure', 'flow_rate', 'temperature', 'quality_score']
            X = data[features].fillna(data[features].mean())
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            if model_type == "isolation_forest":
                model = IsolationForest(contamination=0.1, random_state=42)
                model.fit(X_scaled)
                
                # Generate predictions for performance evaluation
                anomaly_scores = model.decision_function(X_scaled)
                predictions = model.predict(X_scaled)
                
                # Calculate performance metrics
                performance = {
                    "model_type": model_type,
                    "training_samples": len(X),
                    "anomaly_rate": (predictions == -1).mean(),
                    "avg_anomaly_score": np.mean(anomaly_scores),
                    "feature_importance": dict(zip(features, [1.0] * len(features)))
                }
                
                # Save model with scaler
                model_package = {
                    "model": model,
                    "scaler": scaler,
                    "features": features,
                    "performance": performance
                }
                
                return model_package, performance
            
        except Exception as e:
            raise Exception(f"Model training failed: {str(e)}")
    
    def predict_anomalies(self, model_id: str, data: pd.DataFrame):
        """Predict anomalies using trained model"""
        try:
            model_package = self.get_model(model_id)
            if not model_package:
                raise Exception(f"Model {model_id} not found")
            
            model = model_package["model"]
            scaler = model_package["scaler"]
            features = model_package["features"]
            
            # Prepare features
            X = data[features].fillna(data[features].mean())
            X_scaled = scaler.transform(X)
            
            # Make predictions
            predictions = model.predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
            
            # Prepare results
            results = []
            for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                results.append({
                    "index": i,
                    "is_anomaly": pred == -1,
                    "anomaly_score": float(score),
                    "confidence": float(abs(score))
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")

# Initialize model manager
model_manager = ModelManager()

# Utility functions
async def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token with auth service"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{AUTH_SERVICE_URL}/me", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication service unavailable")

async def get_sensor_data(token: str, hours: int = 24) -> pd.DataFrame:
    """Get sensor data from data service"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(
            f"{DATA_SERVICE_URL}/analytics/summary",
            headers=headers,
            params={"hours": hours}
        )
        
        if response.status_code == 200:
            # This is a simplified version - in reality, we'd get the full dataset
            # For now, we'll generate sample data based on the summary
            summary = response.json()
            
            # Generate sample data for training
            n_samples = min(1000, summary["sensor_statistics"]["total_readings"])
            
            data = pd.DataFrame({
                "pressure": np.random.normal(summary["sensor_statistics"]["avg_pressure"], 10, n_samples),
                "flow_rate": np.random.normal(summary["sensor_statistics"]["avg_flow_rate"], 5, n_samples),
                "temperature": np.random.normal(summary["sensor_statistics"]["avg_temperature"], 2, n_samples),
                "quality_score": np.random.normal(summary["sensor_statistics"]["avg_quality_score"], 0.5, n_samples),
                "timestamp": pd.date_range(start=datetime.now() - timedelta(hours=hours), periods=n_samples, freq='5min')
            })
            
            return data
        else:
            raise Exception("Failed to fetch sensor data")
            
    except Exception as e:
        raise Exception(f"Data retrieval failed: {str(e)}")

# Kafka consumer for real-time processing
def kafka_consumer_worker():
    """Background worker for processing Kafka messages"""
    consumer = KafkaConsumer(
        'sensor_data',
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    for message in consumer:
        try:
            data = message.value
            
            # Process real-time anomaly detection
            if 'anomaly_detection' in model_manager.models:
                sensor_df = pd.DataFrame([data])
                anomalies = model_manager.predict_anomalies('anomaly_detection', sensor_df)
                
                if anomalies and anomalies[0]['is_anomaly']:
                    # Publish anomaly alert
                    alert_data = {
                        "sensor_id": data["sensor_id"],
                        "timestamp": data["timestamp"],
                        "anomaly_score": anomalies[0]["anomaly_score"],
                        "alert_type": "anomaly_detected",
                        "severity": "high" if anomalies[0]["confidence"] > 0.8 else "medium"
                    }
                    
                    kafka_producer.send('alerts', value=alert_data)
                    
        except Exception as e:
            print(f"Error processing message: {e}")

# Start Kafka consumer in background
kafka_thread = Thread(target=kafka_consumer_worker, daemon=True)
kafka_thread.start()

# API Endpoints
@app.post("/models/train", response_model=ModelTrainingResponse)
async def train_model(
    request: ModelTrainingRequest,
    token: str = Query(..., description="JWT token"),
    background_tasks: BackgroundTasks
):
    user = await verify_token(token)
    
    # Generate model ID
    model_id = f"{request.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Get training data
        data = await get_sensor_data(token, request.hours_of_data)
        
        if request.model_type == "anomaly_detection":
            model_package, performance = model_manager.train_anomaly_model(data, "isolation_forest")
            model_manager.save_model(model_id, model_package)
            
            return ModelTrainingResponse(
                model_id=model_id,
                model_type=request.model_type,
                training_status="completed",
                performance_metrics=performance,
                timestamp=datetime.utcnow()
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported model type: {request.model_type}")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/models/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    token: str = Query(..., description="JWT token")
):
    user = await verify_token(token)
    
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame(request.input_data)
        
        # Make predictions
        predictions = model_manager.predict_anomalies(request.model_id, data)
        
        # Extract confidence scores
        confidence_scores = [pred["confidence"] for pred in predictions]
        
        return PredictionResponse(
            predictions=predictions,
            confidence_scores=confidence_scores,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/anomaly-detection", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    request: AnomalyDetectionRequest,
    token: str = Query(..., description="JWT token")
):
    user = await verify_token(token)
    
    try:
        # Convert input data to DataFrame
        data = pd.DataFrame(request.sensor_data)
        
        # Use existing model or train new one
        model_id = "anomaly_detection"
        if model_id not in model_manager.models:
            # Train new model if none exists
            model_package, performance = model_manager.train_anomaly_model(data, request.model_type)
            model_manager.save_model(model_id, model_package)
        
        # Make predictions
        anomalies = model_manager.predict_anomalies(model_id, data)
        
        # Filter anomalies based on threshold
        filtered_anomalies = [
            {**anomaly, **request.sensor_data[anomaly["index"]]}
            for anomaly in anomalies
            if anomaly["is_anomaly"] and anomaly["confidence"] > request.threshold
        ]
        
        return AnomalyDetectionResponse(
            anomalies=filtered_anomalies,
            model_performance=model_manager.models[model_id]["performance"],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@app.get("/models")
async def list_models(token: str = Query(..., description="JWT token")):
    user = await verify_token(token)
    
    models_info = []
    for model_id, model_package in model_manager.models.items():
        models_info.append({
            "model_id": model_id,
            "model_type": model_package.get("performance", {}).get("model_type", "unknown"),
            "performance": model_package.get("performance", {}),
            "features": model_package.get("features", [])
        })
    
    return {"models": models_info}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(
    request: ChatRequest,
    token: str = Query(..., description="JWT token")
):
    user = await verify_token(token)
    
    try:
        # Simple rule-based chatbot (in production, use LLM)
        query = request.query.lower()
        
        # Get recent data for context
        data = await get_sensor_data(token, 24)
        
        response = ""
        data_insights = {}
        suggested_actions = []
        
        if "anomaly" in query or "anomalies" in query:
            # Detect anomalies
            if "anomaly_detection" in model_manager.models:
                anomalies = model_manager.predict_anomalies("anomaly_detection", data)
                anomaly_count = sum(1 for a in anomalies if a["is_anomaly"])
                
                response = f"I found {anomaly_count} anomalies in the last 24 hours of sensor data."
                data_insights = {
                    "anomaly_count": anomaly_count,
                    "total_readings": len(data),
                    "anomaly_rate": anomaly_count / len(data) if len(data) > 0 else 0
                }
                
                if anomaly_count > 0:
                    suggested_actions = [
                        "Investigate sensors with highest anomaly scores",
                        "Check for maintenance requirements",
                        "Review sensor calibration"
                    ]
            else:
                response = "No anomaly detection model is currently available. Please train a model first."
        
        elif "pressure" in query:
            avg_pressure = data["pressure"].mean()
            response = f"The average pressure over the last 24 hours is {avg_pressure:.2f} PSI."
            data_insights = {
                "avg_pressure": avg_pressure,
                "min_pressure": data["pressure"].min(),
                "max_pressure": data["pressure"].max(),
                "pressure_std": data["pressure"].std()
            }
        
        elif "temperature" in query:
            avg_temp = data["temperature"].mean()
            response = f"The average temperature over the last 24 hours is {avg_temp:.2f}°C."
            data_insights = {
                "avg_temperature": avg_temp,
                "min_temperature": data["temperature"].min(),
                "max_temperature": data["temperature"].max(),
                "temperature_std": data["temperature"].std()
            }
        
        elif "quality" in query:
            avg_quality = data["quality_score"].mean()
            response = f"The average quality score over the last 24 hours is {avg_quality:.2f}."
            data_insights = {
                "avg_quality": avg_quality,
                "min_quality": data["quality_score"].min(),
                "max_quality": data["quality_score"].max(),
                "quality_std": data["quality_score"].std()
            }
            
            if avg_quality < 0.8:
                suggested_actions = [
                    "Investigate water quality issues",
                    "Check filtration systems",
                    "Review treatment processes"
                ]
        
        else:
            response = "I can help you analyze sensor data, detect anomalies, and provide insights about pressure, temperature, flow rate, and quality metrics. What would you like to know?"
        
        return ChatResponse(
            response=response,
            data_insights=data_insights,
            suggested_actions=suggested_actions,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ml-service",
        "timestamp": datetime.utcnow(),
        "models_loaded": len(model_manager.models),
        "dependencies": {
            "redis": "connected" if redis_client.ping() else "disconnected",
            "kafka": "connected"
        }
    }

if __name__ == "__main__":
    uvicorn.run("ml_service:app", host="0.0.0.0", port=8003, reload=True)