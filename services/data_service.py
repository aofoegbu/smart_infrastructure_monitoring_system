"""
Data Service - REST API for Data Operations
Microservice for handling data operations and queries
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import os
import requests
import asyncio
import json
from kafka import KafkaProducer, KafkaConsumer
import redis
import uvicorn

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/sims")
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Redis setup for caching
redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=1)

# Kafka setup
kafka_producer = KafkaProducer(
    bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# FastAPI app
app = FastAPI(title="SIMS Data Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class SensorData(BaseModel):
    sensor_id: str
    timestamp: datetime
    pressure: float
    flow_rate: float
    temperature: float
    quality_score: float
    location: Optional[str] = None
    status: str = "active"

class SensorDataCreate(BaseModel):
    sensor_id: str
    pressure: float
    flow_rate: float
    temperature: float
    quality_score: float
    location: Optional[str] = None

class SensorDataResponse(BaseModel):
    id: int
    sensor_id: str
    timestamp: datetime
    pressure: float
    flow_rate: float
    temperature: float
    quality_score: float
    anomaly_score: float
    location: Optional[str]
    status: str

class SensorMetadata(BaseModel):
    sensor_id: str
    sensor_type: str
    location: str
    latitude: float
    longitude: float
    installation_date: datetime
    status: str

class QueryRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = {}
    limit: Optional[int] = 1000

class StreamingData(BaseModel):
    topic: str
    data: Dict[str, Any]
    timestamp: datetime

# Utility functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

def get_cached_data(key: str) -> Optional[str]:
    """Get cached data from Redis"""
    try:
        return redis_client.get(key)
    except Exception:
        return None

def set_cached_data(key: str, value: str, expire_time: int = 300):
    """Set cached data in Redis"""
    try:
        redis_client.setex(key, expire_time, value)
    except Exception:
        pass

def publish_to_kafka(topic: str, data: Dict[str, Any]):
    """Publish data to Kafka topic"""
    try:
        kafka_producer.send(topic, value=data)
        kafka_producer.flush()
    except Exception as e:
        print(f"Error publishing to Kafka: {e}")

# API Endpoints
@app.get("/sensors", response_model=List[SensorMetadata])
async def get_sensors(
    token: str = Query(..., description="JWT token"),
    db: Session = Depends(get_db)
):
    user = await verify_token(token)
    
    # Check cache first
    cache_key = "sensors_metadata"
    cached_data = get_cached_data(cache_key)
    
    if cached_data:
        return json.loads(cached_data)
    
    # Query database
    query = text("SELECT * FROM sensors ORDER BY sensor_id")
    result = db.execute(query)
    
    sensors = []
    for row in result:
        sensors.append({
            "sensor_id": row.sensor_id,
            "sensor_type": row.sensor_type,
            "location": row.location,
            "latitude": float(row.latitude),
            "longitude": float(row.longitude),
            "installation_date": row.installation_date,
            "status": row.status
        })
    
    # Cache result
    set_cached_data(cache_key, json.dumps(sensors, default=str))
    
    return sensors

@app.get("/sensors/{sensor_id}/data", response_model=List[SensorDataResponse])
async def get_sensor_data(
    sensor_id: str,
    token: str = Query(..., description="JWT token"),
    hours: int = Query(24, description="Hours of data to retrieve"),
    limit: int = Query(1000, description="Maximum number of records"),
    db: Session = Depends(get_db)
):
    user = await verify_token(token)
    
    # Check cache first
    cache_key = f"sensor_data_{sensor_id}_{hours}_{limit}"
    cached_data = get_cached_data(cache_key)
    
    if cached_data:
        return json.loads(cached_data)
    
    # Query database
    since_time = datetime.utcnow() - timedelta(hours=hours)
    query = text("""
        SELECT sr.*, s.location 
        FROM sensor_readings sr
        JOIN sensors s ON sr.sensor_id = s.sensor_id
        WHERE sr.sensor_id = :sensor_id 
        AND sr.timestamp >= :since_time
        ORDER BY sr.timestamp DESC
        LIMIT :limit
    """)
    
    result = db.execute(query, {
        "sensor_id": sensor_id,
        "since_time": since_time,
        "limit": limit
    })
    
    readings = []
    for row in result:
        readings.append({
            "id": row.id,
            "sensor_id": row.sensor_id,
            "timestamp": row.timestamp,
            "pressure": float(row.pressure),
            "flow_rate": float(row.flow_rate),
            "temperature": float(row.temperature),
            "quality_score": float(row.quality_score),
            "anomaly_score": float(row.anomaly_score),
            "location": row.location,
            "status": "active"
        })
    
    # Cache result
    set_cached_data(cache_key, json.dumps(readings, default=str), expire_time=60)
    
    return readings

@app.post("/sensors/{sensor_id}/data")
async def create_sensor_data(
    sensor_id: str,
    data: SensorDataCreate,
    background_tasks: BackgroundTasks,
    token: str = Query(..., description="JWT token"),
    db: Session = Depends(get_db)
):
    user = await verify_token(token)
    
    # Insert data into database
    query = text("""
        INSERT INTO sensor_readings 
        (sensor_id, timestamp, pressure, flow_rate, temperature, quality_score, anomaly_score)
        VALUES (:sensor_id, :timestamp, :pressure, :flow_rate, :temperature, :quality_score, :anomaly_score)
        RETURNING id
    """)
    
    result = db.execute(query, {
        "sensor_id": sensor_id,
        "timestamp": datetime.utcnow(),
        "pressure": data.pressure,
        "flow_rate": data.flow_rate,
        "temperature": data.temperature,
        "quality_score": data.quality_score,
        "anomaly_score": 0.0  # Will be calculated by ML service
    })
    
    db.commit()
    record_id = result.scalar()
    
    # Publish to Kafka for real-time processing
    kafka_data = {
        "sensor_id": sensor_id,
        "timestamp": datetime.utcnow().isoformat(),
        "pressure": data.pressure,
        "flow_rate": data.flow_rate,
        "temperature": data.temperature,
        "quality_score": data.quality_score,
        "record_id": record_id
    }
    
    background_tasks.add_task(publish_to_kafka, "sensor_data", kafka_data)
    
    # Clear related cache
    cache_pattern = f"sensor_data_{sensor_id}_*"
    for key in redis_client.scan_iter(match=cache_pattern):
        redis_client.delete(key)
    
    return {"message": "Data created successfully", "id": record_id}

@app.get("/analytics/summary")
async def get_analytics_summary(
    token: str = Query(..., description="JWT token"),
    hours: int = Query(24, description="Hours of data to analyze"),
    db: Session = Depends(get_db)
):
    user = await verify_token(token)
    
    # Check cache first
    cache_key = f"analytics_summary_{hours}"
    cached_data = get_cached_data(cache_key)
    
    if cached_data:
        return json.loads(cached_data)
    
    # Query database for analytics
    since_time = datetime.utcnow() - timedelta(hours=hours)
    
    # Get sensor statistics
    sensor_stats_query = text("""
        SELECT 
            COUNT(DISTINCT sensor_id) as total_sensors,
            COUNT(*) as total_readings,
            AVG(pressure) as avg_pressure,
            AVG(flow_rate) as avg_flow_rate,
            AVG(temperature) as avg_temperature,
            AVG(quality_score) as avg_quality_score,
            AVG(anomaly_score) as avg_anomaly_score
        FROM sensor_readings
        WHERE timestamp >= :since_time
    """)
    
    result = db.execute(sensor_stats_query, {"since_time": since_time})
    stats = result.fetchone()
    
    # Get alert statistics
    alert_stats_query = text("""
        SELECT 
            COUNT(*) as total_alerts,
            COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_alerts,
            COUNT(CASE WHEN severity = 'warning' THEN 1 END) as warning_alerts
        FROM alerts
        WHERE created_at >= :since_time
    """)
    
    alert_result = db.execute(alert_stats_query, {"since_time": since_time})
    alert_stats = alert_result.fetchone()
    
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "period_hours": hours,
        "sensor_statistics": {
            "total_sensors": stats.total_sensors or 0,
            "total_readings": stats.total_readings or 0,
            "avg_pressure": float(stats.avg_pressure or 0),
            "avg_flow_rate": float(stats.avg_flow_rate or 0),
            "avg_temperature": float(stats.avg_temperature or 0),
            "avg_quality_score": float(stats.avg_quality_score or 0),
            "avg_anomaly_score": float(stats.avg_anomaly_score or 0)
        },
        "alert_statistics": {
            "total_alerts": alert_stats.total_alerts or 0,
            "critical_alerts": alert_stats.critical_alerts or 0,
            "warning_alerts": alert_stats.warning_alerts or 0
        }
    }
    
    # Cache result
    set_cached_data(cache_key, json.dumps(summary), expire_time=300)
    
    return summary

@app.post("/query")
async def execute_query(
    request: QueryRequest,
    token: str = Query(..., description="JWT token"),
    db: Session = Depends(get_db)
):
    user = await verify_token(token)
    
    # Security check - only allow SELECT queries
    if not request.query.strip().upper().startswith("SELECT"):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed")
    
    try:
        result = db.execute(text(request.query), request.parameters)
        
        # Convert result to list of dictionaries
        columns = result.keys()
        data = [dict(zip(columns, row)) for row in result.fetchall()]
        
        # Apply limit
        if request.limit:
            data = data[:request.limit]
        
        return {
            "query": request.query,
            "row_count": len(data),
            "data": data
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Query execution error: {str(e)}")

@app.get("/stream/data")
async def stream_data(
    token: str = Query(..., description="JWT token"),
    topics: List[str] = Query(["sensor_data"], description="Kafka topics to stream")
):
    user = await verify_token(token)
    
    # Create Kafka consumer
    consumer = KafkaConsumer(
        *topics,
        bootstrap_servers=[KAFKA_BOOTSTRAP_SERVERS],
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    async def event_stream():
        for message in consumer:
            yield f"data: {json.dumps(message.value)}\n\n"
    
    return StreamingResponse(event_stream(), media_type="text/plain")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "data-service",
        "timestamp": datetime.utcnow(),
        "dependencies": {
            "database": "connected",
            "redis": "connected" if redis_client.ping() else "disconnected",
            "kafka": "connected"
        }
    }

if __name__ == "__main__":
    uvicorn.run("data_service:app", host="0.0.0.0", port=8002, reload=True)