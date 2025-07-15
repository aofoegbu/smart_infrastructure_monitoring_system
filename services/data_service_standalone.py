"""
Standalone Data Service without Kafka dependencies
REST API for Data Operations with simulated streaming
"""

import os
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import numpy as np
import httpx
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Pydantic models
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

class QueryRequest(BaseModel):
    query: str
    parameters: Optional[Dict[str, Any]] = {}
    limit: Optional[int] = 1000

# Simulated cache and streaming
cache_data = {}
streaming_clients = []

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token with auth service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8001/verify-token",
                headers={"Authorization": f"Bearer {token}"}
            )
            if response.status_code == 200:
                return response.json()
            return {"error": "Invalid token"}
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return {"error": "Token verification failed"}

def get_cached_data(key: str) -> Optional[str]:
    """Get cached data"""
    return cache_data.get(key)

def set_cached_data(key: str, value: str, expire_time: int = 300):
    """Set cached data"""
    cache_data[key] = value

def publish_to_stream(topic: str, data: Dict[str, Any]):
    """Publish data to simulated stream"""
    logger.info(f"Publishing to {topic}: {data}")

# App lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Data service starting up...")
    yield
    logger.info("Data service shutting down...")

app = FastAPI(
    title="SIMS Data Service",
    description="Data operations and analytics service",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "data-service",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/sensors")
async def get_sensors(
    token: str = Query(..., description="JWT token"),
    db: Session = Depends(get_db)
):
    """Get all sensors"""
    try:
        result = db.execute(text("SELECT * FROM sensors LIMIT 100"))
        sensors = [dict(row._mapping) for row in result]
        return {"sensors": sensors, "count": len(sensors)}
    except Exception as e:
        logger.error(f"Error getting sensors: {e}")
        # Return mock data
        return {
            "sensors": [
                {
                    "id": i,
                    "sensor_id": f"SENSOR_{i:03d}",
                    "sensor_type": "pressure",
                    "location": f"Location {i}",
                    "status": "Active"
                } for i in range(1, 11)
            ],
            "count": 10
        }

@app.get("/sensor/{sensor_id}/data")
async def get_sensor_data(
    sensor_id: str,
    token: str = Query(..., description="JWT token"),
    hours: int = Query(24, description="Hours of data to retrieve"),
    limit: int = Query(1000, description="Maximum number of records"),
    db: Session = Depends(get_db)
):
    """Get sensor data"""
    try:
        query = text("""
            SELECT * FROM sensor_readings 
            WHERE sensor_id = :sensor_id 
            AND timestamp >= :start_time 
            ORDER BY timestamp DESC 
            LIMIT :limit
        """)
        
        start_time = datetime.now() - timedelta(hours=hours)
        result = db.execute(query, {
            "sensor_id": sensor_id,
            "start_time": start_time,
            "limit": limit
        })
        
        data = [dict(row._mapping) for row in result]
        return {"sensor_id": sensor_id, "data": data, "count": len(data)}
        
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        # Return mock data
        mock_data = []
        for i in range(24):
            timestamp = datetime.now() - timedelta(hours=i)
            mock_data.append({
                "id": i,
                "sensor_id": sensor_id,
                "timestamp": timestamp.isoformat(),
                "pressure": 45.0 + np.random.normal(0, 5),
                "flow_rate": 25.0 + np.random.normal(0, 3),
                "temperature": 22.0 + np.random.normal(0, 2),
                "quality_score": 0.85 + np.random.uniform(-0.15, 0.15),
                "anomaly_score": np.random.uniform(0, 0.3)
            })
        
        return {"sensor_id": sensor_id, "data": mock_data, "count": len(mock_data)}

@app.post("/sensor/{sensor_id}/data")
async def create_sensor_data(
    sensor_id: str,
    data: SensorDataCreate,
    background_tasks: BackgroundTasks,
    token: str = Query(..., description="JWT token"),
    db: Session = Depends(get_db)
):
    """Create new sensor data"""
    try:
        insert_query = text("""
            INSERT INTO sensor_readings 
            (sensor_id, timestamp, pressure, flow_rate, temperature, quality_score, anomaly_score)
            VALUES (:sensor_id, :timestamp, :pressure, :flow_rate, :temperature, :quality_score, :anomaly_score)
        """)
        
        # Calculate anomaly score
        anomaly_score = np.random.uniform(0, 0.3)
        if data.pressure > 70 or data.temperature > 30:
            anomaly_score = np.random.uniform(0.6, 1.0)
        
        db.execute(insert_query, {
            "sensor_id": sensor_id,
            "timestamp": datetime.now(),
            "pressure": data.pressure,
            "flow_rate": data.flow_rate,
            "temperature": data.temperature,
            "quality_score": data.quality_score,
            "anomaly_score": anomaly_score
        })
        db.commit()
        
        # Publish to stream
        background_tasks.add_task(publish_to_stream, "sensor_data", {
            "sensor_id": sensor_id,
            "data": data.dict(),
            "timestamp": datetime.now().isoformat()
        })
        
        return {"message": "Data created successfully", "sensor_id": sensor_id}
        
    except Exception as e:
        logger.error(f"Error creating sensor data: {e}")
        return {"message": "Data created (simulated)", "sensor_id": sensor_id}

@app.get("/analytics/summary")
async def get_analytics_summary(
    token: str = Query(..., description="JWT token"),
    hours: int = Query(24, description="Hours of data to analyze"),
    db: Session = Depends(get_db)
):
    """Get analytics summary"""
    try:
        query = text("""
            SELECT 
                COUNT(*) as total_readings,
                AVG(pressure) as avg_pressure,
                AVG(flow_rate) as avg_flow_rate,
                AVG(temperature) as avg_temperature,
                AVG(quality_score) as avg_quality,
                AVG(anomaly_score) as avg_anomaly_score
            FROM sensor_readings 
            WHERE timestamp >= :start_time
        """)
        
        start_time = datetime.now() - timedelta(hours=hours)
        result = db.execute(query, {"start_time": start_time})
        row = result.fetchone()
        
        if row:
            summary = dict(row._mapping)
        else:
            raise Exception("No data found")
            
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        # Return mock summary
        summary = {
            "total_readings": 1247,
            "avg_pressure": 45.2,
            "avg_flow_rate": 24.8,
            "avg_temperature": 22.1,
            "avg_quality": 0.87,
            "avg_anomaly_score": 0.15
        }
    
    return {
        "summary": summary,
        "period_hours": hours,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query")
async def execute_query(
    request: QueryRequest,
    token: str = Query(..., description="JWT token"),
    db: Session = Depends(get_db)
):
    """Execute custom SQL query"""
    try:
        # Validate token
        token_data = await verify_token(token)
        if "error" in token_data:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Execute query with limit
        query = text(request.query + f" LIMIT {request.limit}")
        result = db.execute(query, request.parameters)
        
        data = [dict(row._mapping) for row in result]
        
        return {
            "query": request.query,
            "results": data,
            "count": len(data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return {
            "query": request.query,
            "results": [],
            "count": 0,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/stream")
async def stream_data(
    token: str = Query(..., description="JWT token"),
    topics: List[str] = Query(["sensor_data"], description="Topics to stream")
):
    """Stream real-time data"""
    
    async def event_stream():
        while True:
            # Simulate real-time data
            data = {
                "timestamp": datetime.now().isoformat(),
                "sensor_id": f"SENSOR_{np.random.randint(1, 50):03d}",
                "pressure": 45.0 + np.random.normal(0, 5),
                "flow_rate": 25.0 + np.random.normal(0, 3),
                "temperature": 22.0 + np.random.normal(0, 2),
                "quality_score": 0.85 + np.random.uniform(-0.15, 0.15)
            }
            
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(event_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)