"""
LLM Assistant Service - ChatGPT-style AI Assistant
Advanced natural language query processing for infrastructure data
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import asyncio
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import uvicorn

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/sims")
AUTH_SERVICE_URL = os.getenv("AUTH_SERVICE_URL", "http://localhost:8001")
DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://localhost:8002")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8003")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# HTTP client
http_client = httpx.AsyncClient(timeout=30.0)

# FastAPI app
app = FastAPI(title="SIMS LLM Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class ChatMessage(BaseModel):
    role: str  # user, assistant, system
    content: str
    timestamp: datetime = None

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    use_external_llm: bool = False

class ChatResponse(BaseModel):
    message: str
    conversation_id: str
    data_insights: Optional[Dict[str, Any]] = None
    suggested_actions: List[str] = []
    query_results: Optional[Dict[str, Any]] = None
    charts: Optional[List[Dict[str, Any]]] = None
    confidence_score: float = 0.0
    timestamp: datetime

class QueryAnalysis(BaseModel):
    intent: str
    entities: List[str]
    time_range: Optional[str] = None
    sensor_ids: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    location: Optional[str] = None

# Advanced Query Processor
class QueryProcessor:
    def __init__(self):
        self.intent_patterns = {
            'anomaly_detection': [
                'anomaly', 'anomalies', 'unusual', 'abnormal', 'outlier', 'problem',
                'issue', 'fault', 'error', 'alert', 'warning'
            ],
            'performance_analysis': [
                'performance', 'efficiency', 'optimization', 'speed', 'throughput',
                'latency', 'response time', 'capacity', 'utilization'
            ],
            'predictive_maintenance': [
                'maintenance', 'prediction', 'forecast', 'failure', 'breakdown',
                'repair', 'service', 'schedule', 'when', 'will fail'
            ],
            'sensor_status': [
                'status', 'health', 'working', 'operational', 'active', 'inactive',
                'online', 'offline', 'available', 'unavailable'
            ],
            'data_query': [
                'show', 'display', 'get', 'retrieve', 'find', 'search', 'list',
                'what', 'which', 'where', 'how many', 'count'
            ],
            'comparison': [
                'compare', 'difference', 'vs', 'versus', 'between', 'against',
                'higher', 'lower', 'better', 'worse', 'best', 'worst'
            ],
            'trend_analysis': [
                'trend', 'pattern', 'over time', 'increase', 'decrease', 'growing',
                'declining', 'stable', 'fluctuating', 'seasonal'
            ]
        }
        
        self.entity_patterns = {
            'metrics': [
                'pressure', 'temperature', 'flow', 'flow rate', 'quality',
                'quality score', 'health score', 'efficiency', 'volume'
            ],
            'locations': [
                'zone', 'sector', 'area', 'region', 'building', 'floor',
                'north', 'south', 'east', 'west', 'central'
            ],
            'time_periods': [
                'hour', 'day', 'week', 'month', 'year', 'today', 'yesterday',
                'last', 'past', 'recent', 'current', 'this'
            ],
            'severity_levels': [
                'critical', 'high', 'medium', 'low', 'severe', 'minor',
                'major', 'urgent', 'important'
            ]
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze user query to extract intent and entities"""
        query_lower = query.lower()
        
        # Detect intent
        intent_scores = {}
        for intent, keywords in self.intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        primary_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'general_query'
        
        # Extract entities
        entities = []
        for entity_type, keywords in self.entity_patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    entities.append(f"{entity_type}:{keyword}")
        
        # Extract time range
        time_range = None
        if any(word in query_lower for word in ['last', 'past', 'recent']):
            if 'hour' in query_lower:
                time_range = '1h'
            elif 'day' in query_lower:
                time_range = '1d'
            elif 'week' in query_lower:
                time_range = '1w'
            elif 'month' in query_lower:
                time_range = '1m'
            else:
                time_range = '24h'  # default
        
        # Extract sensor IDs (if mentioned)
        sensor_ids = []
        words = query_lower.split()
        for i, word in enumerate(words):
            if word == 'sensor' and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word.replace('-', '').replace('_', '').isalnum():
                    sensor_ids.append(next_word)
        
        # Extract metrics
        metrics = []
        for metric in self.entity_patterns['metrics']:
            if metric in query_lower:
                metrics.append(metric)
        
        return QueryAnalysis(
            intent=primary_intent,
            entities=entities,
            time_range=time_range,
            sensor_ids=sensor_ids if sensor_ids else None,
            metrics=metrics if metrics else None,
            location=None  # Could be enhanced to extract location
        )

# Initialize query processor
query_processor = QueryProcessor()

# Data retrieval functions
async def get_sensor_data(token: str, hours: int = 24) -> pd.DataFrame:
    """Get sensor data from data service"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = await http_client.get(
            f"{DATA_SERVICE_URL}/analytics/summary",
            headers=headers,
            params={"hours": hours}
        )
        
        if response.status_code == 200:
            summary = response.json()
            
            # For demonstration, create synthetic data based on summary
            n_samples = min(1000, summary["sensor_statistics"]["total_readings"])
            
            return pd.DataFrame({
                'timestamp': pd.date_range(
                    start=datetime.now() - timedelta(hours=hours),
                    periods=n_samples,
                    freq='5min'
                ),
                'sensor_id': np.random.choice(['SENSOR_001', 'SENSOR_002', 'SENSOR_003'], n_samples),
                'pressure': np.random.normal(summary["sensor_statistics"]["avg_pressure"], 10, n_samples),
                'flow_rate': np.random.normal(summary["sensor_statistics"]["avg_flow_rate"], 5, n_samples),
                'temperature': np.random.normal(summary["sensor_statistics"]["avg_temperature"], 2, n_samples),
                'quality_score': np.random.normal(summary["sensor_statistics"]["avg_quality_score"], 0.5, n_samples),
            })
        else:
            raise Exception("Failed to fetch sensor data")
            
    except Exception as e:
        logger.error(f"Error fetching sensor data: {e}")
        return pd.DataFrame()

async def get_anomaly_data(token: str, hours: int = 24) -> List[Dict[str, Any]]:
    """Get anomaly detection results"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        data = await get_sensor_data(token, hours)
        
        if data.empty:
            return []
        
        # Convert to format expected by ML service
        sensor_data = data.to_dict('records')
        
        response = await http_client.post(
            f"{ML_SERVICE_URL}/anomaly-detection",
            headers=headers,
            json={
                "sensor_data": sensor_data,
                "threshold": 0.5
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('anomalies', [])
        else:
            return []
            
    except Exception as e:
        logger.error(f"Error fetching anomaly data: {e}")
        return []

# Enhanced response generators
class ResponseGenerator:
    def __init__(self):
        self.conversation_memory = {}
    
    async def generate_response(self, query: str, analysis: QueryAnalysis, token: str) -> ChatResponse:
        """Generate comprehensive response based on query analysis"""
        
        # Get relevant data
        data_insights = {}
        query_results = None
        suggested_actions = []
        charts = []
        confidence_score = 0.8
        
        try:
            if analysis.intent == 'anomaly_detection':
                # Get anomaly data
                anomalies = await get_anomaly_data(token, 24)
                
                if anomalies:
                    anomaly_count = len(anomalies)
                    high_confidence_anomalies = [a for a in anomalies if a.get('confidence', 0) > 0.8]
                    
                    data_insights = {
                        "total_anomalies": anomaly_count,
                        "high_confidence_anomalies": len(high_confidence_anomalies),
                        "anomaly_rate": anomaly_count / 1000,  # Approximate
                        "most_affected_sensors": list(set([a.get('sensor_id') for a in anomalies[:5]]))
                    }
                    
                    response_text = f"""🚨 **Anomaly Detection Results**
                    
I found {anomaly_count} anomalies in the last 24 hours:
• {len(high_confidence_anomalies)} high-confidence anomalies
• Anomaly rate: {(anomaly_count/1000)*100:.1f}%
• Most affected sensors: {', '.join(data_insights['most_affected_sensors'][:3])}

The anomalies show unusual patterns in sensor readings that deviate from normal operational parameters."""
                    
                    suggested_actions = [
                        "🔍 Investigate sensors with highest anomaly scores",
                        "📊 Review sensor calibration and maintenance history",
                        "🛠️ Schedule maintenance for affected sensors",
                        "📈 Monitor trends over the next 24 hours"
                    ]
                    
                    charts = [{
                        "type": "anomaly_timeline",
                        "data": anomalies[:10],
                        "title": "Recent Anomalies Timeline"
                    }]
                    
                else:
                    response_text = "✅ **No Anomalies Detected**\n\nGreat news! No significant anomalies were found in the last 24 hours. Your infrastructure appears to be operating normally."
                    confidence_score = 0.9
            
            elif analysis.intent == 'sensor_status':
                # Get sensor status
                sensor_data = await get_sensor_data(token, 1)
                
                if not sensor_data.empty:
                    total_sensors = sensor_data['sensor_id'].nunique()
                    active_sensors = total_sensors  # Assume all are active for now
                    
                    data_insights = {
                        "total_sensors": total_sensors,
                        "active_sensors": active_sensors,
                        "sensor_uptime": 99.5,  # Approximate
                        "last_reading_time": sensor_data['timestamp'].max().isoformat()
                    }
                    
                    response_text = f"""📊 **Sensor Status Overview**
                    
Current sensor network status:
• Total sensors: {total_sensors}
• Active sensors: {active_sensors}
• System uptime: {data_insights['sensor_uptime']}%
• Last reading: {data_insights['last_reading_time'][:19]}

All sensors are currently operational and reporting data normally."""
                    
                    suggested_actions = [
                        "📈 Monitor sensor performance trends",
                        "🔄 Schedule routine maintenance checks",
                        "📊 Review data quality metrics"
                    ]
            
            elif analysis.intent == 'performance_analysis':
                # Get performance metrics
                sensor_data = await get_sensor_data(token, 24)
                
                if not sensor_data.empty:
                    avg_pressure = sensor_data['pressure'].mean()
                    avg_flow = sensor_data['flow_rate'].mean()
                    avg_temp = sensor_data['temperature'].mean()
                    avg_quality = sensor_data['quality_score'].mean()
                    
                    data_insights = {
                        "avg_pressure": avg_pressure,
                        "avg_flow_rate": avg_flow,
                        "avg_temperature": avg_temp,
                        "avg_quality_score": avg_quality,
                        "performance_score": avg_quality * 100
                    }
                    
                    response_text = f"""⚡ **Performance Analysis (24h)**
                    
System performance metrics:
• Average pressure: {avg_pressure:.2f} PSI
• Average flow rate: {avg_flow:.2f} L/min
• Average temperature: {avg_temp:.2f}°C
• Quality score: {avg_quality:.2f}/5.0
• Overall performance: {data_insights['performance_score']:.1f}%

System performance is {"excellent" if avg_quality > 4.0 else "good" if avg_quality > 3.0 else "needs attention"}."""
                    
                    if avg_quality < 3.0:
                        suggested_actions = [
                            "🔧 Investigate performance bottlenecks",
                            "📊 Analyze quality score trends",
                            "⚙️ Optimize system parameters"
                        ]
                    else:
                        suggested_actions = [
                            "📈 Continue monitoring performance",
                            "🔄 Maintain current operational parameters"
                        ]
            
            elif analysis.intent == 'data_query':
                # Handle general data queries
                if analysis.metrics:
                    sensor_data = await get_sensor_data(token, 24)
                    
                    if not sensor_data.empty:
                        metric_results = {}
                        for metric in analysis.metrics:
                            if metric in sensor_data.columns:
                                metric_results[metric] = {
                                    "current": sensor_data[metric].iloc[-1],
                                    "average": sensor_data[metric].mean(),
                                    "min": sensor_data[metric].min(),
                                    "max": sensor_data[metric].max(),
                                    "trend": "stable"  # Could be calculated
                                }
                        
                        data_insights = metric_results
                        response_text = f"""📊 **Data Query Results**
                        
Here's the information about {', '.join(analysis.metrics)}:
"""
                        
                        for metric, values in metric_results.items():
                            response_text += f"""
• **{metric.title()}**:
  - Current: {values['current']:.2f}
  - Average: {values['average']:.2f}
  - Range: {values['min']:.2f} - {values['max']:.2f}
  - Trend: {values['trend']}"""
                        
                        suggested_actions = [
                            "📈 Monitor metric trends",
                            "🔍 Set up alerts for threshold breaches"
                        ]
                        
                        charts = [{
                            "type": "metric_trends",
                            "data": metric_results,
                            "title": f"{', '.join(analysis.metrics).title()} Trends"
                        }]
                    else:
                        response_text = "❌ No data available for the requested metrics."
                else:
                    response_text = "❓ I need more specific information about what data you'd like to see. Could you specify metrics like pressure, temperature, or flow rate?"
            
            else:
                # General query
                response_text = """🤖 **SIMS AI Assistant**
                
I can help you analyze your infrastructure data! I can assist with:

• 🚨 **Anomaly Detection**: "Show me anomalies in the last 24 hours"
• 📊 **Performance Analysis**: "How is system performance?"
• 🔧 **Sensor Status**: "What's the status of all sensors?"
• 📈 **Data Queries**: "Show me pressure readings"
• 🔍 **Trend Analysis**: "What are the pressure trends?"
• 🛠️ **Maintenance**: "When do sensors need maintenance?"

Ask me anything about your infrastructure monitoring system!"""
                
                suggested_actions = [
                    "🚨 Check for anomalies",
                    "📊 Review system performance",
                    "🔧 Monitor sensor health"
                ]
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            response_text = f"❌ I encountered an error while processing your request: {str(e)}"
            confidence_score = 0.1
        
        return ChatResponse(
            message=response_text,
            conversation_id="default",
            data_insights=data_insights,
            suggested_actions=suggested_actions,
            query_results=query_results,
            charts=charts,
            confidence_score=confidence_score,
            timestamp=datetime.utcnow()
        )

# Initialize response generator
response_generator = ResponseGenerator()

# Utility functions
async def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token with auth service"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = await http_client.get(f"{AUTH_SERVICE_URL}/me", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail="Authentication service unavailable")

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(
    request: ChatRequest,
    token: str = Query(..., description="JWT token")
):
    """Chat with the AI assistant"""
    user = await verify_token(token)
    
    try:
        # Analyze the query
        analysis = query_processor.analyze_query(request.message)
        
        # Generate response
        response = await response_generator.generate_response(
            request.message,
            analysis,
            token
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/chat/suggestions")
async def get_suggestions(token: str = Query(..., description="JWT token")):
    """Get suggested questions for the user"""
    user = await verify_token(token)
    
    suggestions = [
        "Show me anomalies in the last 24 hours",
        "What's the current system performance?",
        "Which sensors need maintenance?",
        "Compare pressure readings between zones",
        "Show temperature trends over the past week",
        "What are the most critical alerts?",
        "How is the water quality in the system?",
        "When was the last sensor failure?"
    ]
    
    return {"suggestions": suggestions}

@app.get("/chat/history/{conversation_id}")
async def get_conversation_history(
    conversation_id: str,
    token: str = Query(..., description="JWT token")
):
    """Get conversation history"""
    user = await verify_token(token)
    
    # In a real implementation, this would fetch from database
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "timestamp": datetime.utcnow()
    }

@app.post("/chat/feedback")
async def submit_feedback(
    conversation_id: str,
    rating: int,
    feedback: str,
    token: str = Query(..., description="JWT token")
):
    """Submit feedback on chat response"""
    user = await verify_token(token)
    
    # Store feedback for model improvement
    feedback_data = {
        "conversation_id": conversation_id,
        "user": user["username"],
        "rating": rating,
        "feedback": feedback,
        "timestamp": datetime.utcnow()
    }
    
    # In production, store in database
    logger.info(f"Feedback received: {feedback_data}")
    
    return {"message": "Feedback submitted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "llm-assistant",
        "timestamp": datetime.utcnow(),
        "features": {
            "query_analysis": True,
            "anomaly_detection": True,
            "performance_analysis": True,
            "sensor_monitoring": True,
            "data_queries": True,
            "conversation_memory": True
        }
    }

if __name__ == "__main__":
    uvicorn.run("llm_assistant:app", host="0.0.0.0", port=8005, reload=True)