"""
Streaming Service - Apache Kafka/MQTT Integration
Microservice for real-time data streaming and processing
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import asyncio
import threading
import logging
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import uvicorn
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
from kafka.errors import TopicAlreadyExistsError
import paho.mqtt.client as mqtt_client
from concurrent.futures import ThreadPoolExecutor
import websockets
import requests

# Configuration
KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]
REDIS_HOST = "localhost"
REDIS_PORT = 6379
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
AUTH_SERVICE_URL = "http://localhost:8001"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="SIMS Streaming Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=3)

# Kafka setup
kafka_producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    key_serializer=lambda k: k.encode('utf-8') if k else None
)

# Kafka Admin Client
kafka_admin = KafkaAdminClient(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    client_id="streaming_service_admin"
)

# MQTT Client
mqtt_client_instance = mqtt_client.Client("streaming_service_mqtt")

# Pydantic Models
class StreamData(BaseModel):
    topic: str
    key: Optional[str] = None
    value: Dict[str, Any]
    timestamp: Optional[datetime] = None

class TopicConfig(BaseModel):
    name: str
    partitions: int = 3
    replication_factor: int = 1
    config: Dict[str, str] = {}

class StreamingMetrics(BaseModel):
    topic: str
    messages_produced: int
    messages_consumed: int
    last_message_timestamp: Optional[datetime]
    consumer_lag: int
    partition_count: int

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime

# Data structures
@dataclass
class StreamingState:
    active_consumers: Dict[str, KafkaConsumer] = None
    websocket_connections: Dict[str, WebSocket] = None
    metrics: Dict[str, StreamingMetrics] = None
    
    def __post_init__(self):
        self.active_consumers = {}
        self.websocket_connections = {}
        self.metrics = {}

streaming_state = StreamingState()

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

def create_kafka_topics(topics: List[TopicConfig]):
    """Create Kafka topics if they don't exist"""
    topic_list = []
    for topic_config in topics:
        topic_list.append(NewTopic(
            name=topic_config.name,
            num_partitions=topic_config.partitions,
            replication_factor=topic_config.replication_factor,
            topic_configs=topic_config.config
        ))
    
    try:
        kafka_admin.create_topics(topic_list, validate_only=False)
        logger.info(f"Created topics: {[t.name for t in topic_list]}")
    except TopicAlreadyExistsError:
        logger.info("Topics already exist")
    except Exception as e:
        logger.error(f"Error creating topics: {e}")

def setup_mqtt_client():
    """Setup MQTT client with callbacks"""
    def on_connect(client, userdata, flags, rc):
        logger.info(f"MQTT connected with result code {rc}")
        client.subscribe("sensors/+/data")
        client.subscribe("alerts/+")
    
    def on_message(client, userdata, msg):
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())
            
            # Forward MQTT messages to Kafka
            kafka_topic = topic.replace("/", "_")
            kafka_producer.send(kafka_topic, value=payload)
            
            # Store in Redis for real-time access
            redis_client.setex(f"mqtt:{topic}", 300, json.dumps(payload))
            
            logger.info(f"Forwarded MQTT message from {topic} to Kafka {kafka_topic}")
            
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")
    
    mqtt_client_instance.on_connect = on_connect
    mqtt_client_instance.on_message = on_message
    
    try:
        mqtt_client_instance.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client_instance.loop_start()
    except Exception as e:
        logger.error(f"MQTT connection failed: {e}")

def kafka_consumer_worker(topic: str, websocket_key: str):
    """Background worker for Kafka consumption"""
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        group_id=f"streaming_service_{topic}",
        enable_auto_commit=True,
        auto_offset_reset='latest'
    )
    
    streaming_state.active_consumers[websocket_key] = consumer
    
    try:
        for message in consumer:
            # Update metrics
            if topic not in streaming_state.metrics:
                streaming_state.metrics[topic] = StreamingMetrics(
                    topic=topic,
                    messages_produced=0,
                    messages_consumed=0,
                    last_message_timestamp=None,
                    consumer_lag=0,
                    partition_count=len(consumer.partitions_for_topic(topic) or [])
                )
            
            streaming_state.metrics[topic].messages_consumed += 1
            streaming_state.metrics[topic].last_message_timestamp = datetime.utcnow()
            
            # Send to WebSocket if connected
            if websocket_key in streaming_state.websocket_connections:
                websocket = streaming_state.websocket_connections[websocket_key]
                try:
                    asyncio.run(websocket.send_text(json.dumps({
                        "type": "kafka_message",
                        "topic": topic,
                        "data": message.value,
                        "timestamp": datetime.utcnow().isoformat()
                    })))
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    # Clean up disconnected WebSocket
                    if websocket_key in streaming_state.websocket_connections:
                        del streaming_state.websocket_connections[websocket_key]
                    break
            
            # Store in Redis for caching
            redis_client.setex(f"kafka:{topic}:latest", 300, json.dumps(message.value))
            
    except Exception as e:
        logger.error(f"Error in Kafka consumer worker: {e}")
    finally:
        consumer.close()
        if websocket_key in streaming_state.active_consumers:
            del streaming_state.active_consumers[websocket_key]

# Initialize default topics
default_topics = [
    TopicConfig(name="sensor_data", partitions=5, replication_factor=1),
    TopicConfig(name="alerts", partitions=3, replication_factor=1),
    TopicConfig(name="maintenance", partitions=2, replication_factor=1),
    TopicConfig(name="system_metrics", partitions=2, replication_factor=1),
    TopicConfig(name="user_activities", partitions=2, replication_factor=1)
]

# Create topics and setup MQTT on startup
create_kafka_topics(default_topics)
setup_mqtt_client()

# API Endpoints
@app.post("/stream/publish")
async def publish_to_stream(
    data: StreamData,
    token: str = Query(..., description="JWT token")
):
    """Publish data to Kafka stream"""
    user = await verify_token(token)
    
    try:
        # Add timestamp if not provided
        if not data.timestamp:
            data.timestamp = datetime.utcnow()
        
        # Prepare message
        message_value = {
            **data.value,
            "timestamp": data.timestamp.isoformat(),
            "published_by": user["username"]
        }
        
        # Send to Kafka
        future = kafka_producer.send(
            data.topic,
            key=data.key,
            value=message_value
        )
        
        # Wait for acknowledgment
        record_metadata = future.get(timeout=10)
        
        # Update metrics
        if data.topic not in streaming_state.metrics:
            streaming_state.metrics[data.topic] = StreamingMetrics(
                topic=data.topic,
                messages_produced=0,
                messages_consumed=0,
                last_message_timestamp=None,
                consumer_lag=0,
                partition_count=record_metadata.partition + 1
            )
        
        streaming_state.metrics[data.topic].messages_produced += 1
        
        return {
            "status": "success",
            "topic": data.topic,
            "partition": record_metadata.partition,
            "offset": record_metadata.offset,
            "timestamp": record_metadata.timestamp
        }
        
    except Exception as e:
        logger.error(f"Error publishing to stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish: {str(e)}")

@app.websocket("/stream/subscribe/{topic}")
async def websocket_subscribe(websocket: WebSocket, topic: str, token: str):
    """WebSocket endpoint for real-time streaming"""
    try:
        # Verify token
        user = await verify_token(token)
        
        await websocket.accept()
        
        # Generate unique key for this connection
        websocket_key = f"{user['username']}_{topic}_{datetime.utcnow().timestamp()}"
        streaming_state.websocket_connections[websocket_key] = websocket
        
        # Start Kafka consumer in background thread
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(kafka_consumer_worker, topic, websocket_key)
        
        # Keep connection alive
        try:
            while True:
                # Send heartbeat every 30 seconds
                await asyncio.sleep(30)
                await websocket.send_text(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                }))
        except Exception as e:
            logger.info(f"WebSocket connection closed: {e}")
        finally:
            # Clean up
            if websocket_key in streaming_state.websocket_connections:
                del streaming_state.websocket_connections[websocket_key]
            if websocket_key in streaming_state.active_consumers:
                streaming_state.active_consumers[websocket_key].close()
                del streaming_state.active_consumers[websocket_key]
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=1000)

@app.get("/stream/topics")
async def list_topics(token: str = Query(..., description="JWT token")):
    """List available Kafka topics"""
    user = await verify_token(token)
    
    try:
        metadata = kafka_admin.describe_topics()
        topics = []
        
        for topic_name, topic_metadata in metadata.items():
            topics.append({
                "name": topic_name,
                "partitions": len(topic_metadata.partitions),
                "replication_factor": len(topic_metadata.partitions[0].replicas) if topic_metadata.partitions else 0
            })
        
        return {"topics": topics}
        
    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        raise HTTPException(status_code=500, detail="Failed to list topics")

@app.post("/stream/topics")
async def create_topic(
    topic_config: TopicConfig,
    token: str = Query(..., description="JWT token")
):
    """Create a new Kafka topic"""
    user = await verify_token(token)
    
    try:
        create_kafka_topics([topic_config])
        return {"message": f"Topic {topic_config.name} created successfully"}
        
    except Exception as e:
        logger.error(f"Error creating topic: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create topic: {str(e)}")

@app.get("/stream/metrics")
async def get_streaming_metrics(token: str = Query(..., description="JWT token")):
    """Get streaming metrics"""
    user = await verify_token(token)
    
    return {
        "metrics": streaming_state.metrics,
        "active_consumers": len(streaming_state.active_consumers),
        "websocket_connections": len(streaming_state.websocket_connections),
        "timestamp": datetime.utcnow()
    }

@app.get("/stream/latest/{topic}")
async def get_latest_message(
    topic: str,
    token: str = Query(..., description="JWT token")
):
    """Get the latest message from a topic"""
    user = await verify_token(token)
    
    try:
        # Check Redis cache first
        cached_message = redis_client.get(f"kafka:{topic}:latest")
        if cached_message:
            return {
                "topic": topic,
                "data": json.loads(cached_message),
                "source": "cache"
            }
        
        # If not in cache, create a temporary consumer
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            consumer_timeout_ms=5000,
            auto_offset_reset='latest'
        )
        
        message = None
        for msg in consumer:
            message = msg.value
            break
        
        consumer.close()
        
        if message:
            return {
                "topic": topic,
                "data": message,
                "source": "kafka"
            }
        else:
            return {
                "topic": topic,
                "data": None,
                "message": "No messages available"
            }
            
    except Exception as e:
        logger.error(f"Error getting latest message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get message: {str(e)}")

@app.post("/mqtt/publish")
async def publish_to_mqtt(
    topic: str,
    message: Dict[str, Any],
    token: str = Query(..., description="JWT token")
):
    """Publish message to MQTT broker"""
    user = await verify_token(token)
    
    try:
        # Add metadata
        mqtt_message = {
            **message,
            "timestamp": datetime.utcnow().isoformat(),
            "published_by": user["username"]
        }
        
        # Publish to MQTT
        result = mqtt_client_instance.publish(topic, json.dumps(mqtt_message))
        
        if result.rc == 0:
            return {"status": "success", "topic": topic}
        else:
            raise HTTPException(status_code=500, detail="MQTT publish failed")
            
    except Exception as e:
        logger.error(f"Error publishing to MQTT: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to publish: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    kafka_health = "connected"
    mqtt_health = "connected" if mqtt_client_instance.is_connected() else "disconnected"
    redis_health = "connected" if redis_client.ping() else "disconnected"
    
    return {
        "status": "healthy",
        "service": "streaming-service",
        "timestamp": datetime.utcnow(),
        "dependencies": {
            "kafka": kafka_health,
            "mqtt": mqtt_health,
            "redis": redis_health
        },
        "metrics": {
            "active_consumers": len(streaming_state.active_consumers),
            "websocket_connections": len(streaming_state.websocket_connections),
            "topics_with_metrics": len(streaming_state.metrics)
        }
    }

if __name__ == "__main__":
    uvicorn.run("streaming_service:app", host="0.0.0.0", port=8004, reload=True)