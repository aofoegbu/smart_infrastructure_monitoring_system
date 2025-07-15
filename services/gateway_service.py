"""
API Gateway Service - REST/gRPC Gateway
Central gateway for routing requests to microservices
"""

from datetime import datetime
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import httpx
import grpc
import uvicorn
from pydantic import BaseModel
import redis
from prometheus_client import Counter, Histogram, generate_latest
import time

# Configuration
AUTH_SERVICE_URL = "http://localhost:8001"
DATA_SERVICE_URL = "http://localhost:8002"
ML_SERVICE_URL = "http://localhost:8003"
STREAMING_SERVICE_URL = "http://localhost:8004"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SIMS API Gateway",
    version="1.0.0",
    description="Central API Gateway for SIMS microservices"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis client for caching
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=4)

# HTTP client for service communication
http_client = httpx.AsyncClient(timeout=30.0)

# Security
security = HTTPBearer()

# Prometheus metrics
REQUEST_COUNT = Counter('gateway_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('gateway_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
SERVICE_REQUESTS = Counter('gateway_service_requests_total', 'Service requests', ['service', 'endpoint', 'status'])

# Pydantic Models
class ServiceRoute(BaseModel):
    path: str
    service_url: str
    method: str
    auth_required: bool = True
    cache_ttl: int = 0
    rate_limit: int = 100

class HealthStatus(BaseModel):
    service: str
    status: str
    timestamp: datetime
    response_time: float
    dependencies: Dict[str, str]

class GatewayMetrics(BaseModel):
    total_requests: int
    avg_response_time: float
    error_rate: float
    active_services: int
    cache_hit_rate: float

# Service routing configuration
SERVICE_ROUTES = {
    # Authentication Service
    "/auth/register": ServiceRoute(path="/register", service_url=AUTH_SERVICE_URL, method="POST", auth_required=False),
    "/auth/login": ServiceRoute(path="/login", service_url=AUTH_SERVICE_URL, method="POST", auth_required=False),
    "/auth/refresh": ServiceRoute(path="/refresh", service_url=AUTH_SERVICE_URL, method="POST", auth_required=False),
    "/auth/logout": ServiceRoute(path="/logout", service_url=AUTH_SERVICE_URL, method="POST"),
    "/auth/me": ServiceRoute(path="/me", service_url=AUTH_SERVICE_URL, method="GET", cache_ttl=300),
    
    # Data Service
    "/data/sensors": ServiceRoute(path="/sensors", service_url=DATA_SERVICE_URL, method="GET", cache_ttl=60),
    "/data/sensors/{sensor_id}/data": ServiceRoute(path="/sensors/{sensor_id}/data", service_url=DATA_SERVICE_URL, method="GET", cache_ttl=30),
    "/data/analytics/summary": ServiceRoute(path="/analytics/summary", service_url=DATA_SERVICE_URL, method="GET", cache_ttl=300),
    "/data/query": ServiceRoute(path="/query", service_url=DATA_SERVICE_URL, method="POST"),
    
    # ML Service
    "/ml/models/train": ServiceRoute(path="/models/train", service_url=ML_SERVICE_URL, method="POST"),
    "/ml/models/predict": ServiceRoute(path="/models/predict", service_url=ML_SERVICE_URL, method="POST"),
    "/ml/anomaly-detection": ServiceRoute(path="/anomaly-detection", service_url=ML_SERVICE_URL, method="POST"),
    "/ml/models": ServiceRoute(path="/models", service_url=ML_SERVICE_URL, method="GET", cache_ttl=300),
    "/ml/chat": ServiceRoute(path="/chat", service_url=ML_SERVICE_URL, method="POST"),
    
    # Streaming Service
    "/stream/publish": ServiceRoute(path="/stream/publish", service_url=STREAMING_SERVICE_URL, method="POST"),
    "/stream/topics": ServiceRoute(path="/stream/topics", service_url=STREAMING_SERVICE_URL, method="GET", cache_ttl=300),
    "/stream/metrics": ServiceRoute(path="/stream/metrics", service_url=STREAMING_SERVICE_URL, method="GET", cache_ttl=60),
    "/stream/latest/{topic}": ServiceRoute(path="/stream/latest/{topic}", service_url=STREAMING_SERVICE_URL, method="GET", cache_ttl=30),
    "/mqtt/publish": ServiceRoute(path="/mqtt/publish", service_url=STREAMING_SERVICE_URL, method="POST"),
}

# Rate limiting
class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit: int, window: int = 60) -> bool:
        """Check if request is within rate limit"""
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, window)
            results = pipe.execute()
            
            current_count = results[0]
            return current_count <= limit
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return True  # Allow request if Redis is down

rate_limiter = RateLimiter(redis_client)

# Utility functions
def get_cache_key(path: str, query_params: str, user_id: str) -> str:
    """Generate cache key for request"""
    return f"gateway_cache:{path}:{query_params}:{user_id}"

async def get_cached_response(key: str) -> Optional[Dict[str, Any]]:
    """Get cached response"""
    try:
        cached = redis_client.get(key)
        if cached:
            return json.loads(cached)
    except Exception as e:
        logger.error(f"Cache get error: {e}")
    return None

async def set_cached_response(key: str, response: Dict[str, Any], ttl: int):
    """Set cached response"""
    try:
        redis_client.setex(key, ttl, json.dumps(response, default=str))
    except Exception as e:
        logger.error(f"Cache set error: {e}")

async def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = await http_client.get(f"{AUTH_SERVICE_URL}/me", headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(status_code=401, detail="Authentication service unavailable")

async def forward_request(
    service_url: str,
    path: str,
    method: str,
    headers: Dict[str, str],
    params: Dict[str, Any],
    body: Optional[bytes] = None
) -> httpx.Response:
    """Forward request to service"""
    url = f"{service_url}{path}"
    
    try:
        if method.upper() == "GET":
            response = await http_client.get(url, headers=headers, params=params)
        elif method.upper() == "POST":
            response = await http_client.post(url, headers=headers, params=params, content=body)
        elif method.upper() == "PUT":
            response = await http_client.put(url, headers=headers, params=params, content=body)
        elif method.upper() == "DELETE":
            response = await http_client.delete(url, headers=headers, params=params)
        else:
            raise HTTPException(status_code=405, detail="Method not allowed")
        
        return response
        
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Service unavailable")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Service timeout")
    except Exception as e:
        logger.error(f"Request forwarding error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics for requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

# Generic route handler
async def handle_route(request: Request, route_config: ServiceRoute):
    """Handle generic route forwarding"""
    start_time = time.time()
    
    try:
        # Extract token if auth required
        user = None
        token = None
        
        if route_config.auth_required:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
            
            token = auth_header.split(" ")[1]
            user = await verify_token(token)
        
        # Rate limiting
        if user:
            rate_limit_key = f"rate_limit:{user['username']}:{request.url.path}"
            if not await rate_limiter.is_allowed(rate_limit_key, route_config.rate_limit):
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Check cache
        cached_response = None
        if route_config.cache_ttl > 0 and request.method == "GET":
            cache_key = get_cache_key(
                request.url.path,
                str(request.query_params),
                user["username"] if user else "anonymous"
            )
            cached_response = await get_cached_response(cache_key)
            
            if cached_response:
                return cached_response
        
        # Prepare request
        headers = dict(request.headers)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        # Get request body
        body = await request.body() if request.method in ["POST", "PUT"] else None
        
        # Forward request
        response = await forward_request(
            route_config.service_url,
            route_config.path,
            request.method,
            headers,
            dict(request.query_params),
            body
        )
        
        # Record service metrics
        SERVICE_REQUESTS.labels(
            service=route_config.service_url.split("//")[1].split(":")[0],
            endpoint=route_config.path,
            status=response.status_code
        ).inc()
        
        # Handle response
        if response.status_code >= 400:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        response_data = response.json()
        
        # Cache response
        if route_config.cache_ttl > 0 and request.method == "GET":
            cache_key = get_cache_key(
                request.url.path,
                str(request.query_params),
                user["username"] if user else "anonymous"
            )
            await set_cached_response(cache_key, response_data, route_config.cache_ttl)
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Route handling error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Dynamic route registration
for route_path, route_config in SERVICE_ROUTES.items():
    # Register route with FastAPI
    if route_config.method.upper() == "GET":
        app.get(route_path)(lambda request: handle_route(request, route_config))
    elif route_config.method.upper() == "POST":
        app.post(route_path)(lambda request: handle_route(request, route_config))
    elif route_config.method.upper() == "PUT":
        app.put(route_path)(lambda request: handle_route(request, route_config))
    elif route_config.method.upper() == "DELETE":
        app.delete(route_path)(lambda request: handle_route(request, route_config))

# Gateway-specific endpoints
@app.get("/gateway/health")
async def gateway_health():
    """Gateway health check"""
    service_health = {}
    
    # Check all services
    services = {
        "auth": AUTH_SERVICE_URL,
        "data": DATA_SERVICE_URL,
        "ml": ML_SERVICE_URL,
        "streaming": STREAMING_SERVICE_URL
    }
    
    for service_name, service_url in services.items():
        try:
            start_time = time.time()
            response = await http_client.get(f"{service_url}/health", timeout=5.0)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                service_health[service_name] = {
                    "status": "healthy",
                    "response_time": response_time,
                    "details": response.json()
                }
            else:
                service_health[service_name] = {
                    "status": "unhealthy",
                    "response_time": response_time,
                    "error": response.text
                }
        except Exception as e:
            service_health[service_name] = {
                "status": "unavailable",
                "error": str(e)
            }
    
    # Overall health
    healthy_services = sum(1 for s in service_health.values() if s["status"] == "healthy")
    overall_status = "healthy" if healthy_services >= 3 else "degraded" if healthy_services >= 2 else "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow(),
        "services": service_health,
        "gateway_info": {
            "version": "1.0.0",
            "routes_registered": len(SERVICE_ROUTES),
            "redis_connected": redis_client.ping()
        }
    }

@app.get("/gateway/metrics")
async def gateway_metrics():
    """Gateway metrics endpoint"""
    try:
        # Calculate cache hit rate
        cache_hits = redis_client.get("gateway_cache_hits") or 0
        cache_misses = redis_client.get("gateway_cache_misses") or 0
        total_cache_requests = int(cache_hits) + int(cache_misses)
        cache_hit_rate = int(cache_hits) / total_cache_requests if total_cache_requests > 0 else 0
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "total_cache_requests": total_cache_requests,
            "registered_routes": len(SERVICE_ROUTES),
            "redis_connected": redis_client.ping(),
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {"error": str(e)}

@app.get("/gateway/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/gateway/routes")
async def list_routes():
    """List all registered routes"""
    routes = []
    for path, config in SERVICE_ROUTES.items():
        routes.append({
            "path": path,
            "method": config.method,
            "service_url": config.service_url,
            "auth_required": config.auth_required,
            "cache_ttl": config.cache_ttl,
            "rate_limit": config.rate_limit
        })
    
    return {"routes": routes}

# WebSocket proxy for streaming
@app.websocket("/ws/stream/subscribe/{topic}")
async def websocket_proxy(websocket, topic: str, token: str):
    """WebSocket proxy for streaming service"""
    try:
        await websocket.accept()
        
        # Forward to streaming service
        streaming_ws_url = f"ws://localhost:8004/stream/subscribe/{topic}?token={token}"
        
        async with websockets.connect(streaming_ws_url) as streaming_ws:
            # Proxy messages bidirectionally
            async def forward_to_client():
                async for message in streaming_ws:
                    await websocket.send_text(message)
            
            async def forward_to_streaming():
                while True:
                    message = await websocket.receive_text()
                    await streaming_ws.send(message)
            
            # Run both directions concurrently
            await asyncio.gather(
                forward_to_client(),
                forward_to_streaming()
            )
    
    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("gateway_service:app", host="0.0.0.0", port=8000, reload=True)