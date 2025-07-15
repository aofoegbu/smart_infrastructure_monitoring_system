#!/bin/bash

# SIMS Local Development Startup Script
# This script starts all services for local development

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] $1${NC}"
}

# Set environment variables
export DATABASE_URL="postgresql://user:password@localhost/sims"
export JWT_SECRET_KEY="development-secret-key-change-in-production"
export REDIS_HOST="localhost"
export KAFKA_BOOTSTRAP_SERVERS="localhost:9092"
export MQTT_BROKER="localhost"
export AUTH_SERVICE_URL="http://localhost:8001"
export DATA_SERVICE_URL="http://localhost:8002"
export ML_SERVICE_URL="http://localhost:8003"
export STREAMING_SERVICE_URL="http://localhost:8004"
export LLM_SERVICE_URL="http://localhost:8005"
export GATEWAY_SERVICE_URL="http://localhost:8000"

log "Starting SIMS Local Development Environment..."

# Check if services are already running
check_service() {
    local service_name=$1
    local port=$2
    
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" | grep -q "200\|404"; then
        warn "$service_name is already running on port $port"
        return 0
    else
        return 1
    fi
}

# Start services in development mode
start_services() {
    log "Starting core services..."
    
    # Start services in background
    if ! check_service "Auth Service" 8001; then
        log "Starting Auth Service..."
        cd services && python auth_service.py &
        AUTH_PID=$!
        cd ..
    fi
    
    if ! check_service "Data Service" 8002; then
        log "Starting Data Service..."
        cd services && python data_service.py &
        DATA_PID=$!
        cd ..
    fi
    
    if ! check_service "ML Service" 8003; then
        log "Starting ML Service..."
        cd services && python ml_service.py &
        ML_PID=$!
        cd ..
    fi
    
    if ! check_service "Streaming Service" 8004; then
        log "Starting Streaming Service..."
        cd services && python streaming_service.py &
        STREAMING_PID=$!
        cd ..
    fi
    
    if ! check_service "LLM Assistant" 8005; then
        log "Starting LLM Assistant..."
        cd services && python llm_assistant.py &
        LLM_PID=$!
        cd ..
    fi
    
    # Wait for services to start
    sleep 5
    
    if ! check_service "Gateway Service" 8000; then
        log "Starting Gateway Service..."
        cd services && python gateway_service.py &
        GATEWAY_PID=$!
        cd ..
    fi
    
    # Wait for gateway to start
    sleep 5
    
    log "Services started successfully!"
}

# Check service health
check_health() {
    log "Checking service health..."
    
    local services=("auth:8001" "data:8002" "ml:8003" "streaming:8004" "llm:8005" "gateway:8000")
    
    for service in "${services[@]}"; do
        IFS=':' read -ra SERVICE_INFO <<< "$service"
        service_name=${SERVICE_INFO[0]}
        port=${SERVICE_INFO[1]}
        
        if curl -s -f "http://localhost:$port/health" > /dev/null 2>&1; then
            log "✓ $service_name service is healthy"
        else
            warn "✗ $service_name service is not responding"
        fi
    done
}

# Show service URLs
show_urls() {
    log "Service URLs:"
    echo "  📱 Streamlit App:     http://localhost:5000"
    echo "  🌐 API Gateway:       http://localhost:8000"
    echo "  🔐 Auth Service:      http://localhost:8001"
    echo "  📊 Data Service:      http://localhost:8002"
    echo "  🤖 ML Service:        http://localhost:8003"
    echo "  📡 Streaming Service: http://localhost:8004"
    echo "  💬 LLM Assistant:     http://localhost:8005"
    echo ""
    echo "  📋 API Documentation:"
    echo "    - Gateway: http://localhost:8000/docs"
    echo "    - Auth:    http://localhost:8001/docs"
    echo "    - Data:    http://localhost:8002/docs"
    echo "    - ML:      http://localhost:8003/docs"
    echo "    - Stream:  http://localhost:8004/docs"
    echo "    - LLM:     http://localhost:8005/docs"
}

# Test basic functionality
test_functionality() {
    log "Testing basic functionality..."
    
    # Test gateway health
    if curl -s -f "http://localhost:8000/gateway/health" > /dev/null; then
        log "✓ Gateway health check passed"
    else
        warn "✗ Gateway health check failed"
    fi
    
    # Test authentication
    if curl -s -f "http://localhost:8001/health" > /dev/null; then
        log "✓ Authentication service accessible"
    else
        warn "✗ Authentication service not accessible"
    fi
    
    log "Basic functionality tests completed"
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    
    # Kill background processes
    if [ ! -z "$AUTH_PID" ]; then kill $AUTH_PID 2>/dev/null || true; fi
    if [ ! -z "$DATA_PID" ]; then kill $DATA_PID 2>/dev/null || true; fi
    if [ ! -z "$ML_PID" ]; then kill $ML_PID 2>/dev/null || true; fi
    if [ ! -z "$STREAMING_PID" ]; then kill $STREAMING_PID 2>/dev/null || true; fi
    if [ ! -z "$LLM_PID" ]; then kill $LLM_PID 2>/dev/null || true; fi
    if [ ! -z "$GATEWAY_PID" ]; then kill $GATEWAY_PID 2>/dev/null || true; fi
    
    log "Cleanup completed"
}

# Set up signal handlers
trap cleanup EXIT INT TERM

# Main execution
main() {
    case "${1:-start}" in
        "start")
            start_services
            check_health
            show_urls
            test_functionality
            
            log "SIMS development environment is ready!"
            log "Press Ctrl+C to stop all services"
            
            # Keep script running
            while true; do
                sleep 10
            done
            ;;
        "stop")
            cleanup
            ;;
        "health")
            check_health
            ;;
        "urls")
            show_urls
            ;;
        *)
            echo "Usage: $0 {start|stop|health|urls}"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"