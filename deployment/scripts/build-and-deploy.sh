#!/bin/bash

# SIMS - Build and Deploy Script
# This script builds and deploys the SIMS application

set -e

# Configuration
ENVIRONMENT=${1:-production}
REGION=${2:-us-west-2}
CLUSTER_NAME="sims-cluster"
SERVICES=("auth" "data" "ml" "streaming" "gateway" "llm" "streamlit")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker and try again."
    fi
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed. Please install AWS CLI and try again."
    fi
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed. Please install kubectl and try again."
    fi
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed. Please install Terraform and try again."
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials are not configured. Please run 'aws configure' and try again."
    fi
    
    log "Prerequisites check completed successfully."
}

# Build Docker images
build_images() {
    log "Building Docker images..."
    
    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
    
    # Login to ECR
    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}
    
    # Build and push each service image
    for service in "${SERVICES[@]}"; do
        log "Building ${service} service image..."
        
        # Create ECR repository if it doesn't exist
        aws ecr describe-repositories --repository-names "sims/${service}" --region ${REGION} || \
        aws ecr create-repository --repository-name "sims/${service}" --region ${REGION}
        
        # Build image
        if [ "${service}" == "streamlit" ]; then
            docker build -t sims/${service}:latest -f Dockerfile.streamlit .
        else
            docker build -t sims/${service}:latest -f services/Dockerfile.${service} .
        fi
        
        # Tag and push image
        docker tag sims/${service}:latest ${ECR_REGISTRY}/sims/${service}:latest
        docker push ${ECR_REGISTRY}/sims/${service}:latest
        
        log "${service} image built and pushed successfully."
    done
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."
    
    cd deployment/cloud/aws/terraform
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -var="environment=${ENVIRONMENT}" -var="aws_region=${REGION}" -out=tfplan
    
    # Apply deployment
    terraform apply tfplan
    
    log "Infrastructure deployed successfully."
    cd ../../../..
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Update kubeconfig
    aws eks update-kubeconfig --region ${REGION} --name ${CLUSTER_NAME}
    
    # Apply Kubernetes manifests
    kubectl apply -f deployment/kubernetes/namespace.yaml
    kubectl apply -f deployment/kubernetes/configmap.yaml
    kubectl apply -f deployment/kubernetes/postgres.yaml
    kubectl apply -f deployment/kubernetes/redis.yaml
    kubectl apply -f deployment/kubernetes/kafka.yaml
    
    # Wait for infrastructure to be ready
    kubectl wait --for=condition=ready pod -l app=postgres -n sims --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n sims --timeout=300s
    kubectl wait --for=condition=ready pod -l app=kafka -n sims --timeout=300s
    
    # Deploy services
    kubectl apply -f deployment/kubernetes/services.yaml
    
    # Wait for services to be ready
    kubectl wait --for=condition=ready pod -l app=auth-service -n sims --timeout=300s
    kubectl wait --for=condition=ready pod -l app=data-service -n sims --timeout=300s
    kubectl wait --for=condition=ready pod -l app=ml-service -n sims --timeout=300s
    kubectl wait --for=condition=ready pod -l app=streaming-service -n sims --timeout=300s
    kubectl wait --for=condition=ready pod -l app=gateway-service -n sims --timeout=300s
    
    log "Kubernetes deployment completed successfully."
}

# Deploy with Docker Compose (for local/development)
deploy_with_compose() {
    log "Deploying with Docker Compose..."
    
    # Build images locally
    docker-compose -f deployment/docker-compose.yml build
    
    # Start services
    docker-compose -f deployment/docker-compose.yml up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    for service in "${SERVICES[@]}"; do
        if [ "${service}" == "streamlit" ]; then
            continue
        fi
        
        port=""
        case ${service} in
            "auth") port="8001" ;;
            "data") port="8002" ;;
            "ml") port="8003" ;;
            "streaming") port="8004" ;;
            "gateway") port="8000" ;;
            "llm") port="8005" ;;
        esac
        
        if [ -n "$port" ]; then
            curl -f http://localhost:${port}/health || warn "${service} service is not healthy"
        fi
    done
    
    log "Docker Compose deployment completed successfully."
}

# Initialize database
initialize_database() {
    log "Initializing database..."
    
    # Run database initialization
    python init_database.py
    
    log "Database initialization completed successfully."
}

# Run tests
run_tests() {
    log "Running tests..."
    
    # Run feature verification
    python feature_verification.py
    
    # Run comprehensive tests
    python test_all_features.py
    
    log "Tests completed successfully."
}

# Main deployment function
main() {
    log "Starting SIMS deployment..."
    log "Environment: ${ENVIRONMENT}"
    log "Region: ${REGION}"
    
    # Check prerequisites
    check_prerequisites
    
    # Choose deployment method
    case ${ENVIRONMENT} in
        "production"|"staging")
            log "Deploying to cloud environment..."
            build_images
            deploy_infrastructure
            deploy_to_kubernetes
            ;;
        "development"|"local")
            log "Deploying to local environment..."
            deploy_with_compose
            ;;
        *)
            error "Unknown environment: ${ENVIRONMENT}. Please use 'production', 'staging', 'development', or 'local'."
            ;;
    esac
    
    # Initialize database
    initialize_database
    
    # Run tests
    run_tests
    
    log "SIMS deployment completed successfully!"
    log "Access the application at:"
    
    if [ "${ENVIRONMENT}" == "local" ] || [ "${ENVIRONMENT}" == "development" ]; then
        log "  - Frontend: http://localhost:5000"
        log "  - API Gateway: http://localhost:8000"
        log "  - Prometheus: http://localhost:9090"
        log "  - Grafana: http://localhost:3000"
    else
        # Get load balancer URL
        LB_URL=$(terraform -chdir=deployment/cloud/aws/terraform output -raw load_balancer_dns)
        log "  - Application: http://${LB_URL}"
        log "  - API: http://${LB_URL}/api"
    fi
}

# Run main function
main "$@"