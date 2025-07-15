provider "aws" {
  region = var.aws_region
}

# VPC and Networking
resource "aws_vpc" "sims_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "sims-vpc"
    Environment = var.environment
  }
}

resource "aws_internet_gateway" "sims_igw" {
  vpc_id = aws_vpc.sims_vpc.id

  tags = {
    Name = "sims-igw"
    Environment = var.environment
  }
}

resource "aws_subnet" "sims_public_subnet" {
  count = 2
  vpc_id = aws_vpc.sims_vpc.id
  cidr_block = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "sims-public-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

resource "aws_subnet" "sims_private_subnet" {
  count = 2
  vpc_id = aws_vpc.sims_vpc.id
  cidr_block = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "sims-private-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

# Route Tables
resource "aws_route_table" "sims_public_rt" {
  vpc_id = aws_vpc.sims_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.sims_igw.id
  }

  tags = {
    Name = "sims-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "sims_public_rta" {
  count = 2
  subnet_id = aws_subnet.sims_public_subnet[count.index].id
  route_table_id = aws_route_table.sims_public_rt.id
}

# Security Groups
resource "aws_security_group" "sims_alb_sg" {
  name_prefix = "sims-alb-sg"
  vpc_id = aws_vpc.sims_vpc.id

  ingress {
    from_port = 80
    to_port = 80
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "sims-alb-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "sims_ecs_sg" {
  name_prefix = "sims-ecs-sg"
  vpc_id = aws_vpc.sims_vpc.id

  ingress {
    from_port = 0
    to_port = 65535
    protocol = "tcp"
    security_groups = [aws_security_group.sims_alb_sg.id]
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "sims-ecs-sg"
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "sims_cluster" {
  name = "sims-cluster"

  setting {
    name = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "sims-cluster"
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "sims_alb" {
  name = "sims-alb"
  internal = false
  load_balancer_type = "application"
  security_groups = [aws_security_group.sims_alb_sg.id]
  subnets = aws_subnet.sims_public_subnet[*].id

  enable_deletion_protection = false

  tags = {
    Name = "sims-alb"
    Environment = var.environment
  }
}

# Target Groups
resource "aws_lb_target_group" "sims_gateway_tg" {
  name = "sims-gateway-tg"
  port = 8000
  protocol = "HTTP"
  vpc_id = aws_vpc.sims_vpc.id
  target_type = "ip"

  health_check {
    enabled = true
    healthy_threshold = 2
    interval = 30
    matcher = "200"
    path = "/gateway/health"
    port = "traffic-port"
    protocol = "HTTP"
    timeout = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "sims-gateway-tg"
    Environment = var.environment
  }
}

resource "aws_lb_target_group" "sims_streamlit_tg" {
  name = "sims-streamlit-tg"
  port = 5000
  protocol = "HTTP"
  vpc_id = aws_vpc.sims_vpc.id
  target_type = "ip"

  health_check {
    enabled = true
    healthy_threshold = 2
    interval = 30
    matcher = "200"
    path = "/"
    port = "traffic-port"
    protocol = "HTTP"
    timeout = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "sims-streamlit-tg"
    Environment = var.environment
  }
}

# Load Balancer Listeners
resource "aws_lb_listener" "sims_listener" {
  load_balancer_arn = aws_lb.sims_alb.arn
  port = "80"
  protocol = "HTTP"

  default_action {
    type = "forward"
    target_group_arn = aws_lb_target_group.sims_streamlit_tg.arn
  }
}

resource "aws_lb_listener_rule" "sims_api_rule" {
  listener_arn = aws_lb_listener.sims_listener.arn
  priority = 100

  action {
    type = "forward"
    target_group_arn = aws_lb_target_group.sims_gateway_tg.arn
  }

  condition {
    path_pattern {
      values = ["/api/*"]
    }
  }
}

# RDS Database
resource "aws_db_subnet_group" "sims_db_subnet_group" {
  name = "sims-db-subnet-group"
  subnet_ids = aws_subnet.sims_private_subnet[*].id

  tags = {
    Name = "sims-db-subnet-group"
    Environment = var.environment
  }
}

resource "aws_security_group" "sims_rds_sg" {
  name_prefix = "sims-rds-sg"
  vpc_id = aws_vpc.sims_vpc.id

  ingress {
    from_port = 5432
    to_port = 5432
    protocol = "tcp"
    security_groups = [aws_security_group.sims_ecs_sg.id]
  }

  tags = {
    Name = "sims-rds-sg"
    Environment = var.environment
  }
}

resource "aws_db_instance" "sims_postgres" {
  identifier = "sims-postgres"
  engine = "postgres"
  engine_version = "15.7"
  instance_class = "db.t3.micro"
  allocated_storage = 20
  storage_type = "gp2"
  storage_encrypted = true

  db_name = "sims"
  username = "sims_user"
  password = var.db_password

  vpc_security_group_ids = [aws_security_group.sims_rds_sg.id]
  db_subnet_group_name = aws_db_subnet_group.sims_db_subnet_group.name

  backup_retention_period = 7
  backup_window = "03:00-04:00"
  maintenance_window = "sun:04:00-sun:05:00"

  skip_final_snapshot = true

  tags = {
    Name = "sims-postgres"
    Environment = var.environment
  }
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "sims_redis_subnet_group" {
  name = "sims-redis-subnet-group"
  subnet_ids = aws_subnet.sims_private_subnet[*].id

  tags = {
    Name = "sims-redis-subnet-group"
    Environment = var.environment
  }
}

resource "aws_security_group" "sims_redis_sg" {
  name_prefix = "sims-redis-sg"
  vpc_id = aws_vpc.sims_vpc.id

  ingress {
    from_port = 6379
    to_port = 6379
    protocol = "tcp"
    security_groups = [aws_security_group.sims_ecs_sg.id]
  }

  tags = {
    Name = "sims-redis-sg"
    Environment = var.environment
  }
}

resource "aws_elasticache_cluster" "sims_redis" {
  cluster_id = "sims-redis"
  engine = "redis"
  node_type = "cache.t3.micro"
  num_cache_nodes = 1
  parameter_group_name = "default.redis7"
  port = 6379
  subnet_group_name = aws_elasticache_subnet_group.sims_redis_subnet_group.name
  security_group_ids = [aws_security_group.sims_redis_sg.id]

  tags = {
    Name = "sims-redis"
    Environment = var.environment
  }
}

# Amazon MSK (Kafka)
resource "aws_msk_cluster" "sims_kafka" {
  cluster_name = "sims-kafka"
  kafka_version = "2.8.1"
  number_of_broker_nodes = 2

  broker_node_group_info {
    instance_type = "kafka.t3.small"
    client_subnets = aws_subnet.sims_private_subnet[*].id
    storage_info {
      ebs_storage_info {
        volume_size = 20
      }
    }
    security_groups = [aws_security_group.sims_kafka_sg.id]
  }

  encryption_info {
    encryption_in_transit {
      client_broker = "PLAINTEXT"
    }
  }

  tags = {
    Name = "sims-kafka"
    Environment = var.environment
  }
}

resource "aws_security_group" "sims_kafka_sg" {
  name_prefix = "sims-kafka-sg"
  vpc_id = aws_vpc.sims_vpc.id

  ingress {
    from_port = 9092
    to_port = 9092
    protocol = "tcp"
    security_groups = [aws_security_group.sims_ecs_sg.id]
  }

  tags = {
    Name = "sims-kafka-sg"
    Environment = var.environment
  }
}

# ECS Task Definitions and Services
resource "aws_ecs_task_definition" "sims_gateway" {
  family = "sims-gateway"
  network_mode = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu = "256"
  memory = "512"
  execution_role_arn = aws_iam_role.ecs_execution_role.arn
  task_role_arn = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    {
      name = "sims-gateway"
      image = "${aws_ecr_repository.sims_gateway.repository_url}:latest"
      portMappings = [
        {
          containerPort = 8000
          protocol = "tcp"
        }
      ]
      environment = [
        {
          name = "AUTH_SERVICE_URL"
          value = "http://sims-auth:8001"
        },
        {
          name = "DATA_SERVICE_URL"
          value = "http://sims-data:8002"
        },
        {
          name = "ML_SERVICE_URL"
          value = "http://sims-ml:8003"
        },
        {
          name = "STREAMING_SERVICE_URL"
          value = "http://sims-streaming:8004"
        },
        {
          name = "REDIS_HOST"
          value = aws_elasticache_cluster.sims_redis.cache_nodes[0].address
        }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group" = "/ecs/sims-gateway"
          "awslogs-region" = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])
}

# IAM Roles
resource "aws_iam_role" "ecs_execution_role" {
  name = "sims-ecs-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_role_policy" {
  role = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "ecs_task_role" {
  name = "sims-ecs-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# ECR Repositories
resource "aws_ecr_repository" "sims_gateway" {
  name = "sims/gateway"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Name = "sims-gateway"
    Environment = var.environment
  }
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "sims_gateway_logs" {
  name = "/ecs/sims-gateway"
  retention_in_days = 30

  tags = {
    Name = "sims-gateway-logs"
    Environment = var.environment
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}