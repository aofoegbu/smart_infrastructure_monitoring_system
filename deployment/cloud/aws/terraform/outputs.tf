output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.sims_vpc.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.sims_public_subnet[*].id
}

output "private_subnet_ids" {
  description = "Private subnet IDs"
  value       = aws_subnet.sims_private_subnet[*].id
}

output "load_balancer_dns" {
  description = "Load balancer DNS name"
  value       = aws_lb.sims_alb.dns_name
}

output "load_balancer_hosted_zone_id" {
  description = "Load balancer hosted zone ID"
  value       = aws_lb.sims_alb.zone_id
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.sims_cluster.name
}

output "database_endpoint" {
  description = "Database endpoint"
  value       = aws_db_instance.sims_postgres.endpoint
  sensitive   = true
}

output "redis_endpoint" {
  description = "Redis endpoint"
  value       = aws_elasticache_cluster.sims_redis.cache_nodes[0].address
  sensitive   = true
}

output "kafka_bootstrap_servers" {
  description = "Kafka bootstrap servers"
  value       = aws_msk_cluster.sims_kafka.bootstrap_brokers
  sensitive   = true
}

output "ecr_repository_urls" {
  description = "ECR repository URLs"
  value = {
    gateway = aws_ecr_repository.sims_gateway.repository_url
  }
}

output "cloudwatch_log_groups" {
  description = "CloudWatch log group names"
  value = {
    gateway = aws_cloudwatch_log_group.sims_gateway_logs.name
  }
}

output "security_group_ids" {
  description = "Security group IDs"
  value = {
    alb = aws_security_group.sims_alb_sg.id
    ecs = aws_security_group.sims_ecs_sg.id
    rds = aws_security_group.sims_rds_sg.id
    redis = aws_security_group.sims_redis_sg.id
    kafka = aws_security_group.sims_kafka_sg.id
  }
}

output "iam_role_arns" {
  description = "IAM role ARNs"
  value = {
    execution_role = aws_iam_role.ecs_execution_role.arn
    task_role = aws_iam_role.ecs_task_role.arn
  }
}