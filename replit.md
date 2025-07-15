# SIMS - Smart Infrastructure Monitoring System

## Overview

SIMS is a comprehensive Smart Infrastructure Monitoring System built with Streamlit for monitoring and optimizing utility networks (water, energy, telecom). The system provides real-time sensor data visualization, anomaly detection, analytics, and data management capabilities in a web-based dashboard interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (July 15, 2025)

### Complete Feature Implementation
- **ALL PLACEHOLDER FEATURES IMPLEMENTED**: Successfully converted all "would be implemented here" placeholders into fully functional features
- **Infrastructure Map**: Added map export (HTML download) and comprehensive zone analysis reporting with geographical analysis
- **Analytics Hub**: Implemented automated report generation and analysis scheduling system with email notifications
- **Anomaly Detection**: Added custom alert creation, notification system configuration, anomaly reporting, model retraining, and dashboard generation
- **Data Management**: Implemented comprehensive policy update interface and full data management reporting
- **Dashboard**: Added system report generation and complete email notification configuration system

### New Functional Features
- **🗺️ Map Export**: Export interactive maps as HTML files with all sensor data
- **📊 Zone Analysis**: Generate geographical zone reports with sensor performance metrics
- **📈 Automated Reports**: Create comprehensive analytics reports with downloadable JSON/CSV formats
- **⏰ Analysis Scheduling**: Schedule recurring analyses with email notifications
- **🚨 Custom Alerts**: Create and manage custom alerts with multiple severity levels
- **📧 Notification System**: Configure email, SMS, and push notifications with threshold settings
- **📋 Policy Management**: Update data governance policies for retention, access control, and compliance
- **🔄 Model Retraining**: Retrain ML models with new data and performance validation
- **📊 Dashboard Generation**: Create interactive anomaly detection dashboards with visualization

### Technical Enhancements
- **Report Generation**: JSON and CSV export capabilities across all modules
- **Email Configuration**: SMTP setup with recipient management and frequency settings
- **Policy Updates**: Comprehensive governance framework with GDPR, SOX, ISO27001, HIPAA compliance
- **Model Management**: ML model retraining with performance metrics and validation
- **Data Exports**: Multiple export formats for maps, reports, and analysis data

### System Status
- ✅ All 15+ placeholder features converted to working implementations
- ✅ Complete reporting system with downloadable formats
- ✅ Full notification and alert management
- ✅ Comprehensive policy and governance management
- ✅ Advanced ML model management and retraining capabilities
- ✅ Interactive dashboard and visualization generation
- ✅ Multi-format data export capabilities

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit - Python-based web application framework
- **Structure**: Multi-page application with modular page components
- **UI Components**: Interactive dashboards, maps, charts, and control panels
- **Visualization**: Plotly for charts and graphs, Folium for interactive maps

### Backend Architecture
- **Language**: Python
- **Architecture Pattern**: Modular utility-based structure
- **Authentication**: Session-based authentication with role-based access control
- **Data Processing**: Real-time data generation and processing utilities
- **ML Integration**: Scikit-learn based anomaly detection and predictive models

### Application Structure
```
├── app.py (main application entry point)
├── pages/ (Streamlit pages)
│   ├── Dashboard.py
│   ├── Infrastructure_Map.py
│   ├── Analytics_Hub.py
│   ├── Anomaly_Detection.py
│   ├── Data_Management.py
│   └── System_Health.py
└── utils/ (utility modules)
    ├── auth.py
    ├── data_generator.py
    ├── data_governance.py
    ├── data_quality.py
    └── ml_models.py
```

## Key Components

### 1. Authentication System (`utils/auth.py`)
- **Purpose**: User authentication and role-based access control
- **Features**: Default user roles (admin, operator, analyst, manager), password hashing, session management
- **Security**: SHA-256 password hashing, permission-based access control

### 2. Data Generation (`utils/data_generator.py`)
- **Purpose**: Simulate real-time sensor data for infrastructure monitoring
- **Features**: Generates realistic sensor readings (pressure, flow, temperature, quality)
- **Data Types**: Pressure sensors, flow meters, temperature sensors, quality monitors

### 3. Data Governance (`utils/data_governance.py`)
- **Purpose**: Data policy management and compliance framework
- **Features**: Data retention policies, classification systems, access control rules
- **Compliance**: GDPR, SOX, ISO27001, HIPAA framework support

### 4. Data Quality Management (`utils/data_quality.py`)
- **Purpose**: Comprehensive data quality assessment and monitoring
- **Features**: Completeness, accuracy, consistency, timeliness, and uniqueness checks
- **Monitoring**: Real-time quality scoring and alerting

### 5. Machine Learning Models (`utils/ml_models.py`)
- **Purpose**: AI-powered anomaly detection and predictive analytics
- **Algorithms**: Isolation Forest, DBSCAN clustering, linear regression
- **Features**: Real-time anomaly detection, predictive maintenance, data preprocessing

### 6. Dashboard Pages
- **Main Dashboard**: System overview and KPIs
- **Infrastructure Map**: Interactive sensor network visualization
- **Analytics Hub**: Advanced data analysis and insights
- **Anomaly Detection**: AI-powered anomaly detection interface
- **Data Management**: Data governance and quality management
- **System Health**: Infrastructure monitoring and performance metrics

## Data Flow

1. **Data Generation**: Simulated sensor data generated in real-time
2. **Data Processing**: Raw sensor data processed and validated
3. **Quality Assessment**: Data quality checks performed automatically
4. **Storage**: Data stored with appropriate governance policies
5. **Analysis**: ML models analyze data for anomalies and patterns
6. **Visualization**: Processed data displayed in interactive dashboards
7. **Alerting**: Anomalies and quality issues trigger alerts

## External Dependencies

### Core Libraries
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualization
- **Folium**: Interactive mapping
- **Scikit-learn**: Machine learning algorithms

### Visualization and Mapping
- **streamlit-folium**: Streamlit-Folium integration
- **plotly.express**: High-level plotting interface
- **plotly.graph_objects**: Low-level plotting interface

### System Monitoring
- **psutil**: System and process utilities (for system health monitoring)

## Deployment Strategy

### Current Setup
- **Platform**: Streamlit application (suitable for Replit deployment)
- **Architecture**: Single-application deployment with modular components
- **Session Management**: Streamlit session state for user authentication
- **Data Storage**: PostgreSQL database with comprehensive data models
- **Database Integration**: Full CRUD operations with SQLAlchemy ORM

### Database Architecture
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Tables**: 
  - sensors (metadata and configuration)
  - sensor_readings (real-time data storage)
  - infrastructure_assets (physical infrastructure)
  - alerts (system notifications)
  - users (authentication and roles)
  - user_activities (audit trail)
  - data_quality_checks (quality monitoring)
  - maintenance_records (maintenance tracking)
- **Features**: Automatic data persistence, real-time storage, historical analysis

### Production Considerations
- **Database**: PostgreSQL database fully integrated and operational
- **Scalability**: Microservices architecture ready (as per project requirements)
- **Cloud Deployment**: Compatible with cloud platforms (AWS, GCP, Azure)
- **Container Support**: Docker-ready structure
- **Real-time Processing**: Database-backed real-time data processing

### Security Features
- Role-based access control with four user levels
- Password hashing with SHA-256
- Session-based authentication
- Data classification and governance policies
- Audit trail capabilities

The system follows a modular design pattern that allows for easy extension and integration with external services, making it suitable for scaling from a prototype to a production infrastructure monitoring system.