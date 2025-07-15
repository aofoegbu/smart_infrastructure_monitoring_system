# SIMS - Smart Infrastructure Monitoring System

## Overview

SIMS is a comprehensive Smart Infrastructure Monitoring System built with Streamlit for monitoring and optimizing utility networks (water, energy, telecom). The system provides real-time sensor data visualization, anomaly detection, analytics, and data management capabilities in a web-based dashboard interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (July 15, 2025)

### Navigation and UI Updates
- Added emoji icons to all navigation pages (🏠 Home, 📊 Dashboard, 🗺️ Infrastructure Map, etc.)
- Renamed "app" to "Home" in navigation
- Fixed navigation links to work with icon-based filenames
- Fixed deprecation warnings (updated 'T' to 'min' in pandas, replaced density_mapbox with scatter_mapbox)

### System Verification
- All major features verified and working:
  - Database connectivity and data persistence (PostgreSQL)
  - User authentication and authorization system
  - Real-time sensor data generation and storage
  - ML-based anomaly detection with Isolation Forest
  - Data quality assessment and monitoring
  - Data governance and compliance management
  - Infrastructure and asset management
  - Predictive analytics and maintenance scoring
  - Alert system and notifications
  - Multi-page Streamlit web interface

### Feature Status
- ✅ Database: 50 sensors, 25 assets, 87+ sensor readings, 2+ active alerts
- ✅ Authentication: Role-based access control with 4 user levels
- ✅ Real-time Processing: Live data generation and storage
- ✅ ML Analytics: Anomaly detection, predictive maintenance, health scoring
- ✅ Data Quality: Comprehensive quality checks and reporting
- ✅ Governance: GDPR, SOX, ISO27001, HIPAA compliance frameworks
- ✅ Web Interface: All 7 pages functional with proper navigation

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