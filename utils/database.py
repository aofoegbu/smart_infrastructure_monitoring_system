"""
Database module for Smart Infrastructure Monitoring System (SIMS)
Handles PostgreSQL database operations, models, and data persistence
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid
import logging

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Sensor(Base):
    """Sensor metadata and configuration table"""
    __tablename__ = "sensors"
    
    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String(50), unique=True, index=True, nullable=False)
    sensor_type = Column(String(50), nullable=False)  # pressure, flow, temperature, quality
    location = Column(String(200), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    installation_date = Column(DateTime, nullable=False)
    maintenance_interval = Column(Integer, nullable=False)  # days
    manufacturer = Column(String(100))
    model = Column(String(100))
    status = Column(String(20), default='Active')  # Active, Maintenance, Offline
    calibration_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    readings = relationship("SensorReading", back_populates="sensor")
    alerts = relationship("Alert", back_populates="sensor")

class SensorReading(Base):
    """Real-time sensor data readings table"""
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, index=True)
    sensor_id = Column(String(50), ForeignKey("sensors.sensor_id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    pressure = Column(Float)
    flow_rate = Column(Float)
    temperature = Column(Float)
    quality_score = Column(Float)
    anomaly_score = Column(Float, default=0.0)
    is_anomaly = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sensor = relationship("Sensor", back_populates="readings")

class InfrastructureAsset(Base):
    """Infrastructure assets table (pipelines, pumps, etc.)"""
    __tablename__ = "infrastructure_assets"
    
    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(String(50), unique=True, index=True, nullable=False)
    asset_type = Column(String(50), nullable=False)  # Pipeline, Pump Station, etc.
    name = Column(String(200), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    installation_year = Column(Integer)
    capacity = Column(Float)
    material = Column(String(100))
    condition = Column(String(50))  # Excellent, Good, Fair, Poor
    last_inspection = Column(DateTime)
    next_maintenance = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Alert(Base):
    """System alerts and notifications table"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(50), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    sensor_id = Column(String(50), ForeignKey("sensors.sensor_id"), nullable=True)
    alert_type = Column(String(50), nullable=False)  # anomaly, maintenance, quality, etc.
    severity = Column(String(20), nullable=False)  # critical, warning, info
    title = Column(String(200), nullable=False)
    description = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    status = Column(String(20), default='active')  # active, acknowledged, resolved
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sensor = relationship("Sensor", back_populates="alerts")

class User(Base):
    """User accounts and authentication table"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False)  # admin, operator, analyst, manager
    department = Column(String(100))
    permissions = Column(Text)  # JSON string of permissions
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserActivity(Base):
    """User activity logging table"""
    __tablename__ = "user_activities"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    action = Column(String(100), nullable=False)
    details = Column(Text)
    ip_address = Column(String(45))
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

class DataQualityCheck(Base):
    """Data quality assessment results table"""
    __tablename__ = "data_quality_checks"
    
    id = Column(Integer, primary_key=True, index=True)
    check_id = Column(String(50), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    check_type = Column(String(50), nullable=False)  # completeness, accuracy, etc.
    dataset_name = Column(String(100), nullable=False)
    score = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    details = Column(Text)  # JSON string with detailed results
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

class MaintenanceRecord(Base):
    """Maintenance activities and schedules table"""
    __tablename__ = "maintenance_records"
    
    id = Column(Integer, primary_key=True, index=True)
    record_id = Column(String(50), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    asset_id = Column(String(50), nullable=False)
    maintenance_type = Column(String(100), nullable=False)  # Preventive, Corrective, Emergency
    scheduled_date = Column(DateTime, nullable=False)
    completed_date = Column(DateTime)
    status = Column(String(20), default='scheduled')  # scheduled, in_progress, completed, cancelled
    technician = Column(String(100))
    description = Column(Text)
    cost = Column(Float)
    parts_used = Column(Text)  # JSON string
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Database utility functions
def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

def init_database():
    """Initialize database tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        return False

def populate_initial_data():
    """Populate database with initial sensor and infrastructure data"""
    db = SessionLocal()
    try:
        # Check if data already exists
        existing_sensors = db.query(Sensor).count()
        if existing_sensors > 0:
            logger.info("Database already contains data, skipping initial population")
            return True
        
        logger.info("Populating database with initial data...")
        
        # Load sensor metadata from CSV
        import pandas as pd
        sensors_df = pd.read_csv('data/sensors.csv')
        
        # Insert sensors
        for _, row in sensors_df.iterrows():
            sensor = Sensor(
                sensor_id=row['sensor_id'],
                sensor_type=row['sensor_type'],
                location=row['location'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                installation_date=pd.to_datetime(row['installation_date']),
                maintenance_interval=row['maintenance_interval'],
                manufacturer=row['manufacturer'],
                model=row['model'],
                status=row['status'],
                calibration_date=pd.to_datetime(row['calibration_date'])
            )
            db.add(sensor)
        
        # Load infrastructure assets
        assets_df = pd.read_csv('data/infrastructure_assets.csv')
        
        # Insert infrastructure assets
        for _, row in assets_df.iterrows():
            asset = InfrastructureAsset(
                asset_id=row['asset_id'],
                asset_type=row['asset_type'],
                name=row['name'],
                latitude=row['latitude'],
                longitude=row['longitude'],
                installation_year=row['installation_year'],
                capacity=row['capacity'],
                material=row['material'],
                condition=row['condition'],
                last_inspection=pd.to_datetime(row['last_inspection']),
                next_maintenance=pd.to_datetime(row['next_maintenance'])
            )
            db.add(asset)
        
        db.commit()
        logger.info("Initial data populated successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error populating initial data: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def store_sensor_reading(sensor_id: str, reading_data: Dict) -> bool:
    """Store a sensor reading in the database"""
    db = SessionLocal()
    try:
        reading = SensorReading(
            sensor_id=sensor_id,
            timestamp=reading_data.get('timestamp', datetime.utcnow()),
            pressure=reading_data.get('pressure'),
            flow_rate=reading_data.get('flow_rate'),
            temperature=reading_data.get('temperature'),
            quality_score=reading_data.get('quality_score'),
            anomaly_score=reading_data.get('anomaly_score', 0.0),
            is_anomaly=reading_data.get('is_anomaly', False)
        )
        db.add(reading)
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error storing sensor reading: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def get_sensor_readings(sensor_id: str = None, hours: int = 24) -> pd.DataFrame:
    """Get sensor readings from database"""
    db = SessionLocal()
    try:
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        query = db.query(SensorReading).filter(SensorReading.timestamp >= cutoff_time)
        
        if sensor_id:
            query = query.filter(SensorReading.sensor_id == sensor_id)
        
        readings = query.order_by(SensorReading.timestamp.desc()).all()
        
        # Convert to DataFrame
        data = []
        for reading in readings:
            data.append({
                'sensor_id': reading.sensor_id,
                'timestamp': reading.timestamp,
                'pressure': reading.pressure,
                'flow_rate': reading.flow_rate,
                'temperature': reading.temperature,
                'quality_score': reading.quality_score,
                'anomaly_score': reading.anomaly_score,
                'is_anomaly': reading.is_anomaly
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error retrieving sensor readings: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def get_sensors() -> pd.DataFrame:
    """Get all sensors from database"""
    db = SessionLocal()
    try:
        sensors = db.query(Sensor).all()
        
        data = []
        for sensor in sensors:
            data.append({
                'sensor_id': sensor.sensor_id,
                'sensor_type': sensor.sensor_type,
                'location': sensor.location,
                'latitude': sensor.latitude,
                'longitude': sensor.longitude,
                'installation_date': sensor.installation_date,
                'maintenance_interval': sensor.maintenance_interval,
                'manufacturer': sensor.manufacturer,
                'model': sensor.model,
                'status': sensor.status,
                'calibration_date': sensor.calibration_date
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error retrieving sensors: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def get_infrastructure_assets() -> pd.DataFrame:
    """Get all infrastructure assets from database"""
    db = SessionLocal()
    try:
        assets = db.query(InfrastructureAsset).all()
        
        data = []
        for asset in assets:
            data.append({
                'asset_id': asset.asset_id,
                'asset_type': asset.asset_type,
                'name': asset.name,
                'latitude': asset.latitude,
                'longitude': asset.longitude,
                'installation_year': asset.installation_year,
                'capacity': asset.capacity,
                'material': asset.material,
                'condition': asset.condition,
                'last_inspection': asset.last_inspection,
                'next_maintenance': asset.next_maintenance
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error retrieving infrastructure assets: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def create_alert(sensor_id: str, alert_type: str, severity: str, title: str, description: str) -> bool:
    """Create a new alert in the database"""
    db = SessionLocal()
    try:
        alert = Alert(
            sensor_id=sensor_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description
        )
        db.add(alert)
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def get_active_alerts(limit: int = 50) -> pd.DataFrame:
    """Get active alerts from database"""
    db = SessionLocal()
    try:
        alerts = db.query(Alert).filter(Alert.status == 'active').order_by(Alert.timestamp.desc()).limit(limit).all()
        
        data = []
        for alert in alerts:
            data.append({
                'alert_id': alert.alert_id,
                'sensor_id': alert.sensor_id,
                'alert_type': alert.alert_type,
                'severity': alert.severity,
                'title': alert.title,
                'description': alert.description,
                'timestamp': alert.timestamp,
                'status': alert.status
            })
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}")
        return pd.DataFrame()
    finally:
        db.close()

def log_user_activity_db(username: str, action: str, details: str = "", ip_address: str = ""):
    """Log user activity to database"""
    db = SessionLocal()
    try:
        activity = UserActivity(
            username=username,
            action=action,
            details=details,
            ip_address=ip_address
        )
        db.add(activity)
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error logging user activity: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def store_data_quality_check(check_type: str, dataset_name: str, score: float, passed: bool, details: Dict):
    """Store data quality check results"""
    db = SessionLocal()
    try:
        import json
        check = DataQualityCheck(
            check_type=check_type,
            dataset_name=dataset_name,
            score=score,
            passed=passed,
            details=json.dumps(details)
        )
        db.add(check)
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error storing data quality check: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def check_database_connection() -> bool:
    """Check if database connection is working"""
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

def get_system_stats() -> Dict:
    """Get system statistics from database"""
    db = SessionLocal()
    try:
        stats = {
            'total_sensors': db.query(Sensor).count(),
            'active_sensors': db.query(Sensor).filter(Sensor.status == 'Active').count(),
            'maintenance_sensors': db.query(Sensor).filter(Sensor.status == 'Maintenance').count(),
            'offline_sensors': db.query(Sensor).filter(Sensor.status == 'Offline').count(),
            'total_assets': db.query(InfrastructureAsset).count(),
            'active_alerts': db.query(Alert).filter(Alert.status == 'active').count(),
            'total_readings_24h': db.query(SensorReading).filter(
                SensorReading.timestamp >= datetime.utcnow() - timedelta(hours=24)
            ).count()
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        return {}
    finally:
        db.close()