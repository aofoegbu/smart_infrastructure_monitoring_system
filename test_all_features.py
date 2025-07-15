#!/usr/bin/env python3
"""
Comprehensive test script for SIMS (Smart Infrastructure Monitoring System)
Tests all major features and components to ensure they're working properly.
"""

import sys
import os
sys.path.append('/home/runner/workspace')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Test database functionality
def test_database():
    print("Testing database functionality...")
    try:
        from utils.database import (
            check_database_connection, get_system_stats, get_sensors, 
            get_infrastructure_assets, get_sensor_readings, get_active_alerts,
            store_sensor_reading, create_alert
        )
        
        # Test database connection
        assert check_database_connection(), "Database connection failed"
        print("✓ Database connection successful")
        
        # Test system stats
        stats = get_system_stats()
        assert isinstance(stats, dict), "System stats should be a dictionary"
        assert 'total_sensors' in stats, "System stats should contain total_sensors"
        print("✓ System stats working")
        
        # Test sensors retrieval
        sensors = get_sensors()
        assert isinstance(sensors, pd.DataFrame), "Sensors should be a DataFrame"
        print("✓ Sensors retrieval working")
        
        # Test infrastructure assets
        assets = get_infrastructure_assets()
        assert isinstance(assets, pd.DataFrame), "Assets should be a DataFrame"
        print("✓ Infrastructure assets retrieval working")
        
        # Test sensor readings
        readings = get_sensor_readings(hours=24)
        assert isinstance(readings, pd.DataFrame), "Readings should be a DataFrame"
        print("✓ Sensor readings retrieval working")
        
        # Test alerts
        alerts = get_active_alerts()
        assert isinstance(alerts, pd.DataFrame), "Alerts should be a DataFrame"
        print("✓ Active alerts retrieval working")
        
        # Test storing sensor reading
        test_reading = {
            'timestamp': datetime.now(),
            'pressure': 45.0,
            'flow_rate': 25.0,
            'temperature': 22.0,
            'quality_score': 8.5,
            'anomaly_score': 0.1,
            'is_anomaly': False
        }
        result = store_sensor_reading('SENSOR_001', test_reading)
        assert result == True, "Storing sensor reading failed"
        print("✓ Sensor reading storage working")
        
        return True
        
    except Exception as e:
        print(f"✗ Database test failed: {e}")
        return False

# Test data generation
def test_data_generation():
    print("\nTesting data generation...")
    try:
        from utils.data_generator import (
            generate_real_time_data, generate_historical_data, 
            generate_and_store_real_time_data, get_recent_sensor_data_from_db
        )
        
        # Test real-time data generation
        data = generate_real_time_data(10)
        assert isinstance(data, pd.DataFrame), "Real-time data should be a DataFrame"
        assert len(data) == 10, "Should generate 10 sensor readings"
        print("✓ Real-time data generation working")
        
        # Test historical data generation
        historical = generate_historical_data(hours=2, sensors_count=5)
        assert isinstance(historical, pd.DataFrame), "Historical data should be a DataFrame"
        assert len(historical) > 0, "Should generate historical data"
        print("✓ Historical data generation working")
        
        # Test data storage
        generate_and_store_real_time_data(5)
        recent = get_recent_sensor_data_from_db(hours=1)
        assert isinstance(recent, pd.DataFrame), "Recent data should be a DataFrame"
        print("✓ Data storage and retrieval working")
        
        return True
        
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False

# Test ML models
def test_ml_models():
    print("\nTesting ML models...")
    try:
        from utils.ml_models import (
            AnomalyDetector, PredictiveMaintenanceModel, 
            predict_maintenance, calculate_sensor_health_score
        )
        
        # Test anomaly detection
        detector = AnomalyDetector()
        test_data = pd.DataFrame({
            'sensor_id': ['SENSOR_001', 'SENSOR_002', 'SENSOR_003', 'SENSOR_004', 'SENSOR_005'],
            'pressure': [40, 45, 50, 100, 45],  # One anomaly
            'flow_rate': [20, 25, 30, 20, 25],
            'temperature': [20, 22, 24, 22, 23]
        })
        
        # Train the detector first
        detector.train(test_data)
        labels, scores = detector.predict(test_data)
        assert isinstance(labels, np.ndarray), "Labels should be numpy array"
        assert isinstance(scores, np.ndarray), "Scores should be numpy array"
        print("✓ Anomaly detection working")
        
        # Test predictive maintenance
        test_data_pm = pd.DataFrame({
            'sensor_id': ['SENSOR_001', 'SENSOR_002', 'SENSOR_003', 'SENSOR_004', 'SENSOR_005'],
            'pressure': [40, 45, 50, 45, 42],
            'flow_rate': [20, 25, 30, 25, 22],
            'temperature': [20, 22, 24, 22, 21],
            'anomaly_score': [0.1, 0.2, 0.3, 0.15, 0.25]
        })
        
        maintenance_pred = predict_maintenance(test_data_pm)
        assert isinstance(maintenance_pred, np.ndarray), "Maintenance predictions should be numpy array"
        print("✓ Predictive maintenance working")
        
        # Test sensor health score
        health_scores = calculate_sensor_health_score(test_data_pm)
        assert isinstance(health_scores, pd.DataFrame), "Health scores should be DataFrame"
        print("✓ Sensor health scoring working")
        
        return True
        
    except Exception as e:
        print(f"✗ ML models test failed: {e}")
        return False

# Test data quality
def test_data_quality():
    print("\nTesting data quality...")
    try:
        from utils.data_quality import (
            DataQualityChecker, generate_quality_report,
            validate_sensor_data, create_quality_dashboard_data
        )
        
        # Test data quality checker
        checker = DataQualityChecker()
        test_data = pd.DataFrame({
            'sensor_id': ['SENSOR_001', 'SENSOR_002', 'SENSOR_003'],
            'timestamp': [datetime.now(), datetime.now(), datetime.now()],
            'pressure': [45.0, 50.0, 40.0],
            'flow_rate': [25.0, 30.0, 20.0],
            'temperature': [22.0, 24.0, 20.0]
        })
        
        quality_results = checker.run_all_checks(test_data)
        assert isinstance(quality_results, dict), "Quality results should be a dictionary"
        print("✓ Data quality checks working")
        
        # Test quality report
        report = generate_quality_report(test_data)
        assert isinstance(report, dict), "Quality report should be a dictionary"
        print("✓ Quality report generation working")
        
        # Test data validation
        is_valid, errors = validate_sensor_data(test_data)
        assert isinstance(is_valid, bool), "Validation should return boolean"
        assert isinstance(errors, list), "Errors should be a list"
        print("✓ Data validation working")
        
        return True
        
    except Exception as e:
        print(f"✗ Data quality test failed: {e}")
        return False

# Test authentication
def test_authentication():
    print("\nTesting authentication...")
    try:
        from utils.auth import (
            authenticate_user, hash_password, verify_password,
            get_users, check_permission
        )
        
        # Test password hashing
        password = "test123"
        hashed = hash_password(password)
        assert verify_password(password, hashed), "Password verification failed"
        print("✓ Password hashing working")
        
        # Test user authentication
        result = authenticate_user("admin", "admin123")
        assert result is not None, "Admin authentication failed"
        print("✓ User authentication working")
        
        # Test users retrieval
        users = get_users()
        assert isinstance(users, dict), "Users should be a dictionary"
        assert len(users) > 0, "Should have users"
        print("✓ Users retrieval working")
        
        return True
        
    except Exception as e:
        print(f"✗ Authentication test failed: {e}")
        return False

# Test data governance
def test_data_governance():
    print("\nTesting data governance...")
    try:
        from utils.data_governance import (
            DataGovernance, DataCatalog, AccessControl, 
            ComplianceManager, calculate_data_governance_score
        )
        
        # Test data governance
        governance = DataGovernance()
        compliance = governance.evaluate_policy_compliance('sensor_data', 'admin')
        assert isinstance(compliance, dict), "Compliance should be a dictionary"
        print("✓ Data governance working")
        
        # Test data catalog
        catalog = DataCatalog()
        datasets = catalog.search_datasets('sensor')
        assert isinstance(datasets, list), "Datasets should be a list"
        print("✓ Data catalog working")
        
        # Test access control
        access_control = AccessControl()
        has_access = access_control.check_access('admin', 'sensor_data', 'read')
        assert isinstance(has_access, bool), "Access check should return boolean"
        print("✓ Access control working")
        
        # Test compliance manager
        compliance_mgr = ComplianceManager()
        compliance_result = compliance_mgr.evaluate_compliance('GDPR')
        assert isinstance(compliance_result, dict), "Compliance result should be a dictionary"
        print("✓ Compliance manager working")
        
        return True
        
    except Exception as e:
        print(f"✗ Data governance test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("SIMS - Smart Infrastructure Monitoring System")
    print("Comprehensive Feature Testing")
    print("=" * 60)
    
    tests = [
        ("Database", test_database),
        ("Data Generation", test_data_generation),
        ("ML Models", test_ml_models),
        ("Data Quality", test_data_quality),
        ("Authentication", test_authentication),
        ("Data Governance", test_data_governance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! SIMS is fully functional.")
        return True
    else:
        print(f"\n⚠️  {failed} tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)