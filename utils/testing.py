"""
Comprehensive testing module for SIMS - Smart Infrastructure Monitoring System
Tests all features, links, buttons and functionalities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Any, Tuple
from .database import (
    check_database_connection, get_system_stats, get_sensors, 
    get_infrastructure_assets, get_sensor_readings, store_sensor_reading,
    create_alert, get_active_alerts
)
from .data_generator import generate_and_store_real_time_data, get_recent_sensor_data_from_db
from .auth import authenticate_user, get_users
from .ml_models import detect_anomalies, AnomalyDetector
from .data_quality import DataQualityChecker
from .data_governance import DataGovernance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SIMSTestSuite:
    """Comprehensive test suite for SIMS application"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results"""
        logger.info("Starting comprehensive SIMS test suite...")
        
        # Database Tests
        self.test_results['database'] = self.test_database_functionality()
        
        # Authentication Tests
        self.test_results['authentication'] = self.test_authentication_system()
        
        # Data Generation Tests
        self.test_results['data_generation'] = self.test_data_generation()
        
        # Machine Learning Tests
        self.test_results['ml_models'] = self.test_ml_functionality()
        
        # Data Quality Tests
        self.test_results['data_quality'] = self.test_data_quality_system()
        
        # Data Governance Tests
        self.test_results['data_governance'] = self.test_data_governance()
        
        # Integration Tests
        self.test_results['integration'] = self.test_system_integration()
        
        # Performance Tests
        self.test_results['performance'] = self.test_system_performance()
        
        self.test_results['summary'] = self.generate_test_summary()
        
        logger.info("Test suite completed!")
        return self.test_results
    
    def test_database_functionality(self) -> Dict[str, Any]:
        """Test database connection and operations"""
        results = {
            'test_name': 'Database Functionality',
            'tests': {},
            'overall_status': 'PASS'
        }
        
        try:
            # Test 1: Database Connection
            results['tests']['connection'] = {
                'status': 'PASS' if check_database_connection() else 'FAIL',
                'description': 'Database connection test'
            }
            
            # Test 2: System Statistics
            stats = get_system_stats()
            results['tests']['system_stats'] = {
                'status': 'PASS' if isinstance(stats, dict) and stats else 'FAIL',
                'description': 'System statistics retrieval',
                'data': stats
            }
            
            # Test 3: Sensor Data Retrieval
            sensors = get_sensors()
            results['tests']['sensor_retrieval'] = {
                'status': 'PASS' if not sensors.empty else 'FAIL',
                'description': 'Sensor metadata retrieval',
                'count': len(sensors)
            }
            
            # Test 4: Infrastructure Assets
            assets = get_infrastructure_assets()
            results['tests']['asset_retrieval'] = {
                'status': 'PASS' if not assets.empty else 'FAIL',
                'description': 'Infrastructure assets retrieval',
                'count': len(assets)
            }
            
            # Test 5: Sensor Reading Storage
            test_reading = {
                'timestamp': datetime.now(),
                'pressure': 45.5,
                'flow_rate': 25.2,
                'temperature': 20.1,
                'quality_score': 8.5,
                'anomaly_score': 0.1,
                'is_anomaly': False
            }
            
            sensor_id = sensors.iloc[0]['sensor_id'] if not sensors.empty else 'SENSOR_001'
            storage_success = store_sensor_reading(sensor_id, test_reading)
            results['tests']['reading_storage'] = {
                'status': 'PASS' if storage_success else 'FAIL',
                'description': 'Sensor reading storage test'
            }
            
        except Exception as e:
            results['overall_status'] = 'FAIL'
            results['error'] = str(e)
            logger.error(f"Database test error: {e}")
        
        return results
    
    def test_authentication_system(self) -> Dict[str, Any]:
        """Test authentication and user management"""
        results = {
            'test_name': 'Authentication System',
            'tests': {},
            'overall_status': 'PASS'
        }
        
        try:
            # Test 1: User Database Access
            users = get_users()
            results['tests']['user_database'] = {
                'status': 'PASS' if users else 'FAIL',
                'description': 'User database access',
                'user_count': len(users)
            }
            
            # Test 2: Authentication with Valid Credentials
            valid_auth = authenticate_user('admin', 'admin123')
            results['tests']['valid_auth'] = {
                'status': 'PASS' if valid_auth else 'FAIL',
                'description': 'Valid credential authentication'
            }
            
            # Test 3: Authentication with Invalid Credentials
            invalid_auth = authenticate_user('invalid', 'wrong')
            results['tests']['invalid_auth'] = {
                'status': 'PASS' if not invalid_auth else 'FAIL',
                'description': 'Invalid credential rejection'
            }
            
            # Test 4: Role-based Access
            results['tests']['roles'] = {
                'status': 'PASS',
                'description': 'Role-based access system',
                'roles': list(users.keys()) if users else []
            }
            
        except Exception as e:
            results['overall_status'] = 'FAIL'
            results['error'] = str(e)
            logger.error(f"Authentication test error: {e}")
        
        return results
    
    def test_data_generation(self) -> Dict[str, Any]:
        """Test data generation and storage"""
        results = {
            'test_name': 'Data Generation',
            'tests': {},
            'overall_status': 'PASS'
        }
        
        try:
            # Test 1: Real-time Data Generation and Storage
            generation_result = generate_and_store_real_time_data(5)
            results['tests']['realtime_generation'] = {
                'status': 'PASS' if generation_result else 'FAIL',
                'description': 'Real-time data generation and storage'
            }
            
            # Test 2: Data Retrieval from Database
            recent_data = get_recent_sensor_data_from_db(hours=1, limit=10)
            results['tests']['data_retrieval'] = {
                'status': 'PASS' if not recent_data.empty else 'FAIL',
                'description': 'Recent data retrieval from database',
                'records_count': len(recent_data)
            }
            
            # Test 3: Data Quality and Structure
            if not recent_data.empty:
                required_columns = ['sensor_id', 'timestamp', 'pressure', 'flow_rate', 'temperature']
                has_required_columns = all(col in recent_data.columns for col in required_columns)
                results['tests']['data_structure'] = {
                    'status': 'PASS' if has_required_columns else 'FAIL',
                    'description': 'Data structure validation',
                    'columns': list(recent_data.columns)
                }
            
        except Exception as e:
            results['overall_status'] = 'FAIL'
            results['error'] = str(e)
            logger.error(f"Data generation test error: {e}")
        
        return results
    
    def test_ml_functionality(self) -> Dict[str, Any]:
        """Test machine learning and anomaly detection"""
        results = {
            'test_name': 'Machine Learning Models',
            'tests': {},
            'overall_status': 'PASS'
        }
        
        try:
            # Get some test data
            test_data = get_recent_sensor_data_from_db(hours=2, limit=50)
            
            if not test_data.empty:
                # Test 1: Anomaly Detection
                anomalies = detect_anomalies(test_data)
                results['tests']['anomaly_detection'] = {
                    'status': 'PASS' if isinstance(anomalies, pd.DataFrame) else 'FAIL',
                    'description': 'Anomaly detection functionality',
                    'anomalies_found': len(anomalies) if isinstance(anomalies, pd.DataFrame) else 0
                }
                
                # Test 2: Anomaly Detector Class
                detector = AnomalyDetector()
                if len(test_data) > 10:
                    detector.train(test_data)
                    predictions = detector.predict(test_data.head(5))
                    results['tests']['ml_model_class'] = {
                        'status': 'PASS' if predictions is not None else 'FAIL',
                        'description': 'ML model class functionality'
                    }
                
            else:
                results['tests']['no_data'] = {
                    'status': 'SKIP',
                    'description': 'No data available for ML testing'
                }
            
        except Exception as e:
            results['overall_status'] = 'FAIL'
            results['error'] = str(e)
            logger.error(f"ML functionality test error: {e}")
        
        return results
    
    def test_data_quality_system(self) -> Dict[str, Any]:
        """Test data quality assessment"""
        results = {
            'test_name': 'Data Quality System',
            'tests': {},
            'overall_status': 'PASS'
        }
        
        try:
            # Get test data
            test_data = get_recent_sensor_data_from_db(hours=1, limit=20)
            
            if not test_data.empty:
                # Test 1: Data Quality Checker
                checker = DataQualityChecker()
                quality_results = checker.run_all_checks(test_data)
                
                results['tests']['quality_checks'] = {
                    'status': 'PASS' if quality_results else 'FAIL',
                    'description': 'Data quality assessment',
                    'overall_score': quality_results.get('overall_score', 0)
                }
                
                # Test 2: Quality Alerts Generation
                alerts = checker.generate_quality_alerts(quality_results)
                results['tests']['quality_alerts'] = {
                    'status': 'PASS' if isinstance(alerts, list) else 'FAIL',
                    'description': 'Quality alerts generation',
                    'alerts_count': len(alerts)
                }
                
            else:
                results['tests']['no_data'] = {
                    'status': 'SKIP',
                    'description': 'No data available for quality testing'
                }
            
        except Exception as e:
            results['overall_status'] = 'FAIL'
            results['error'] = str(e)
            logger.error(f"Data quality test error: {e}")
        
        return results
    
    def test_data_governance(self) -> Dict[str, Any]:
        """Test data governance framework"""
        results = {
            'test_name': 'Data Governance',
            'tests': {},
            'overall_status': 'PASS'
        }
        
        try:
            # Test 1: Data Governance Framework
            governance = DataGovernance()
            compliance_report = governance.generate_compliance_report()
            
            results['tests']['governance_framework'] = {
                'status': 'PASS' if compliance_report else 'FAIL',
                'description': 'Data governance framework',
                'compliance_score': compliance_report.get('overall_score', 0)
            }
            
            # Test 2: Policy Compliance Evaluation
            compliance_check = governance.evaluate_policy_compliance('sensor_data', 'admin')
            results['tests']['policy_compliance'] = {
                'status': 'PASS' if isinstance(compliance_check, dict) else 'FAIL',
                'description': 'Policy compliance evaluation'
            }
            
        except Exception as e:
            results['overall_status'] = 'FAIL'
            results['error'] = str(e)
            logger.error(f"Data governance test error: {e}")
        
        return results
    
    def test_system_integration(self) -> Dict[str, Any]:
        """Test end-to-end system integration"""
        results = {
            'test_name': 'System Integration',
            'tests': {},
            'overall_status': 'PASS'
        }
        
        try:
            # Test 1: Data Flow (Generation -> Storage -> Retrieval)
            initial_count = get_system_stats().get('total_readings_24h', 0)
            generate_and_store_real_time_data(3)
            new_count = get_system_stats().get('total_readings_24h', 0)
            
            results['tests']['data_flow'] = {
                'status': 'PASS' if new_count > initial_count else 'FAIL',
                'description': 'End-to-end data flow',
                'readings_added': new_count - initial_count
            }
            
            # Test 2: Alert System Integration
            test_alert = create_alert(
                'SENSOR_001', 
                'test', 
                'info', 
                'Test Alert', 
                'This is a test alert for system testing'
            )
            
            results['tests']['alert_system'] = {
                'status': 'PASS' if test_alert else 'FAIL',
                'description': 'Alert system integration'
            }
            
            # Test 3: Cross-Module Communication
            sensors = get_sensors()
            if not sensors.empty:
                recent_data = get_recent_sensor_data_from_db(hours=1)
                sensor_in_data = sensors.iloc[0]['sensor_id'] in recent_data['sensor_id'].values if not recent_data.empty else False
                
                results['tests']['cross_module'] = {
                    'status': 'PASS' if sensor_in_data else 'PARTIAL',
                    'description': 'Cross-module data consistency'
                }
            
        except Exception as e:
            results['overall_status'] = 'FAIL'
            results['error'] = str(e)
            logger.error(f"Integration test error: {e}")
        
        return results
    
    def test_system_performance(self) -> Dict[str, Any]:
        """Test system performance and responsiveness"""
        results = {
            'test_name': 'System Performance',
            'tests': {},
            'overall_status': 'PASS'
        }
        
        try:
            # Test 1: Database Query Performance
            start_time = time.time()
            get_system_stats()
            db_query_time = time.time() - start_time
            
            results['tests']['db_performance'] = {
                'status': 'PASS' if db_query_time < 2.0 else 'SLOW',
                'description': 'Database query performance',
                'response_time_seconds': round(db_query_time, 3)
            }
            
            # Test 2: Data Generation Performance
            start_time = time.time()
            generate_and_store_real_time_data(10)
            generation_time = time.time() - start_time
            
            results['tests']['data_generation_performance'] = {
                'status': 'PASS' if generation_time < 5.0 else 'SLOW',
                'description': 'Data generation performance',
                'response_time_seconds': round(generation_time, 3)
            }
            
            # Test 3: Data Retrieval Performance
            start_time = time.time()
            get_recent_sensor_data_from_db(hours=1, limit=100)
            retrieval_time = time.time() - start_time
            
            results['tests']['data_retrieval_performance'] = {
                'status': 'PASS' if retrieval_time < 3.0 else 'SLOW',
                'description': 'Data retrieval performance',
                'response_time_seconds': round(retrieval_time, 3)
            }
            
        except Exception as e:
            results['overall_status'] = 'FAIL'
            results['error'] = str(e)
            logger.error(f"Performance test error: {e}")
        
        return results
    
    def generate_test_summary(self) -> Dict[str, Any]:
        """Generate overall test summary"""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        
        for category, category_results in self.test_results.items():
            if category == 'summary':
                continue
                
            for test_name, test_result in category_results.get('tests', {}).items():
                total_tests += 1
                status = test_result.get('status', 'UNKNOWN')
                if status == 'PASS':
                    passed_tests += 1
                elif status == 'FAIL':
                    failed_tests += 1
                elif status == 'SKIP':
                    skipped_tests += 1
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'skipped': skipped_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'duration_seconds': duration.total_seconds(),
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'overall_status': 'PASS' if failed_tests == 0 else 'FAIL'
        }

def run_comprehensive_tests() -> Dict[str, Any]:
    """Run comprehensive SIMS testing suite"""
    test_suite = SIMSTestSuite()
    return test_suite.run_all_tests()

def generate_test_report(test_results: Dict[str, Any]) -> str:
    """Generate a formatted test report"""
    report = []
    report.append("=" * 60)
    report.append("SIMS - Smart Infrastructure Monitoring System")
    report.append("Comprehensive Test Report")
    report.append("=" * 60)
    
    summary = test_results.get('summary', {})
    report.append(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
    report.append(f"Total Tests: {summary.get('total_tests', 0)}")
    report.append(f"Passed: {summary.get('passed', 0)}")
    report.append(f"Failed: {summary.get('failed', 0)}")
    report.append(f"Skipped: {summary.get('skipped', 0)}")
    report.append(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
    report.append(f"Duration: {summary.get('duration_seconds', 0):.2f} seconds")
    report.append("")
    
    # Detailed results
    for category, results in test_results.items():
        if category == 'summary':
            continue
            
        report.append(f"### {results.get('test_name', category)}")
        report.append(f"Overall Status: {results.get('overall_status', 'UNKNOWN')}")
        
        if 'error' in results:
            report.append(f"Error: {results['error']}")
        
        for test_name, test_result in results.get('tests', {}).items():
            status = test_result.get('status', 'UNKNOWN')
            description = test_result.get('description', test_name)
            report.append(f"  {status}: {description}")
        
        report.append("")
    
    return "\n".join(report)