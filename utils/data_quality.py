import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json

class DataQualityChecker:
    """
    Comprehensive data quality assessment and monitoring
    """
    
    def __init__(self):
        self.quality_rules = self._load_default_rules()
        self.quality_history = []
    
    def _load_default_rules(self):
        """
        Load default data quality rules
        """
        return {
            'completeness': {
                'required_fields': ['sensor_id', 'timestamp', 'pressure', 'flow_rate'],
                'threshold': 95.0  # Minimum percentage of complete records
            },
            'accuracy': {
                'pressure_range': (0, 100),  # PSI
                'flow_rate_range': (0, 200),  # L/min
                'temperature_range': (-10, 60)  # Celsius
            },
            'consistency': {
                'sensor_id_format': r'^SENSOR_\d{3}$',
                'timestamp_format': 'datetime',
                'coordinate_precision': 6  # Decimal places
            },
            'timeliness': {
                'max_age_minutes': 60,  # Data should not be older than 1 hour
                'frequency_check': True
            },
            'uniqueness': {
                'duplicate_threshold': 1.0  # Maximum percentage of duplicates
            }
        }
    
    def check_completeness(self, data: pd.DataFrame) -> Dict:
        """
        Check data completeness
        """
        required_fields = self.quality_rules['completeness']['required_fields']
        threshold = self.quality_rules['completeness']['threshold']
        
        results = {
            'rule': 'completeness',
            'passed': True,
            'score': 100.0,
            'details': {}
        }
        
        for field in required_fields:
            if field in data.columns:
                non_null_count = data[field].notna().sum()
                completeness_pct = (non_null_count / len(data)) * 100
                
                results['details'][field] = {
                    'completeness_percentage': completeness_pct,
                    'missing_count': len(data) - non_null_count,
                    'passed': completeness_pct >= threshold
                }
                
                if completeness_pct < threshold:
                    results['passed'] = False
            else:
                results['details'][field] = {
                    'completeness_percentage': 0.0,
                    'missing_count': len(data),
                    'passed': False
                }
                results['passed'] = False
        
        # Calculate overall completeness score
        if results['details']:
            avg_completeness = np.mean([d['completeness_percentage'] for d in results['details'].values()])
            results['score'] = avg_completeness
        
        return results
    
    def check_accuracy(self, data: pd.DataFrame) -> Dict:
        """
        Check data accuracy based on value ranges
        """
        accuracy_rules = self.quality_rules['accuracy']
        
        results = {
            'rule': 'accuracy',
            'passed': True,
            'score': 100.0,
            'details': {}
        }
        
        total_violations = 0
        total_records = len(data)
        
        for field, (min_val, max_val) in accuracy_rules.items():
            if field in data.columns:
                # Check for values outside acceptable range
                valid_mask = (data[field] >= min_val) & (data[field] <= max_val) & data[field].notna()
                valid_count = valid_mask.sum()
                invalid_count = len(data) - valid_count
                
                accuracy_pct = (valid_count / len(data)) * 100
                
                results['details'][field] = {
                    'accuracy_percentage': accuracy_pct,
                    'invalid_count': invalid_count,
                    'range': (min_val, max_val),
                    'passed': accuracy_pct >= 95.0
                }
                
                total_violations += invalid_count
                
                if accuracy_pct < 95.0:
                    results['passed'] = False
        
        # Calculate overall accuracy score
        overall_accuracy = ((total_records - total_violations) / total_records) * 100 if total_records > 0 else 100
        results['score'] = overall_accuracy
        
        return results
    
    def check_consistency(self, data: pd.DataFrame) -> Dict:
        """
        Check data consistency
        """
        results = {
            'rule': 'consistency',
            'passed': True,
            'score': 100.0,
            'details': {}
        }
        
        # Check sensor ID format
        if 'sensor_id' in data.columns:
            sensor_id_pattern = self.quality_rules['consistency']['sensor_id_format']
            valid_ids = data['sensor_id'].str.match(sensor_id_pattern, na=False)
            consistency_pct = (valid_ids.sum() / len(data)) * 100
            
            results['details']['sensor_id_format'] = {
                'consistency_percentage': consistency_pct,
                'invalid_count': len(data) - valid_ids.sum(),
                'passed': consistency_pct >= 98.0
            }
            
            if consistency_pct < 98.0:
                results['passed'] = False
        
        # Check for consistent data types
        numeric_fields = ['pressure', 'flow_rate', 'temperature', 'latitude', 'longitude']
        for field in numeric_fields:
            if field in data.columns:
                numeric_count = pd.to_numeric(data[field], errors='coerce').notna().sum()
                consistency_pct = (numeric_count / len(data)) * 100
                
                results['details'][f'{field}_type_consistency'] = {
                    'consistency_percentage': consistency_pct,
                    'type_violations': len(data) - numeric_count,
                    'passed': consistency_pct >= 99.0
                }
                
                if consistency_pct < 99.0:
                    results['passed'] = False
        
        # Calculate overall consistency score
        if results['details']:
            avg_consistency = np.mean([d['consistency_percentage'] for d in results['details'].values()])
            results['score'] = avg_consistency
        
        return results
    
    def check_timeliness(self, data: pd.DataFrame) -> Dict:
        """
        Check data timeliness
        """
        results = {
            'rule': 'timeliness',
            'passed': True,
            'score': 100.0,
            'details': {}
        }
        
        if 'timestamp' in data.columns:
            current_time = datetime.now()
            max_age = timedelta(minutes=self.quality_rules['timeliness']['max_age_minutes'])
            
            # Convert timestamp to datetime if it's not already
            timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
            
            # Check for recent data
            recent_mask = (current_time - timestamps) <= max_age
            timely_count = recent_mask.sum()
            timeliness_pct = (timely_count / len(data)) * 100
            
            results['details']['data_freshness'] = {
                'timeliness_percentage': timeliness_pct,
                'stale_records': len(data) - timely_count,
                'max_age_minutes': self.quality_rules['timeliness']['max_age_minutes'],
                'passed': timeliness_pct >= 80.0
            }
            
            if timeliness_pct < 80.0:
                results['passed'] = False
            
            results['score'] = timeliness_pct
        
        return results
    
    def check_uniqueness(self, data: pd.DataFrame) -> Dict:
        """
        Check for duplicate records
        """
        results = {
            'rule': 'uniqueness',
            'passed': True,
            'score': 100.0,
            'details': {}
        }
        
        # Check for exact duplicates
        duplicate_mask = data.duplicated()
        duplicate_count = duplicate_mask.sum()
        duplicate_pct = (duplicate_count / len(data)) * 100
        
        threshold = self.quality_rules['uniqueness']['duplicate_threshold']
        
        results['details']['exact_duplicates'] = {
            'duplicate_percentage': duplicate_pct,
            'duplicate_count': duplicate_count,
            'unique_records': len(data) - duplicate_count,
            'passed': duplicate_pct <= threshold
        }
        
        if duplicate_pct > threshold:
            results['passed'] = False
        
        # Check for sensor-timestamp duplicates (should be unique)
        if 'sensor_id' in data.columns and 'timestamp' in data.columns:
            sensor_time_duplicates = data.duplicated(subset=['sensor_id', 'timestamp']).sum()
            sensor_time_dup_pct = (sensor_time_duplicates / len(data)) * 100
            
            results['details']['sensor_timestamp_duplicates'] = {
                'duplicate_percentage': sensor_time_dup_pct,
                'duplicate_count': sensor_time_duplicates,
                'passed': sensor_time_dup_pct <= threshold
            }
            
            if sensor_time_dup_pct > threshold:
                results['passed'] = False
        
        # Calculate overall uniqueness score
        overall_uniqueness = 100 - duplicate_pct
        results['score'] = max(0, overall_uniqueness)
        
        return results
    
    def run_all_checks(self, data: pd.DataFrame) -> Dict:
        """
        Run all data quality checks
        """
        check_methods = [
            self.check_completeness,
            self.check_accuracy,
            self.check_consistency,
            self.check_timeliness,
            self.check_uniqueness
        ]
        
        results = {
            'overall_passed': True,
            'overall_score': 0.0,
            'timestamp': datetime.now().isoformat(),
            'record_count': len(data),
            'checks': {}
        }
        
        scores = []
        
        for check_method in check_methods:
            try:
                check_result = check_method(data)
                rule_name = check_result['rule']
                results['checks'][rule_name] = check_result
                scores.append(check_result['score'])
                
                if not check_result['passed']:
                    results['overall_passed'] = False
                    
            except Exception as e:
                rule_name = check_method.__name__.replace('check_', '')
                results['checks'][rule_name] = {
                    'rule': rule_name,
                    'passed': False,
                    'score': 0.0,
                    'error': str(e)
                }
                scores.append(0.0)
                results['overall_passed'] = False
        
        # Calculate overall score
        results['overall_score'] = np.mean(scores) if scores else 0.0
        
        # Store in history
        self.quality_history.append(results)
        
        return results
    
    def get_quality_trends(self, days: int = 30) -> Dict:
        """
        Get data quality trends over time
        """
        if not self.quality_history:
            return {'error': 'No quality history available'}
        
        # Filter recent history
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in self.quality_history 
            if datetime.fromisoformat(h['timestamp']) >= cutoff_date
        ]
        
        if not recent_history:
            return {'error': 'No recent quality data available'}
        
        # Calculate trends
        overall_scores = [h['overall_score'] for h in recent_history]
        timestamps = [h['timestamp'] for h in recent_history]
        
        trends = {
            'timestamps': timestamps,
            'overall_scores': overall_scores,
            'average_score': np.mean(overall_scores),
            'trend_direction': 'improving' if len(overall_scores) > 1 and overall_scores[-1] > overall_scores[0] else 'declining'
        }
        
        # Rule-specific trends
        rule_trends = {}
        for rule in ['completeness', 'accuracy', 'consistency', 'timeliness', 'uniqueness']:
            rule_scores = [h['checks'].get(rule, {}).get('score', 0) for h in recent_history]
            rule_trends[rule] = {
                'scores': rule_scores,
                'average': np.mean(rule_scores) if rule_scores else 0
            }
        
        trends['rule_trends'] = rule_trends
        
        return trends
    
    def generate_quality_alerts(self, results: Dict) -> List[Dict]:
        """
        Generate alerts based on quality check results
        """
        alerts = []
        
        # Overall score alert
        if results['overall_score'] < 90:
            severity = 'critical' if results['overall_score'] < 80 else 'warning'
            alerts.append({
                'type': 'overall_quality',
                'severity': severity,
                'message': f"Overall data quality score is {results['overall_score']:.1f}%",
                'timestamp': datetime.now().isoformat()
            })
        
        # Rule-specific alerts
        for rule_name, rule_result in results['checks'].items():
            if not rule_result.get('passed', True):
                alerts.append({
                    'type': f'{rule_name}_violation',
                    'severity': 'warning',
                    'message': f"Data quality rule '{rule_name}' failed with score {rule_result.get('score', 0):.1f}%",
                    'details': rule_result.get('details', {}),
                    'timestamp': datetime.now().isoformat()
                })
        
        return alerts
    
    def suggest_improvements(self, results: Dict) -> List[str]:
        """
        Suggest improvements based on quality check results
        """
        suggestions = []
        
        for rule_name, rule_result in results['checks'].items():
            if rule_result.get('score', 100) < 95:
                if rule_name == 'completeness':
                    suggestions.append("Implement data validation at ingestion to ensure required fields are populated")
                elif rule_name == 'accuracy':
                    suggestions.append("Add range validation checks to prevent out-of-bounds values")
                elif rule_name == 'consistency':
                    suggestions.append("Standardize data formats and implement schema validation")
                elif rule_name == 'timeliness':
                    suggestions.append("Optimize data pipeline to reduce processing delays")
                elif rule_name == 'uniqueness':
                    suggestions.append("Implement deduplication logic in data processing pipeline")
        
        return suggestions

def check_data_completeness(data):
    """
    Check completeness of sensor data
    Returns percentage of non-null values for each column
    """
    if data.empty:
        return {'score': 0, 'details': 'No data available'}
    
    completeness = {}
    for column in data.columns:
        non_null_count = data[column].notna().sum()
        total_count = len(data)
        completeness[column] = (non_null_count / total_count) * 100
    
    overall_score = sum(completeness.values()) / len(completeness)
    
    return {
        'score': overall_score,
        'column_scores': completeness,
        'total_records': len(data),
        'complete_records': sum(1 for _, row in data.iterrows() if row.notna().all())
    }

def check_data_accuracy(data):
    """
    Check data accuracy based on value ranges
    """
    checker = DataQualityChecker()
    return checker.check_accuracy(data)

def generate_quality_report(data: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive data quality report
    """
    checker = DataQualityChecker()
    results = checker.run_all_checks(data)
    
    # Add summary statistics
    results['summary'] = {
        'total_records': len(data),
        'total_fields': len(data.columns),
        'numeric_fields': len(data.select_dtypes(include=[np.number]).columns),
        'text_fields': len(data.select_dtypes(include=['object']).columns),
        'date_fields': len(data.select_dtypes(include=['datetime']).columns)
    }
    
    # Add alerts and suggestions
    results['alerts'] = checker.generate_quality_alerts(results)
    results['suggestions'] = checker.suggest_improvements(results)
    
    # Add profiling information
    results['profiling'] = {
        'null_counts': data.isnull().sum().to_dict(),
        'unique_counts': data.nunique().to_dict(),
        'data_types': data.dtypes.astype(str).to_dict()
    }
    
    return results

def validate_sensor_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate sensor data against business rules
    """
    errors = []
    
    # Required columns
    required_columns = ['sensor_id', 'timestamp', 'pressure', 'flow_rate', 'temperature']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")
    
    # Data type validation
    if 'pressure' in data.columns:
        non_numeric_pressure = data[~pd.to_numeric(data['pressure'], errors='coerce').notna()]
        if not non_numeric_pressure.empty:
            errors.append(f"Non-numeric pressure values found in {len(non_numeric_pressure)} records")
    
    # Business rule validation
    if 'pressure' in data.columns and 'flow_rate' in data.columns:
        # Pressure and flow rate should be correlated
        if len(data) > 10:
            correlation = data['pressure'].corr(data['flow_rate'])
            if abs(correlation) < 0.1:
                errors.append("Pressure and flow rate show no correlation - possible data quality issue")
    
    # Temporal validation
    if 'timestamp' in data.columns:
        timestamps = pd.to_datetime(data['timestamp'], errors='coerce')
        future_timestamps = timestamps > datetime.now()
        if future_timestamps.any():
            errors.append(f"Found {future_timestamps.sum()} timestamps in the future")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def create_quality_dashboard_data() -> Dict:
    """
    Create data for quality dashboard visualization
    """
    # Simulate quality metrics over time
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    
    dashboard_data = {
        'quality_trends': {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'overall_quality': np.random.uniform(85, 98, len(dates)).tolist(),
            'completeness': np.random.uniform(90, 99, len(dates)).tolist(),
            'accuracy': np.random.uniform(88, 96, len(dates)).tolist(),
            'consistency': np.random.uniform(92, 99, len(dates)).tolist(),
            'timeliness': np.random.uniform(85, 95, len(dates)).tolist()
        },
        'current_metrics': {
            'overall_score': np.random.uniform(90, 96),
            'records_processed': 1245632,
            'quality_issues': 45,
            'data_freshness': '2 minutes ago'
        },
        'rule_violations': [
            {'rule': 'Range Check', 'count': 23, 'severity': 'Medium'},
            {'rule': 'Format Validation', 'count': 12, 'severity': 'Low'},
            {'rule': 'Completeness', 'count': 8, 'severity': 'High'},
            {'rule': 'Uniqueness', 'count': 2, 'severity': 'Low'}
        ]
    }
    
    return dashboard_data
