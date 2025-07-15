import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import hashlib
from typing import Dict, List, Any

class DataGovernance:
    """
    Data governance framework for managing data policies, compliance, and quality
    """
    
    def __init__(self):
        self.policies = self._load_default_policies()
        self.data_classifications = ['public', 'internal', 'confidential', 'restricted']
        self.compliance_frameworks = ['GDPR', 'SOX', 'ISO27001', 'HIPAA']
    
    def _load_default_policies(self):
        """
        Load default data governance policies
        """
        return {
            'data_retention': {
                'name': 'Data Retention Policy',
                'description': 'Defines how long data should be retained',
                'rules': {
                    'sensor_data': '2 years',
                    'user_logs': '1 year',
                    'audit_trails': '7 years',
                    'backup_data': '90 days'
                },
                'last_updated': '2023-06-15',
                'status': 'active'
            },
            'data_classification': {
                'name': 'Data Classification Policy',
                'description': 'Defines data classification levels and handling',
                'rules': {
                    'sensor_readings': 'internal',
                    'user_credentials': 'confidential',
                    'system_logs': 'internal',
                    'financial_data': 'restricted'
                },
                'last_updated': '2023-07-01',
                'status': 'active'
            },
            'access_control': {
                'name': 'Access Control Policy',
                'description': 'Defines who can access what data',
                'rules': {
                    'role_based_access': True,
                    'principle_of_least_privilege': True,
                    'regular_access_review': 'quarterly'
                },
                'last_updated': '2023-06-20',
                'status': 'active'
            }
        }
    
    def evaluate_policy_compliance(self, data_type: str, user_role: str) -> Dict:
        """
        Evaluate compliance for data access
        """
        compliance_result = {
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Check data classification compliance
        if data_type in self.policies['data_classification']['rules']:
            classification = self.policies['data_classification']['rules'][data_type]
            
            # Check if user role can access this classification
            access_matrix = {
                'public': ['administrator', 'manager', 'operator', 'analyst', 'viewer'],
                'internal': ['administrator', 'manager', 'operator', 'analyst'],
                'confidential': ['administrator', 'manager'],
                'restricted': ['administrator']
            }
            
            if user_role not in access_matrix.get(classification, []):
                compliance_result['compliant'] = False
                compliance_result['violations'].append(
                    f"User role '{user_role}' not authorized for '{classification}' data"
                )
        
        return compliance_result
    
    def get_retention_policy(self, data_type: str) -> str:
        """
        Get retention policy for specific data type
        """
        return self.policies['data_retention']['rules'].get(data_type, 'default: 1 year')
    
    def update_policy(self, policy_name: str, new_rules: Dict):
        """
        Update governance policy
        """
        if policy_name in self.policies:
            self.policies[policy_name]['rules'].update(new_rules)
            self.policies[policy_name]['last_updated'] = datetime.now().strftime('%Y-%m-%d')
            return True
        return False
    
    def generate_compliance_report(self) -> Dict:
        """
        Generate compliance report
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'policies_count': len(self.policies),
            'active_policies': sum(1 for p in self.policies.values() if p['status'] == 'active'),
            'compliance_score': np.random.uniform(85, 98),  # Simulated score
            'frameworks': self.compliance_frameworks,
            'recommendations': [
                'Review access permissions quarterly',
                'Update data classification guidelines',
                'Implement automated compliance monitoring'
            ]
        }
        
        return report

class DataCatalog:
    """
    Data catalog for managing metadata and data lineage
    """
    
    def __init__(self):
        self.datasets = self._initialize_catalog()
    
    def _initialize_catalog(self):
        """
        Initialize data catalog with default datasets
        """
        return {
            'sensor_readings': {
                'name': 'Sensor Readings',
                'description': 'Real-time sensor data from infrastructure monitoring',
                'schema': {
                    'sensor_id': 'string',
                    'timestamp': 'datetime',
                    'pressure': 'float',
                    'flow_rate': 'float',
                    'temperature': 'float',
                    'latitude': 'float',
                    'longitude': 'float'
                },
                'source': 'IoT Sensors',
                'owner': 'Infrastructure Team',
                'classification': 'internal',
                'last_updated': datetime.now().isoformat(),
                'record_count': 2453621,
                'quality_score': 96.2,
                'lineage': ['Raw Sensor Data', 'Data Validation', 'Data Storage']
            },
            'infrastructure_assets': {
                'name': 'Infrastructure Assets',
                'description': 'Master data for infrastructure assets',
                'schema': {
                    'asset_id': 'string',
                    'asset_type': 'string',
                    'location': 'string',
                    'installation_date': 'date',
                    'condition': 'string'
                },
                'source': 'Asset Management System',
                'owner': 'Asset Management',
                'classification': 'internal',
                'last_updated': datetime.now().isoformat(),
                'record_count': 1247,
                'quality_score': 98.7,
                'lineage': ['Asset Registration', 'Data Enrichment', 'Master Data Store']
            }
        }
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """
        Get information about a dataset
        """
        return self.datasets.get(dataset_name, {})
    
    def update_dataset_metadata(self, dataset_name: str, metadata: Dict):
        """
        Update dataset metadata
        """
        if dataset_name in self.datasets:
            self.datasets[dataset_name].update(metadata)
            self.datasets[dataset_name]['last_updated'] = datetime.now().isoformat()
            return True
        return False
    
    def search_datasets(self, query: str) -> List[Dict]:
        """
        Search datasets by name or description
        """
        results = []
        query_lower = query.lower()
        
        for name, info in self.datasets.items():
            if (query_lower in name.lower() or 
                query_lower in info.get('description', '').lower()):
                results.append({'name': name, **info})
        
        return results
    
    def get_data_lineage(self, dataset_name: str) -> List[str]:
        """
        Get data lineage for a dataset
        """
        dataset = self.datasets.get(dataset_name, {})
        return dataset.get('lineage', [])

class AccessControl:
    """
    Access control management for data governance
    """
    
    def __init__(self):
        self.access_logs = []
        self.access_rules = self._load_access_rules()
    
    def _load_access_rules(self):
        """
        Load access control rules
        """
        return {
            'role_permissions': {
                'administrator': ['read', 'write', 'delete', 'admin'],
                'manager': ['read', 'write', 'approve'],
                'operator': ['read', 'write'],
                'analyst': ['read'],
                'viewer': ['read']
            },
            'data_access_matrix': {
                'sensor_readings': {
                    'administrator': 'full',
                    'manager': 'read_write',
                    'operator': 'read_write',
                    'analyst': 'read',
                    'viewer': 'read'
                },
                'infrastructure_assets': {
                    'administrator': 'full',
                    'manager': 'read_write',
                    'operator': 'read',
                    'analyst': 'read',
                    'viewer': 'none'
                }
            }
        }
    
    def check_access(self, user_role: str, resource: str, operation: str) -> bool:
        """
        Check if user has access to perform operation on resource
        """
        if resource in self.access_rules['data_access_matrix']:
            access_level = self.access_rules['data_access_matrix'][resource].get(user_role, 'none')
            
            if access_level == 'full':
                return True
            elif access_level == 'read_write' and operation in ['read', 'write']:
                return True
            elif access_level == 'read' and operation == 'read':
                return True
        
        return False
    
    def log_access(self, user: str, resource: str, operation: str, result: str):
        """
        Log access attempt
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'resource': resource,
            'operation': operation,
            'result': result,
            'ip_address': f"192.168.1.{np.random.randint(100, 200)}"  # Simulated
        }
        
        self.access_logs.append(log_entry)
    
    def get_access_logs(self, limit: int = 100) -> List[Dict]:
        """
        Get recent access logs
        """
        return self.access_logs[-limit:]
    
    def generate_access_report(self) -> Dict:
        """
        Generate access control report
        """
        if not self.access_logs:
            return {'error': 'No access logs available'}
        
        logs_df = pd.DataFrame(self.access_logs)
        
        report = {
            'total_accesses': len(logs_df),
            'successful_accesses': len(logs_df[logs_df['result'] == 'success']),
            'failed_accesses': len(logs_df[logs_df['result'] == 'failed']),
            'unique_users': logs_df['user'].nunique(),
            'most_accessed_resources': logs_df['resource'].value_counts().head().to_dict(),
            'access_by_operation': logs_df['operation'].value_counts().to_dict()
        }
        
        return report

class ComplianceManager:
    """
    Compliance management for various regulatory frameworks
    """
    
    def __init__(self):
        self.frameworks = {
            'GDPR': {
                'requirements': [
                    'Data minimization',
                    'Consent management',
                    'Right to erasure',
                    'Data portability',
                    'Breach notification'
                ],
                'compliance_score': 96
            },
            'SOX': {
                'requirements': [
                    'Internal controls',
                    'Financial reporting accuracy',
                    'Audit trails',
                    'Change management'
                ],
                'compliance_score': 94
            },
            'ISO27001': {
                'requirements': [
                    'Information security management',
                    'Risk assessment',
                    'Access controls',
                    'Incident management'
                ],
                'compliance_score': 92
            }
        }
    
    def evaluate_compliance(self, framework: str) -> Dict:
        """
        Evaluate compliance for a specific framework
        """
        if framework not in self.frameworks:
            return {'error': f'Framework {framework} not supported'}
        
        framework_data = self.frameworks[framework]
        
        # Simulate compliance evaluation
        compliant_requirements = int(len(framework_data['requirements']) * 0.9)
        
        evaluation = {
            'framework': framework,
            'total_requirements': len(framework_data['requirements']),
            'compliant_requirements': compliant_requirements,
            'compliance_percentage': (compliant_requirements / len(framework_data['requirements'])) * 100,
            'non_compliant_items': framework_data['requirements'][-1:],  # Last item as non-compliant
            'last_assessment': datetime.now().strftime('%Y-%m-%d'),
            'next_assessment': (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
        }
        
        return evaluation
    
    def generate_compliance_dashboard(self) -> Dict:
        """
        Generate compliance dashboard data
        """
        dashboard = {
            'overall_compliance': np.mean([f['compliance_score'] for f in self.frameworks.values()]),
            'framework_scores': {name: data['compliance_score'] for name, data in self.frameworks.items()},
            'critical_gaps': [
                'Automated breach detection needs improvement',
                'Data retention policy enforcement',
                'Third-party risk assessment'
            ],
            'recent_updates': [
                {'date': '2023-07-10', 'framework': 'GDPR', 'change': 'Updated consent management process'},
                {'date': '2023-07-05', 'framework': 'SOX', 'change': 'Enhanced audit trail logging'}
            ]
        }
        
        return dashboard

def create_data_lineage_graph(dataset_name: str) -> Dict:
    """
    Create data lineage graph for visualization
    """
    lineage_data = {
        'sensor_readings': {
            'nodes': [
                {'id': 'sensors', 'label': 'IoT Sensors', 'type': 'source'},
                {'id': 'ingestion', 'label': 'Data Ingestion', 'type': 'process'},
                {'id': 'validation', 'label': 'Data Validation', 'type': 'process'},
                {'id': 'storage', 'label': 'Time Series DB', 'type': 'storage'},
                {'id': 'analytics', 'label': 'Analytics Engine', 'type': 'process'},
                {'id': 'dashboard', 'label': 'Dashboard', 'type': 'output'}
            ],
            'edges': [
                {'from': 'sensors', 'to': 'ingestion'},
                {'from': 'ingestion', 'to': 'validation'},
                {'from': 'validation', 'to': 'storage'},
                {'from': 'storage', 'to': 'analytics'},
                {'from': 'analytics', 'to': 'dashboard'}
            ]
        }
    }
    
    return lineage_data.get(dataset_name, {'nodes': [], 'edges': []})

def calculate_data_governance_score() -> Dict:
    """
    Calculate overall data governance score
    """
    components = {
        'policy_coverage': np.random.uniform(85, 95),
        'compliance_adherence': np.random.uniform(90, 98),
        'data_quality': np.random.uniform(88, 96),
        'access_control': np.random.uniform(92, 99),
        'documentation': np.random.uniform(80, 90)
    }
    
    overall_score = np.mean(list(components.values()))
    
    return {
        'overall_score': overall_score,
        'components': components,
        'grade': 'A' if overall_score >= 90 else 'B' if overall_score >= 80 else 'C',
        'recommendations': [
            'Improve documentation coverage',
            'Automate compliance monitoring',
            'Enhance data quality checks'
        ]
    }

# Helper functions for external use
governance = DataGovernance()
catalog = DataCatalog()

def get_governance_policies():
    """Get all governance policies"""
    return governance.policies

def check_compliance():
    """Check overall compliance status"""
    return governance.generate_compliance_report()

def get_retention_policy():
    """Get data retention policies"""
    return governance.policies.get('data_retention', {})
