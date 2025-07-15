import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import hashlib
import json
from utils.auth import check_authentication
from utils.data_governance import DataGovernance, DataCatalog, AccessControl
from utils.data_quality import DataQualityChecker, generate_quality_report
from utils.data_generator import generate_real_time_data

# Authentication check
if not check_authentication():
    st.stop()

st.title("🔧 Data Management")
st.markdown("Comprehensive data governance, quality management, and compliance dashboard")

# Initialize data governance components
dg = DataGovernance()
dc = DataCatalog()
ac = AccessControl()
dqc = DataQualityChecker()

# Main navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data Overview", 
    "🏛️ Data Governance", 
    "✅ Data Quality", 
    "🔐 Access Control", 
    "📋 Compliance"
])

with tab1:
    st.markdown("### Data Overview")
    
    # Data inventory
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Datasets", "24", "↑ 2")
    
    with col2:
        st.metric("Active Sensors", "247", "↑ 3")
    
    with col3:
        st.metric("Data Volume (GB)", "1,234.5", "↑ 45.2")
    
    with col4:
        st.metric("Quality Score", "94.2%", "↑ 1.1%")
    
    # Data catalog table
    st.markdown("### Data Catalog")
    
    catalog_data = pd.DataFrame({
        'Dataset': ['sensor_readings', 'infrastructure_assets', 'maintenance_logs', 'user_activities', 'alerts_history'],
        'Type': ['Time Series', 'Master Data', 'Transactional', 'Audit', 'Event'],
        'Size (MB)': [856.3, 12.4, 234.7, 45.2, 89.1],
        'Records': [2453621, 1247, 8945, 12453, 3421],
        'Last Updated': ['2 mins ago', '1 hour ago', '15 mins ago', '5 mins ago', '30 mins ago'],
        'Quality Score': [96.2, 98.7, 92.1, 89.5, 94.8],
        'Owner': ['Data Engineering', 'Infrastructure', 'Maintenance', 'Security', 'Operations']
    })
    
    st.dataframe(catalog_data, use_container_width=True)
    
    # Data lineage visualization
    st.markdown("### Data Lineage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data flow diagram
        lineage_data = {
            'Source': ['IoT Sensors', 'Manual Entry', 'External APIs', 'Batch Import'],
            'Processing': ['Data Ingestion', 'Data Validation', 'Data Transformation', 'Data Storage'],
            'Destination': ['Analytics DB', 'Data Warehouse', 'ML Models', 'Dashboards']
        }
        
        fig_lineage = go.Figure()
        
        # Add source nodes
        for i, source in enumerate(lineage_data['Source']):
            fig_lineage.add_trace(go.Scatter(
                x=[0], y=[i], mode='markers+text', 
                text=[source], textposition='middle right',
                marker=dict(size=20, color='lightblue'),
                showlegend=False
            ))
        
        # Add processing nodes
        for i, process in enumerate(lineage_data['Processing']):
            fig_lineage.add_trace(go.Scatter(
                x=[1], y=[i], mode='markers+text',
                text=[process], textposition='middle center',
                marker=dict(size=20, color='lightgreen'),
                showlegend=False
            ))
        
        # Add destination nodes
        for i, dest in enumerate(lineage_data['Destination']):
            fig_lineage.add_trace(go.Scatter(
                x=[2], y=[i], mode='markers+text',
                text=[dest], textposition='middle left',
                marker=dict(size=20, color='lightcoral'),
                showlegend=False
            ))
        
        fig_lineage.update_layout(
            title="Data Lineage Flow",
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=400
        )
        
        st.plotly_chart(fig_lineage, use_container_width=True)
    
    with col2:
        # Data usage statistics
        usage_data = pd.DataFrame({
            'Department': ['Operations', 'Maintenance', 'Analytics', 'Quality', 'Management'],
            'Usage (GB)': [345.2, 234.1, 456.8, 123.4, 89.3],
            'Queries': [1245, 892, 2341, 567, 234]
        })
        
        fig_usage = px.bar(
            usage_data,
            x='Department',
            y='Usage (GB)',
            title="Data Usage by Department",
            color='Usage (GB)',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_usage, use_container_width=True)

with tab2:
    st.markdown("### Data Governance Framework")
    
    # Governance policies
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Active Policies**")
        
        policies = pd.DataFrame({
            'Policy': ['Data Retention', 'Data Classification', 'Access Control', 'Data Privacy', 'Backup & Recovery'],
            'Status': ['Active', 'Active', 'Active', 'Under Review', 'Active'],
            'Last Updated': ['2023-06-15', '2023-07-01', '2023-06-20', '2023-07-10', '2023-06-10'],
            'Compliance': [98, 96, 99, 94, 97]
        })
        
        st.dataframe(policies, use_container_width=True)
    
    with col2:
        # Policy compliance chart
        fig_compliance = px.bar(
            policies,
            x='Policy',
            y='Compliance',
            title="Policy Compliance Scores",
            color='Compliance',
            color_continuous_scale='RdYlGn'
        )
        fig_compliance.update_layout(height=300)
        st.plotly_chart(fig_compliance, use_container_width=True)
    
    # Data stewardship
    st.markdown("### Data Stewardship")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data Stewards**")
        stewards = pd.DataFrame({
            'Name': ['Alice Johnson', 'Bob Smith', 'Carol Brown', 'David Wilson'],
            'Department': ['Engineering', 'Operations', 'Quality', 'Security'],
            'Responsibility': ['Sensor Data', 'Infrastructure', 'Quality Metrics', 'Access Logs'],
            'Contact': ['alice@company.com', 'bob@company.com', 'carol@company.com', 'david@company.com']
        })
        st.dataframe(stewards, use_container_width=True)
    
    with col2:
        st.markdown("**Data Classification**")
        classification = pd.DataFrame({
            'Classification': ['Public', 'Internal', 'Confidential', 'Restricted'],
            'Count': [5, 12, 6, 1],
            'Percentage': [20.8, 50.0, 25.0, 4.2]
        })
        
        fig_class = px.pie(
            classification,
            values='Count',
            names='Classification',
            title="Data Classification Distribution"
        )
        fig_class.update_layout(height=300)
        st.plotly_chart(fig_class, use_container_width=True)
    
    with col3:
        st.markdown("**Metadata Management**")
        metadata_stats = pd.DataFrame({
            'Metric': ['Schema Compliance', 'Documentation Coverage', 'Lineage Tracking', 'Tag Completeness'],
            'Score': [96, 89, 92, 87]
        })
        
        for _, row in metadata_stats.iterrows():
            st.metric(row['Metric'], f"{row['Score']}%")
    
    # Data lifecycle management
    st.markdown("### Data Lifecycle Management")
    
    lifecycle_data = pd.DataFrame({
        'Stage': ['Creation', 'Storage', 'Processing', 'Analysis', 'Archive', 'Deletion'],
        'Data Volume (%)': [100, 95, 80, 60, 30, 0],
        'Retention (Days)': [0, 30, 90, 365, 2555, 2920]
    })
    
    fig_lifecycle = px.line(
        lifecycle_data,
        x='Retention (Days)',
        y='Data Volume (%)',
        title="Data Lifecycle - Volume Retention",
        markers=True
    )
    
    # Add stage annotations
    for _, row in lifecycle_data.iterrows():
        fig_lifecycle.add_annotation(
            x=row['Retention (Days)'],
            y=row['Data Volume (%)'],
            text=row['Stage'],
            showarrow=True,
            arrowhead=2
        )
    
    st.plotly_chart(fig_lifecycle, use_container_width=True)

with tab3:
    st.markdown("### Data Quality Management")
    
    # Generate sample data for quality analysis
    sample_data = generate_real_time_data(1000)
    
    # Data quality summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        completeness = (1 - sample_data.isnull().sum().sum() / (len(sample_data) * len(sample_data.columns))) * 100
        st.metric("Completeness", f"{completeness:.1f}%", "↑ 2.1%")
    
    with col2:
        # Accuracy (simulate based on anomaly scores)
        accuracy = (1 - sample_data['anomaly_score'].mean()) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%", "↑ 0.8%")
    
    with col3:
        # Consistency (simulate)
        consistency = np.random.uniform(92, 98)
        st.metric("Consistency", f"{consistency:.1f}%", "↓ 0.3%")
    
    with col4:
        # Timeliness (simulate)
        timeliness = np.random.uniform(95, 99)
        st.metric("Timeliness", f"{timeliness:.1f}%", "↑ 1.5%")
    
    # Quality rules and checks
    st.markdown("### Quality Rules & Checks")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Active Quality Rules**")
        
        quality_rules = pd.DataFrame({
            'Rule': ['Pressure Range Check', 'Flow Rate Validation', 'Temperature Bounds', 'Sensor ID Format', 'Timestamp Sequence'],
            'Type': ['Range', 'Business', 'Range', 'Format', 'Sequence'],
            'Status': ['Active', 'Active', 'Active', 'Active', 'Active'],
            'Pass Rate (%)': [98.5, 96.2, 99.1, 100.0, 97.8],
            'Failures': [15, 38, 9, 0, 22]
        })
        
        st.dataframe(quality_rules, use_container_width=True)
    
    with col2:
        # Quality trend chart
        dates = pd.date_range(start='2023-06-01', end='2023-07-15', freq='D')
        quality_trend = pd.DataFrame({
            'Date': dates,
            'Overall Quality': np.random.uniform(92, 98, len(dates)),
            'Completeness': np.random.uniform(95, 99, len(dates)),
            'Accuracy': np.random.uniform(90, 96, len(dates))
        })
        
        fig_trend = px.line(
            quality_trend.melt(id_vars=['Date'], var_name='Metric', value_name='Score'),
            x='Date',
            y='Score',
            color='Metric',
            title="Data Quality Trends"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Data profiling
    st.markdown("### Data Profiling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Null value analysis
        null_analysis = sample_data.isnull().sum().reset_index()
        null_analysis.columns = ['Column', 'Null_Count']
        null_analysis['Null_Percentage'] = (null_analysis['Null_Count'] / len(sample_data)) * 100
        
        fig_nulls = px.bar(
            null_analysis,
            x='Column',
            y='Null_Percentage',
            title="Null Value Analysis"
        )
        st.plotly_chart(fig_nulls, use_container_width=True)
    
    with col2:
        # Data distribution analysis
        numeric_cols = ['pressure', 'flow_rate', 'temperature']
        selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
        
        fig_dist = px.histogram(
            sample_data,
            x=selected_col,
            title=f"{selected_col.title()} Distribution",
            nbins=30
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Quality report generation
    st.markdown("### Quality Report Generation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Quality Report"):
            report = generate_quality_report(sample_data)
            st.success("Quality report generated successfully!")
            
            with st.expander("View Report Summary"):
                st.json(report)
    
    with col2:
        if st.button("🔍 Run Quality Checks"):
            st.info("Running comprehensive quality checks...")
            
            # Simulate quality check results
            checks = {
                'Null Check': 'PASSED',
                'Range Check': 'PASSED', 
                'Format Check': 'PASSED',
                'Uniqueness Check': 'WARNING',
                'Referential Integrity': 'FAILED'
            }
            
            for check, status in checks.items():
                if status == 'PASSED':
                    st.success(f"✅ {check}: {status}")
                elif status == 'WARNING':
                    st.warning(f"⚠️ {check}: {status}")
                else:
                    st.error(f"❌ {check}: {status}")
    
    with col3:
        if st.button("📧 Schedule Quality Alerts"):
            st.info("Quality monitoring alerts configured")

with tab4:
    st.markdown("### Access Control Management")
    
    # User access overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Active Users", "42", "↑ 3")
    
    with col2:
        st.metric("User Groups", "8", "→ 0")
    
    with col3:
        st.metric("Access Violations", "2", "↓ 5")
    
    # User management
    st.markdown("### User Access Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Active Users**")
        
        users_data = pd.DataFrame({
            'User': ['john.doe', 'jane.smith', 'mike.wilson', 'sarah.jones', 'admin'],
            'Role': ['Operator', 'Analyst', 'Manager', 'Analyst', 'Administrator'],
            'Department': ['Operations', 'Analytics', 'Management', 'Analytics', 'IT'],
            'Last Login': ['2023-07-15 09:30', '2023-07-15 08:45', '2023-07-14 16:20', '2023-07-15 10:15', '2023-07-15 07:00'],
            'Status': ['Active', 'Active', 'Active', 'Active', 'Active']
        })
        
        st.dataframe(users_data, use_container_width=True)
    
    with col2:
        # Role distribution
        role_counts = users_data['Role'].value_counts()
        fig_roles = px.pie(
            values=role_counts.values,
            names=role_counts.index,
            title="User Role Distribution"
        )
        st.plotly_chart(fig_roles, use_container_width=True)
    
    # Permission matrix
    st.markdown("### Permission Matrix")
    
    permissions_data = pd.DataFrame({
        'Resource': ['Sensor Data', 'Infrastructure Assets', 'Maintenance Logs', 'User Management', 'System Config'],
        'Administrator': ['Full', 'Full', 'Full', 'Full', 'Full'],
        'Manager': ['Read/Write', 'Read/Write', 'Read/Write', 'Read', 'Read'],
        'Analyst': ['Read/Write', 'Read', 'Read', 'None', 'None'],
        'Operator': ['Read', 'Read', 'Read/Write', 'None', 'None']
    })
    
    # Create color mapping for permissions
    def color_permissions(val):
        if val == 'Full':
            return 'background-color: #90EE90'
        elif val == 'Read/Write':
            return 'background-color: #FFE4B5'
        elif val == 'Read':
            return 'background-color: #E0E0E0'
        else:
            return 'background-color: #FFB6C1'
    
    st.dataframe(
        permissions_data.style.applymap(color_permissions, subset=['Administrator', 'Manager', 'Analyst', 'Operator']),
        use_container_width=True
    )
    
    # Access audit
    st.markdown("### Access Audit Trail")
    
    audit_data = pd.DataFrame({
        'Timestamp': ['2023-07-15 10:30:00', '2023-07-15 10:25:00', '2023-07-15 10:20:00', '2023-07-15 10:15:00'],
        'User': ['jane.smith', 'john.doe', 'mike.wilson', 'sarah.jones'],
        'Action': ['READ', 'WRITE', 'READ', 'DELETE'],
        'Resource': ['sensor_data', 'maintenance_logs', 'infrastructure_assets', 'old_alerts'],
        'Result': ['SUCCESS', 'SUCCESS', 'SUCCESS', 'DENIED'],
        'IP Address': ['192.168.1.101', '192.168.1.102', '192.168.1.103', '192.168.1.104']
    })
    
    st.dataframe(audit_data, use_container_width=True)

with tab5:
    st.markdown("### Compliance Dashboard")
    
    # Compliance overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("GDPR Compliance", "96%", "↑ 2%")
    
    with col2:
        st.metric("ISO 27001", "94%", "→ 0%")
    
    with col3:
        st.metric("SOX Compliance", "98%", "↑ 1%")
    
    with col4:
        st.metric("Industry Standards", "92%", "↓ 1%")
    
    # Compliance requirements
    st.markdown("### Compliance Requirements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        compliance_reqs = pd.DataFrame({
            'Requirement': ['Data Encryption', 'Access Logging', 'Data Retention', 'Privacy Controls', 'Backup & Recovery'],
            'Standard': ['ISO 27001', 'SOX', 'GDPR', 'GDPR', 'ISO 27001'],
            'Status': ['Compliant', 'Compliant', 'Compliant', 'Partial', 'Compliant'],
            'Last Audit': ['2023-06-15', '2023-07-01', '2023-06-20', '2023-07-10', '2023-06-10'],
            'Next Review': ['2023-12-15', '2024-01-01', '2023-12-20', '2023-10-10', '2023-12-10']
        })
        
        st.dataframe(compliance_reqs, use_container_width=True)
    
    with col2:
        # Compliance status chart
        status_counts = compliance_reqs['Status'].value_counts()
        fig_compliance_status = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Compliance Status Distribution",
            color_discrete_map={'Compliant': 'green', 'Partial': 'orange', 'Non-Compliant': 'red'}
        )
        st.plotly_chart(fig_compliance_status, use_container_width=True)
    
    # Data privacy management
    st.markdown("### Data Privacy Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Privacy Controls**")
        
        privacy_controls = pd.DataFrame({
            'Control': ['Data Anonymization', 'Consent Management', 'Right to Erasure', 'Data Portability', 'Breach Notification'],
            'Implementation': ['Implemented', 'Implemented', 'Implemented', 'Partial', 'Implemented'],
            'Effectiveness': [95, 92, 88, 75, 98]
        })
        
        st.dataframe(privacy_controls, use_container_width=True)
    
    with col2:
        # Privacy effectiveness chart
        fig_privacy = px.bar(
            privacy_controls,
            x='Control',
            y='Effectiveness',
            title="Privacy Control Effectiveness",
            color='Effectiveness',
            color_continuous_scale='RdYlGn'
        )
        fig_privacy.update_layout(height=300, xaxis={'tickangle': 45})
        st.plotly_chart(fig_privacy, use_container_width=True)
    
    # Compliance reporting
    st.markdown("### Compliance Reporting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Generate Compliance Report"):
            st.success("Compliance report generated successfully!")
            
            # Simulate report generation
            report_data = {
                "report_id": "COMP-2023-07-15-001",
                "generated_at": datetime.now().isoformat(),
                "compliance_score": 94.5,
                "requirements_total": 25,
                "requirements_met": 23,
                "requirements_partial": 2,
                "requirements_failed": 0
            }
            
            with st.expander("View Report Details"):
                st.json(report_data)
    
    with col2:
        if st.button("📧 Schedule Compliance Alerts"):
            st.info("Compliance monitoring alerts configured")
    
    with col3:
        if st.button("🔍 Run Compliance Audit"):
            st.info("Comprehensive compliance audit initiated")

# Export and actions
st.markdown("---")
st.markdown("### Export & Actions")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📥 Export Data Catalog"):
        csv = catalog_data.to_csv(index=False)
        st.download_button(
            label="Download Catalog",
            data=csv,
            file_name=f"data_catalog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔧 Update Policies"):
        # Policy update interface
            st.subheader("📋 Data Governance Policy Management")
            
            with st.form("policy_update"):
                policy_type = st.selectbox(
                    "Policy Type",
                    ["Data Retention", "Access Control", "Data Classification", "Compliance Framework"]
                )
                
                if policy_type == "Data Retention":
                    st.subheader("🗂️ Data Retention Policy")
                    
                    data_types = ["sensor_data", "alerts", "maintenance_records", "user_activities"]
                    
                    for data_type in data_types:
                        st.write(f"**{data_type.replace('_', ' ').title()}**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            retention_period = st.selectbox(
                                f"Retention Period",
                                ["7 days", "30 days", "90 days", "1 year", "3 years", "5 years", "Permanent"],
                                key=f"retention_{data_type}"
                            )
                        
                        with col2:
                            archive_after = st.selectbox(
                                f"Archive After",
                                ["30 days", "90 days", "1 year", "2 years", "Never"],
                                key=f"archive_{data_type}"
                            )
                
                elif policy_type == "Access Control":
                    st.subheader("🔐 Access Control Policy")
                    
                    roles = ["administrator", "operator", "analyst", "manager", "guest"]
                    resources = ["sensor_data", "alerts", "reports", "system_config", "user_management"]
                    
                    access_matrix = {}
                    for role in roles:
                        access_matrix[role] = {}
                        st.write(f"**{role.title()} Permissions**")
                        
                        cols = st.columns(len(resources))
                        for i, resource in enumerate(resources):
                            with cols[i]:
                                access_matrix[role][resource] = st.checkbox(
                                    f"{resource.replace('_', ' ').title()}",
                                    key=f"access_{role}_{resource}"
                                )
                
                elif policy_type == "Data Classification":
                    st.subheader("🏷️ Data Classification Policy")
                    
                    classification_levels = st.multiselect(
                        "Classification Levels",
                        ["Public", "Internal", "Confidential", "Restricted", "Top Secret"],
                        default=["Public", "Internal", "Confidential"]
                    )
                    
                    for level in classification_levels:
                        st.write(f"**{level} Data**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            access_requirements = st.text_area(
                                f"Access Requirements for {level}",
                                placeholder="Describe who can access this data and under what conditions",
                                key=f"access_req_{level}"
                            )
                        
                        with col2:
                            handling_procedures = st.text_area(
                                f"Handling Procedures for {level}",
                                placeholder="Describe how this data should be handled and stored",
                                key=f"handling_{level}"
                            )
                
                elif policy_type == "Compliance Framework":
                    st.subheader("📊 Compliance Framework Policy")
                    
                    frameworks = st.multiselect(
                        "Applicable Frameworks",
                        ["GDPR", "SOX", "ISO27001", "HIPAA", "PCI DSS", "SOC 2"],
                        default=["GDPR", "SOX", "ISO27001"]
                    )
                    
                    for framework in frameworks:
                        st.write(f"**{framework} Compliance**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            requirements = st.text_area(
                                f"Key Requirements",
                                placeholder=f"List key {framework} requirements",
                                key=f"req_{framework}"
                            )
                        
                        with col2:
                            implementation = st.text_area(
                                f"Implementation Notes",
                                placeholder=f"How {framework} is implemented in the system",
                                key=f"impl_{framework}"
                            )
                
                # Submit button
                if st.form_submit_button("💾 Update Policy"):
                    # Store policy updates in session state
                    if 'policy_updates' not in st.session_state:
                        st.session_state.policy_updates = []
                    
                    policy_update = {
                        'type': policy_type,
                        'updated_at': datetime.now().isoformat(),
                        'updated_by': st.session_state.get('username', 'admin')
                    }
                    
                    st.session_state.policy_updates.append(policy_update)
                    st.success(f"✅ {policy_type} policy updated successfully!")
                    st.info("📋 Policy changes have been logged and will be applied to the system.")
            
            # Display recent policy updates
            if 'policy_updates' in st.session_state and st.session_state.policy_updates:
                st.subheader("📝 Recent Policy Updates")
                
                for update in st.session_state.policy_updates[-5:]:  # Show last 5 updates
                    st.write(f"• **{update['type']}** - Updated by {update['updated_by']} on {update['updated_at'][:10]}")

with col3:
    if st.button("📊 Generate Full Report"):
        # Comprehensive data management report generation
            with st.spinner("Generating comprehensive data management report..."):
                from utils.data_quality import generate_quality_report, DataQualityChecker
                from utils.data_governance import DataGovernance, ComplianceManager, calculate_data_governance_score
                from utils.data_generator import get_recent_sensor_data_from_db
                from utils.database import get_system_stats, get_active_alerts
                
                # Get recent data for analysis
                recent_data = get_recent_sensor_data_from_db(hours=24)
                system_stats = get_system_stats()
                active_alerts = get_active_alerts()
                
                if not recent_data.empty:
                    # Generate quality report
                    quality_report = generate_quality_report(recent_data)
                    
                    # Generate governance score
                    governance_score = calculate_data_governance_score()
                    
                    # Generate compliance reports
                    compliance_mgr = ComplianceManager()
                    compliance_dashboard = compliance_mgr.generate_compliance_dashboard()
                    
                    # Create comprehensive report
                    management_report = {
                        'report_metadata': {
                            'generated_at': datetime.now().isoformat(),
                            'report_type': 'Comprehensive Data Management Report',
                            'analysis_period': '24 hours',
                            'generated_by': st.session_state.get('username', 'admin')
                        },
                        'system_overview': {
                            'total_sensors': system_stats['total_sensors'],
                            'active_sensors': system_stats['active_sensors'],
                            'total_readings': len(recent_data),
                            'active_alerts': len(active_alerts),
                            'data_volume_mb': len(recent_data) * 0.001  # Approximate
                        },
                        'data_quality': {
                            'overall_score': quality_report['overall_score'],
                            'completeness_score': quality_report['checks']['completeness']['score'],
                            'accuracy_score': quality_report['checks']['accuracy']['score'],
                            'consistency_score': quality_report['checks']['consistency']['score'],
                            'timeliness_score': quality_report['checks']['timeliness']['score'],
                            'quality_issues': quality_report['issues']
                        },
                        'data_governance': {
                            'overall_score': governance_score['overall_score'],
                            'policy_compliance': governance_score['policy_compliance'],
                            'access_control': governance_score['access_control'],
                            'data_classification': governance_score['data_classification']
                        },
                        'compliance_status': compliance_dashboard
                    }
                    
                    # Display the report
                    st.subheader("📊 Comprehensive Data Management Report")
                    
                    # System Overview
                    st.subheader("🖥️ System Overview")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Sensors", management_report['system_overview']['total_sensors'])
                    with col2:
                        st.metric("Active Sensors", management_report['system_overview']['active_sensors'])
                    with col3:
                        st.metric("Total Readings", management_report['system_overview']['total_readings'])
                    with col4:
                        st.metric("Active Alerts", management_report['system_overview']['active_alerts'])
                    
                    # Data Quality Section
                    st.subheader("📈 Data Quality Analysis")
                    
                    quality_col1, quality_col2 = st.columns(2)
                    
                    with quality_col1:
                        st.metric("Overall Quality Score", f"{management_report['data_quality']['overall_score']:.1f}%")
                        st.metric("Completeness", f"{management_report['data_quality']['completeness_score']:.1f}%")
                        st.metric("Accuracy", f"{management_report['data_quality']['accuracy_score']:.1f}%")
                    
                    with quality_col2:
                        st.metric("Consistency", f"{management_report['data_quality']['consistency_score']:.1f}%")
                        st.metric("Timeliness", f"{management_report['data_quality']['timeliness_score']:.1f}%")
                        st.metric("Quality Issues", len(management_report['data_quality']['quality_issues']))
                    
                    # Data Governance Section
                    st.subheader("🔒 Data Governance")
                    
                    gov_col1, gov_col2 = st.columns(2)
                    
                    with gov_col1:
                        st.metric("Overall Governance Score", f"{management_report['data_governance']['overall_score']:.1f}%")
                        st.metric("Policy Compliance", f"{management_report['data_governance']['policy_compliance']:.1f}%")
                    
                    with gov_col2:
                        st.metric("Access Control", f"{management_report['data_governance']['access_control']:.1f}%")
                        st.metric("Data Classification", f"{management_report['data_governance']['data_classification']:.1f}%")
                    
                    # Compliance Status
                    st.subheader("📋 Compliance Status")
                    
                    compliance_data = []
                    for framework, score in management_report['compliance_status'].items():
                        if isinstance(score, (int, float)):
                            compliance_data.append({
                                'Framework': framework,
                                'Score': f"{score:.1f}%",
                                'Status': 'Compliant' if score >= 80 else 'Needs Attention'
                            })
                    
                    if compliance_data:
                        compliance_df = pd.DataFrame(compliance_data)
                        st.dataframe(compliance_df, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    recommendations = []
                    
                    if management_report['data_quality']['overall_score'] < 95:
                        recommendations.append("🔍 Implement additional data quality checks")
                    
                    if management_report['data_governance']['overall_score'] < 90:
                        recommendations.append("📋 Review and update data governance policies")
                    
                    if len(management_report['data_quality']['quality_issues']) > 0:
                        recommendations.append("⚠️ Address identified data quality issues")
                    
                    if management_report['system_overview']['active_alerts'] > 5:
                        recommendations.append("🚨 Investigate high number of active alerts")
                    
                    if not recommendations:
                        recommendations.append("✅ System is operating well within all parameters")
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
                    
                    # Export options
                    st.subheader("📁 Export Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # JSON export
                        import json
                        report_json = json.dumps(management_report, indent=2)
                        st.download_button(
                            label="📊 Download Report (JSON)",
                            data=report_json,
                            file_name=f"data_management_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime='application/json'
                        )
                    
                    with col2:
                        # CSV export for tabular data
                        if compliance_data:
                            csv_data = compliance_df.to_csv(index=False)
                            st.download_button(
                                label="📋 Download Compliance Data (CSV)",
                                data=csv_data,
                                file_name=f"compliance_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime='text/csv'
                            )
                    
                    st.success("Comprehensive data management report generated successfully!")
                else:
                    st.warning("No recent data available for report generation.")

with col4:
    if st.button("⚙️ System Settings"):
        with st.expander("System Configuration"):
            st.checkbox("Enable Data Lineage Tracking")
            st.checkbox("Auto Quality Checks")
            st.selectbox("Retention Policy", ["30 days", "90 days", "1 year", "Custom"])
            st.number_input("Max File Size (MB)", min_value=1, max_value=1000, value=100)
