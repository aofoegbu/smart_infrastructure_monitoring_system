import streamlit as st
import hashlib
import os
from datetime import datetime, timedelta

# Default users for demonstration
DEFAULT_USERS = {
    'admin': {
        'password': 'admin123',
        'role': 'administrator',
        'permissions': ['read', 'write', 'admin', 'delete'],
        'department': 'IT'
    },
    'operator': {
        'password': 'operator123',
        'role': 'operator',
        'permissions': ['read', 'write'],
        'department': 'Operations'
    },
    'analyst': {
        'password': 'analyst123',
        'role': 'analyst',
        'permissions': ['read'],
        'department': 'Analytics'
    },
    'manager': {
        'password': 'manager123',
        'role': 'manager',
        'permissions': ['read', 'write', 'admin'],
        'department': 'Management'
    }
}

def hash_password(password):
    """
    Hash password using SHA-256
    """
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_password(password, hashed):
    """
    Verify password against hash
    """
    return hash_password(password) == hashed

def get_users():
    """
    Get user database (in production, this would be from a database)
    """
    return DEFAULT_USERS

def authenticate_user(username, password):
    """
    Authenticate user credentials
    """
    users = get_users()
    
    if username in users:
        # For demonstration, we're using plain text passwords
        # In production, passwords should be hashed
        if users[username]['password'] == password:
            return users[username]
    
    return None

def init_session_state():
    """
    Initialize session state variables
    """
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    
    if 'user_permissions' not in st.session_state:
        st.session_state.user_permissions = []
    
    if 'user_department' not in st.session_state:
        st.session_state.user_department = None
    
    if 'login_time' not in st.session_state:
        st.session_state.login_time = None

def login_form():
    """
    Display login form
    """
    st.title("🔐 SIMS Login")
    st.markdown("Smart Infrastructure Monitoring System")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            user = authenticate_user(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = user['role']
                st.session_state.user_permissions = user['permissions']
                st.session_state.user_department = user['department']
                st.session_state.login_time = datetime.now()
                
                # Log the login
                log_user_activity(username, 'LOGIN', 'Successful login')
                
                st.success(f"Welcome, {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    # Display demo credentials
    st.markdown("---")
    st.markdown("### Demo Credentials")
    st.info("""
    **Administrator:** admin / admin123  
    **Operator:** operator / operator123  
    **Analyst:** analyst / analyst123  
    **Manager:** manager / manager123
    """)

def logout():
    """
    Logout user and clear session
    """
    if st.session_state.username:
        log_user_activity(st.session_state.username, 'LOGOUT', 'User logged out')
    
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.user_permissions = []
    st.session_state.user_department = None
    st.session_state.login_time = None

def check_authentication():
    """
    Check if user is authenticated and show login form if not
    """
    init_session_state()
    
    if not st.session_state.authenticated:
        login_form()
        return False
    
    # Add logout button to sidebar
    with st.sidebar:
        st.markdown(f"**User:** {st.session_state.username}")
        st.markdown(f"**Role:** {st.session_state.user_role.title()}")
        st.markdown(f"**Department:** {st.session_state.user_department}")
        
        if st.button("🚪 Logout"):
            logout()
            st.rerun()
        
        # Session info
        if st.session_state.login_time:
            session_duration = datetime.now() - st.session_state.login_time
            st.markdown(f"**Session:** {str(session_duration).split('.')[0]}")
    
    return True

def check_permission(required_permission):
    """
    Check if current user has required permission
    """
    if not st.session_state.authenticated:
        return False
    
    return required_permission in st.session_state.user_permissions

def require_permission(required_permission, error_message="You don't have permission to access this feature."):
    """
    Decorator/function to require specific permission
    """
    if not check_permission(required_permission):
        st.error(error_message)
        st.stop()

def log_user_activity(username, action, details=""):
    """
    Log user activity (in production, this would go to a database)
    """
    timestamp = datetime.now().isoformat()
    log_entry = f"{timestamp} - {username} - {action} - {details}"
    
    # In a real application, this would be stored in a database
    # For now, we'll just store in session state for demonstration
    if 'activity_log' not in st.session_state:
        st.session_state.activity_log = []
    
    st.session_state.activity_log.append({
        'timestamp': timestamp,
        'username': username,
        'action': action,
        'details': details
    })

def get_user_activity_log():
    """
    Get user activity log
    """
    if 'activity_log' not in st.session_state:
        return []
    
    return st.session_state.activity_log

def create_user(username, password, role, permissions, department):
    """
    Create a new user (admin function)
    """
    require_permission('admin', "Only administrators can create users.")
    
    users = get_users()
    
    if username in users:
        return False, "User already exists"
    
    users[username] = {
        'password': password,  # In production, this should be hashed
        'role': role,
        'permissions': permissions,
        'department': department
    }
    
    log_user_activity(st.session_state.username, 'CREATE_USER', f"Created user: {username}")
    
    return True, "User created successfully"

def update_user_permissions(username, new_permissions):
    """
    Update user permissions (admin function)
    """
    require_permission('admin', "Only administrators can update permissions.")
    
    users = get_users()
    
    if username not in users:
        return False, "User not found"
    
    users[username]['permissions'] = new_permissions
    
    log_user_activity(st.session_state.username, 'UPDATE_PERMISSIONS', f"Updated permissions for: {username}")
    
    return True, "Permissions updated successfully"

def delete_user(username):
    """
    Delete a user (admin function)
    """
    require_permission('admin', "Only administrators can delete users.")
    
    if username == st.session_state.username:
        return False, "Cannot delete your own account"
    
    users = get_users()
    
    if username not in users:
        return False, "User not found"
    
    del users[username]
    
    log_user_activity(st.session_state.username, 'DELETE_USER', f"Deleted user: {username}")
    
    return True, "User deleted successfully"

def get_role_permissions():
    """
    Get default permissions for each role
    """
    return {
        'administrator': ['read', 'write', 'admin', 'delete'],
        'manager': ['read', 'write', 'admin'],
        'operator': ['read', 'write'],
        'analyst': ['read'],
        'viewer': ['read']
    }

def validate_access_to_data(data_classification):
    """
    Validate user access to data based on classification
    """
    user_role = st.session_state.user_role
    
    access_matrix = {
        'public': ['administrator', 'manager', 'operator', 'analyst', 'viewer'],
        'internal': ['administrator', 'manager', 'operator', 'analyst'],
        'confidential': ['administrator', 'manager'],
        'restricted': ['administrator']
    }
    
    return user_role in access_matrix.get(data_classification, [])

def get_user_profile():
    """
    Get current user profile
    """
    if not st.session_state.authenticated:
        return None
    
    return {
        'username': st.session_state.username,
        'role': st.session_state.user_role,
        'permissions': st.session_state.user_permissions,
        'department': st.session_state.user_department,
        'login_time': st.session_state.login_time
    }

def session_timeout_check(timeout_minutes=60):
    """
    Check for session timeout
    """
    if st.session_state.login_time:
        session_duration = datetime.now() - st.session_state.login_time
        if session_duration > timedelta(minutes=timeout_minutes):
            st.warning("Session expired. Please log in again.")
            logout()
            st.rerun()
