"""
Database initialization script for SIMS
Run this script to set up the PostgreSQL database with initial data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.database import init_database, populate_initial_data, check_database_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database and populate with initial data"""
    
    # Check database connection
    if not check_database_connection():
        logger.error("Failed to connect to database. Please check your database configuration.")
        return False
    
    logger.info("Database connection successful")
    
    # Initialize database tables
    if not init_database():
        logger.error("Failed to initialize database tables")
        return False
    
    # Populate with initial data
    if not populate_initial_data():
        logger.error("Failed to populate initial data")
        return False
    
    logger.info("Database initialization completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)