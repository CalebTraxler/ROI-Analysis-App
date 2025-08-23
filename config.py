"""
Configuration file for ROI Analysis Application

This file contains all the configurable parameters for the application,
including performance settings, caching parameters, and geocoding options.
"""

import os
from pathlib import Path

# Application Settings
APP_TITLE = "3D Neighborhood ROI Analysis"
APP_LAYOUT = "wide"

# Performance Settings
MAX_WORKERS = int(os.getenv('MAX_WORKERS', 5))  # Parallel geocoding workers
GEOCODING_TIMEOUT = int(os.getenv('GEOCODING_TIMEOUT', 15))  # Seconds
MAX_RETRIES = int(os.getenv('MAX_RETRIES', 3))  # Geocoding retry attempts
RATE_LIMIT_PAUSE = int(os.getenv('RATE_LIMIT_PAUSE', 1))  # Seconds between batches
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 10))  # Locations per batch

# Caching Settings
COORDINATE_CACHE_TTL = int(os.getenv('COORDINATE_CACHE_TTL', 86400))  # 24 hours
DATA_CACHE_TTL = int(os.getenv('DATA_CACHE_TTL', 3600))  # 1 hour
MAIN_DATASET_CACHE_TTL = int(os.getenv('MAIN_DATASET_CACHE_TTL', 86400))  # 24 hours

# File Paths
CACHE_DIR = Path("cache")
GEOCODE_CACHE_FILE = CACHE_DIR / "geocode_cache.pkl"
PROCESSED_DATA_CACHE = CACHE_DIR / "processed_data_cache.pkl"
COORDINATES_DB = "coordinates_cache.db"

# Database Settings
DB_TIMEOUT = int(os.getenv('DB_TIMEOUT', 30))  # SQLite timeout

# Geocoding Service Settings
GEOCODING_USER_AGENT = "my_roi_app_v2"
GEOCODING_RATE_LIMIT = 1  # Requests per second (Nominatim limit)

# Map Visualization Settings
MAP_STYLE = 'mapbox://styles/mapbox/dark-v10'
HEATMAP_RADIUS = 60
HEATMAP_INTENSITY = 2
HEATMAP_THRESHOLD = 0.02
SCATTER_RADIUS = 30
SCATTER_OPACITY = 0.8

# Color Settings for ROI Visualization
COLOR_RANGES = {
    'heatmap': [
        [255, 255, 178, 100],  # Light yellow
        [254, 204, 92, 150],   # Yellow
        [253, 141, 60, 200],   # Orange
        [240, 59, 32, 250],    # Red-Orange
        [189, 0, 38, 255]      # Deep Red
    ],
    'scatter': {
        'red': 255,
        'green_base': 140,
        'blue': 0,
        'alpha': 180
    }
}

# Data Processing Settings
PAGE_SIZE = int(os.getenv('PAGE_SIZE', 50))  # Data table pagination
SEARCH_MIN_LENGTH = int(os.getenv('SEARCH_MIN_LENGTH', 2))  # Minimum search term length

# Logging Settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'roi_analysis.log'

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING = os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true').lower() == 'true'
PERFORMANCE_METRICS_TTL = int(os.getenv('PERFORMANCE_METRICS_TTL', 300))  # 5 minutes

# Error Handling
ENABLE_ERROR_RECOVERY = os.getenv('ENABLE_ERROR_RECOVERY', 'true').lower() == 'true'
MAX_ERROR_RETRIES = int(os.getenv('MAX_ERROR_RETRIES', 3))

# Development Settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
SHOW_PERFORMANCE_INFO = os.getenv('SHOW_PERFORMANCE_INFO', 'true').lower() == 'true'

# Cache Management
ENABLE_CACHE_CLEANUP = os.getenv('ENABLE_CACHE_CLEANUP', 'true').lower() == 'true'
CACHE_CLEANUP_INTERVAL = int(os.getenv('CACHE_CLEANUP_INTERVAL', 86400))  # 24 hours
MAX_CACHE_SIZE_MB = int(os.getenv('MAX_CACHE_SIZE_MB', 100))  # Maximum cache size in MB

def get_config_summary():
    """Return a summary of current configuration"""
    return {
        'performance': {
            'max_workers': MAX_WORKERS,
            'geocoding_timeout': GEOCODING_TIMEOUT,
            'rate_limit_pause': RATE_LIMIT_PAUSE,
            'batch_size': BATCH_SIZE
        },
        'caching': {
            'coordinate_cache_ttl': COORDINATE_CACHE_TTL,
            'data_cache_ttl': DATA_CACHE_TTL,
            'main_dataset_cache_ttl': MAIN_DATASET_CACHE_TTL
        },
        'geocoding': {
            'user_agent': GEOCODING_USER_AGENT,
            'rate_limit': GEOCODING_RATE_LIMIT,
            'max_retries': MAX_RETRIES
        },
        'development': {
            'debug_mode': DEBUG_MODE,
            'show_performance_info': SHOW_PERFORMANCE_INFO,
            'enable_performance_monitoring': ENABLE_PERFORMANCE_MONITORING
        }
    }

def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    if MAX_WORKERS < 1 or MAX_WORKERS > 20:
        errors.append("MAX_WORKERS must be between 1 and 20")
    
    if GEOCODING_TIMEOUT < 5 or GEOCODING_TIMEOUT > 60:
        errors.append("GEOCODING_TIMEOUT must be between 5 and 60 seconds")
    
    if RATE_LIMIT_PAUSE < 0 or RATE_LIMIT_PAUSE > 10:
        errors.append("RATE_LIMIT_PAUSE must be between 0 and 10 seconds")
    
    if BATCH_SIZE < 1 or BATCH_SIZE > 100:
        errors.append("BATCH_SIZE must be between 1 and 100")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    return True
