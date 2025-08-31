import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import sqlite3
import time
from pathlib import Path
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import requests
from openstreetmap_properties import OpenStreetMapProperties

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(
    page_title="Real Estate ROI Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Modern Professional Design System */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        margin-bottom: 3rem;
        text-align: center;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="white" opacity="0.1"/><circle cx="75" cy="75" r="1" fill="white" opacity="0.1"/><circle cx="50" cy="10" r="0.5" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }
    .main-header p {
        font-size: 1.4rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    .section-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.3);
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #ec4899);
    }
    .metric-container strong {
        color: #374151;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        display: block;
        margin-bottom: 0.5rem;
    }
    .metric-container span {
        display: block;
        margin-top: 0.5rem;
        font-weight: 700;
    }
    
    .sidebar-section {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .info-box {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #3b82f6;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #22c55e;
        color: #065f46;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(34, 197, 94, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        color: #92400e;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.1);
    }
    
    .card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
    }
    
    .btn-primary {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(79, 70, 229, 0.4);
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
GEOCODE_CACHE_FILE = CACHE_DIR / "geocode_cache.pkl"
PROCESSED_DATA_CACHE = CACHE_DIR / "processed_data_cache.pkl"

# Default coordinates for major US counties (fallback data)
DEFAULT_COORDINATES = {
    "Los Angeles County": (34.0522, -118.2437),
    "Cook County": (41.8781, -87.6298),
    "Harris County": (29.7604, -95.3698),
    "Maricopa County": (33.4484, -112.0740),
    "San Diego County": (32.7157, -117.1611),
    "Orange County": (33.7175, -117.8311),
    "Miami-Dade County": (25.7617, -80.1918),
    "Kings County": (40.6782, -73.9442),
    "Dallas County": (32.7767, -96.7970),
    "Queens County": (40.7282, -73.7949),
}

# Create a coordinates cache database with better indexing
def setup_coordinates_cache():
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS coordinates
                 (location_key TEXT PRIMARY KEY, latitude REAL, longitude REAL, 
                  state TEXT, county TEXT, timestamp REAL)''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_state_county ON coordinates(state, county)')
    conn.commit()
    conn.close()

# Check network connectivity
@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_network_status():
    """Check if external services are accessible"""
    services = {
        "geocoding": "https://nominatim.openstreetmap.org/status",
        "mapbox": "https://api.mapbox.com/v1/",
        "general": "https://httpbin.org/status/200"
    }
    
    status = {}
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            status[service] = response.status_code == 200 or response.status_code == 401  # 401 is OK for mapbox without token
        except Exception as e:
            logger.warning(f"Network check failed for {service}: {e}")
            status[service] = False
    
    return status

# Enhanced cache decorator for expensive computations
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_cached_location(location_key):
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('SELECT latitude, longitude FROM coordinates WHERE location_key = ?', (location_key,))
    result = c.fetchone()
    conn.close()
    return result

def save_location_to_cache(location_key, lat, lon, state, county):
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO coordinates 
                 (location_key, latitude, longitude, state, county, timestamp) 
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (location_key, lat, lon, state, county, time.time()))
    conn.commit()
    conn.close()

# Generate fallback coordinates using county center and random offsets
def generate_fallback_coordinates(locations, state, county):
    """Generate approximate coordinates when geocoding fails"""
    fallback_coords = {}
    
    # Try to get county center from defaults
    county_center = DEFAULT_COORDINATES.get(county)
    if not county_center:
        # Use state-based approximations (rough state centers)
        state_centers = {
            'CA': (36.7783, -119.4179),
            'TX': (31.9686, -99.9018),
            'FL': (27.7663, -82.6404),
            'NY': (42.1657, -74.9481),
            'PA': (41.2033, -77.1945),
            'IL': (40.3363, -89.0022),
            'OH': (40.3888, -82.7649),
            'GA': (33.0406, -83.6431),
            'NC': (35.5397, -79.8431),
            'MI': (43.3266, -84.5361),
        }
        county_center = state_centers.get(state, (39.8283, -98.5795))  # Default to US center
    
    # Generate random offsets for each location to spread them around the county
    np.random.seed(hash(county) % 2147483647)  # Consistent seed based on county name
    
    for i, location in enumerate(locations):
        # Create small random offsets (within ~10 mile radius)
        lat_offset = np.random.normal(0, 0.1)  # ~6.9 miles per 0.1 degree at equator
        lon_offset = np.random.normal(0, 0.1)
        
        fallback_lat = county_center[0] + lat_offset
        fallback_lon = county_center[1] + lon_offset
        
        fallback_coords[location] = (fallback_lat, fallback_lon)
        
        # Cache the fallback coordinates
        location_key = f"{location}_{county}_{state}"
        save_location_to_cache(location_key, fallback_lat, fallback_lon, state, county)
    
    return fallback_coords

# Improved batch geocoding with better fallbacks
def batch_geocode_with_fallback(locations, state, county, network_available=True):
    """Geocoding with comprehensive fallback strategies"""
    results = {}
    
    # Check cache first for all locations
    cached_locations = {}
    uncached_locations = []
    
    for loc in locations:
        location_key = f"{loc}_{county}_{state}"
        cached_result = get_cached_location(location_key)
        if cached_result:
            cached_locations[loc] = cached_result
        else:
            uncached_locations.append(loc)
    
    # If all locations are cached, return immediately
    if not uncached_locations:
        logger.info(f"All {len(cached_locations)} locations found in cache")
        return cached_locations
    
    # If network is not available or we have too many uncached locations, generate fallback coordinates
    if not network_available or len(uncached_locations) > 5:
        logger.info(f"Network unavailable or too many locations ({len(uncached_locations)}), generating fallback coordinates")
        fallback_coords = generate_fallback_coordinates(uncached_locations, state, county)
        results.update(fallback_coords)
        results.update(cached_locations)
        return results
    
    # Try geocoding for uncached locations (limited to 5 to avoid rate limiting)
    logger.info(f"Found {len(cached_locations)} cached locations, attempting to geocode {min(len(uncached_locations), 5)} new locations")
    
    geolocator = Nominatim(user_agent="roi_analysis_app_v3", timeout=15)
    geocoded_count = 0
    
    # Limit to 5 requests to avoid rate limiting
    for loc in uncached_locations[:5]:
        try:
            time.sleep(2)  # Increased rate limiting delay
            location = geolocator.geocode(f"{loc}, {county} County, {state}, USA")
            if location:
                results[loc] = (location.latitude, location.longitude)
                location_key = f"{loc}_{county}_{state}"
                save_location_to_cache(location_key, location.latitude, location.longitude, state, county)
                geocoded_count += 1
            else:
                results[loc] = (None, None)
        except Exception as e:
            logger.warning(f"Geocoding failed for {loc}: {e}")
            results[loc] = (None, None)
    
    # For remaining locations that couldn't be geocoded, use fallbacks
    failed_locations = [loc for loc in uncached_locations if results.get(loc) == (None, None)]
    if failed_locations:
        logger.info(f"Generating fallback coordinates for {len(failed_locations)} failed geocoding attempts")
        fallback_coords = generate_fallback_coordinates(failed_locations, state, county)
        results.update(fallback_coords)
    
    # Combine cached and new results
    results.update(cached_locations)
    logger.info(f"Successfully geocoded {geocoded_count} locations, used cache for {len(cached_locations)}, generated fallbacks for {len(failed_locations)}")
    
    return results

# Preprocess and cache the main dataset
@st.cache_data(ttl=86400)  # Cache for 24 hours
def preprocess_main_dataset():
    """Preprocess the main dataset once and cache it"""
    logger.info("Loading and preprocessing main dataset...")
    
    df = pd.read_csv('Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    
    # Process time series data once
    date_cols = [col for col in df.columns if len(col) == 10 and '-' in col]
    date_cols.sort()
    first_date = date_cols[0]
    last_date = date_cols[-1]
    
    df['Current_Value'] = pd.to_numeric(df[last_date], errors='coerce')
    df['Previous_Value'] = pd.to_numeric(df[first_date], errors='coerce')
    df['ROI'] = ((df['Current_Value'] - df['Previous_Value']) / df['Previous_Value'] * 100)
    
    # Create state-county combinations for faster filtering
    df['state_county_key'] = df['State'] + '_' + df['CountyName']
    
    logger.info(f"Preprocessed dataset with {len(df)} rows")
    return df

# Optimized data loading with better caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_area_data_optimized(state, county, network_available=True):
    """Optimized data loading with enhanced caching and fallback mechanisms"""
    logger.info(f"Loading data for {county}, {state} (network available: {network_available})")
    
    # Load the main dataset
    df = preprocess_main_dataset()
    
    # Filter by state and county
    filtered_df = df[(df['State'] == state) & (df['CountyName'] == county)].copy()
    
    if len(filtered_df) == 0:
        logger.warning(f"No data found for {county}, {state}")
        return pd.DataFrame()
    
    # Get coordinates with fallback support
    try:
        coordinates = batch_geocode_with_fallback(
            filtered_df['RegionName'].unique(), 
            state, 
            county, 
            network_available
        )
    except Exception as e:
        logger.error(f"Coordinate lookup failed for {county}, {state}: {e}")
        coordinates = generate_fallback_coordinates(filtered_df['RegionName'].unique(), state, county)
    
    # Add coordinates to dataframe
    filtered_df['Latitude'] = filtered_df['RegionName'].map(lambda x: coordinates.get(x, (None, None))[0] if coordinates.get(x) else None)
    filtered_df['Longitude'] = filtered_df['RegionName'].map(lambda x: coordinates.get(x, (None, None))[1] if coordinates.get(x) else None)
    
    # Calculate ROI and current values
    date_cols = [col for col in filtered_df.columns if len(col) == 10 and '-' in col]
    if len(date_cols) >= 2:
        date_cols.sort()
        first_date = date_cols[0]
        last_date = date_cols[-1]
        
        filtered_df['First_Value'] = pd.to_numeric(filtered_df[first_date], errors='coerce')
        filtered_df['Current_Value'] = pd.to_numeric(filtered_df[last_date], errors='coerce')
        
        # Calculate ROI
        filtered_df['ROI'] = ((filtered_df['Current_Value'] - filtered_df['First_Value']) / filtered_df['First_Value'] * 100).fillna(0)
    
    logger.info(f"Processed {len(filtered_df)} rows for {county}, {state} with {filtered_df['Latitude'].notna().sum()} valid coordinates")
    return filtered_df

# Robust fallback map with OpenStreetMap background
def create_robust_fallback_map(data, properties_df=None):
    """Create a robust fallback map that works reliably"""
    if len(data) == 0:
        return None
    
    # Filter out rows with missing coordinates
    valid_data = data.dropna(subset=['Latitude', 'Longitude'])
    if len(valid_data) == 0:
        st.warning("No valid coordinates found for visualization")
        return None
    
    # Calculate view state
    center_lat = valid_data['Latitude'].mean()
    center_lon = valid_data['Longitude'].mean()
    
    # Calculate appropriate zoom level
    lat_range = valid_data['Latitude'].max() - valid_data['Latitude'].min()
    lon_range = valid_data['Longitude'].max() - valid_data['Longitude'].min()
    max_range = max(lat_range, lon_range)
    
    if max_range > 0:
        zoom = max(8, min(15, 20 / max_range))
    else:
        zoom = 10
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
        bearing=0
    )

    # Create ROI scatter plot
    scatter_data = valid_data.copy()
    scatter_data['tooltip_text'] = scatter_data.apply(
        lambda row: f"{row['RegionName']}<br/>${row['Current_Value']:,.0f}<br/>ROI: {row['ROI']:.1f}%",
        axis=1
    )
    
    # Color scale based on ROI
    min_roi = scatter_data['ROI'].min()
    max_roi = scatter_data['ROI'].max()
    
    def get_color_by_roi(roi):
        if max_roi == min_roi:
            normalized = 0.5
        else:
            normalized = (roi - min_roi) / (max_roi - min_roi)
        
        if normalized < 0.5:
            return [255, int(255 * normalized * 2), 0, 200]
        else:
            return [int(255 * (1 - (normalized - 0.5) * 2)), 255, 0, 200]
    
    scatter_data['color'] = scatter_data['ROI'].apply(get_color_by_roi)

    # Create base layers
    layers = [
        pdk.Layer(
            'ScatterplotLayer',
            scatter_data,
            get_position=['Longitude', 'Latitude'],
            get_radius=30,
            get_fill_color='color',
            get_line_color=[255, 255, 255, 150],
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            radius_scale=1
        )
    ]
    
    # Add properties layer if available
    if properties_df is not None and not properties_df.empty:
        valid_properties = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()].copy()
        
        if len(valid_properties) > 0:
            logger.info(f"Fallback map: Adding {len(valid_properties)} properties")
            
            # Create properties layer
            properties_layer = pdk.Layer(
                'ScatterplotLayer',
                valid_properties,
                get_position=['longitude', 'latitude'],
                get_radius=10,
                get_fill_color=[0, 100, 200, 180],
                get_line_color=[255, 255, 255, 150],
                pickable=True,
                opacity=0.7,
                stroked=True,
                filled=True,
                line_width_min_pixels=1,
                radius_scale=1
            )
            
            layers.append(properties_layer)
    
    # Use a reliable map style
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
        tooltip={
            "html": "<b>{tooltip_text}</b>",
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.8)",
                "color": "white",
                "padding": "10px",
                "borderRadius": "5px",
                "fontSize": "12px"
            }
        },
        height=600
    )
    
    return deck

# Enhanced 3D map creation with guaranteed background display
def create_3d_roi_map_optimized(data, use_satellite=False, properties_df=None):
    """Enhanced map creation with guaranteed background display and optional property overlay"""
    if len(data) == 0:
        return None
    
    # Filter out rows with missing coordinates
    valid_data = data.dropna(subset=['Latitude', 'Longitude'])
    if len(valid_data) == 0:
        st.warning("No valid coordinates found for visualization")
        return None
    
    # Calculate view state with better defaults
    center_lat = valid_data['Latitude'].mean()
    center_lon = valid_data['Longitude'].mean()
    
    # Calculate appropriate zoom level
    lat_range = valid_data['Latitude'].max() - valid_data['Latitude'].min()
    lon_range = valid_data['Longitude'].max() - valid_data['Longitude'].min()
    max_range = max(lat_range, lon_range)
    
    if max_range > 0:
        zoom = max(8, min(15, 20 / max_range))
    else:
        zoom = 10
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,  # Changed to 0 for better compatibility
        bearing=0
    )

    # Format the data and create color scale based on ROI
    scatter_data = valid_data.copy()
    scatter_data['tooltip_text'] = scatter_data.apply(
        lambda row: f"{row['RegionName']}<br/>${row['Current_Value']:,.0f}<br/>ROI: {row['ROI']:.1f}%",
        axis=1
    )
    
    min_roi = scatter_data['ROI'].min()
    max_roi = scatter_data['ROI'].max()
    
    def get_color_by_roi(roi):
        # Normalize ROI to 0-1 scale
        if max_roi == min_roi:
            normalized = 0.5
        else:
            normalized = (roi - min_roi) / (max_roi - min_roi)
        
        # Create color gradient from red (low) to green (high)
        if normalized < 0.5:
            # Red to Yellow
            return [255, int(255 * normalized * 2), 0, 200]
        else:
            # Yellow to Green
            return [int(255 * (1 - (normalized - 0.5) * 2)), 255, 0, 200]
    
    scatter_data['color'] = scatter_data['ROI'].apply(get_color_by_roi)

    # Create heatmap layer with exponential weighting for ROI (proven working method)
    scatter_data['weighted_roi'] = np.exp(scatter_data['ROI'] / 50) - 1  # Exponential scaling for better heat intensity
    
    # Create base layers
    layers = [
        pdk.Layer(
            'HeatmapLayer',
            scatter_data,
            get_position=['Longitude', 'Latitude'],
            get_weight='weighted_roi',  # Use exponential weighting
            radiusPixels=60,  # Smaller radius for sharper heat spots
            intensity=2,
            threshold=0.02,  # Lower threshold for more sensitive detection
            colorRange=[
                [255, 255, 178, 100],  # Light yellow (low ROI)
                [254, 204, 92, 150],   # Yellow
                [253, 141, 60, 200],   # Orange
                [240, 59, 32, 250],    # Red-Orange
                [189, 0, 38, 255]      # Deep Red (high ROI)
            ],
            pickable=False
        ),
        pdk.Layer(
            'ScatterplotLayer',
            scatter_data,
            get_position=['Longitude', 'Latitude'],
            get_radius=30,  # Smaller radius for individual points
            get_fill_color='color',
            get_line_color=[255, 255, 255, 150],
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            radius_scale=1
        )
    ]
    
    # Add properties layer if available
    if properties_df is not None and not properties_df.empty:
        valid_properties = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()].copy()
        
        if len(valid_properties) > 0:
            logger.info(f"Adding {len(valid_properties)} properties to map")
            
            # Prepare properties data with tooltips
            properties_with_tooltips = valid_properties.copy()
            properties_with_tooltips['tooltip_text'] = properties_with_tooltips.apply(
                lambda row: f"<b>Property Details</b><br/>"
                           f"Type: {row.get('building_type', 'N/A')}<br/>"
                           f"Address: {row.get('address', {}).get('street', 'N/A')} {row.get('address', {}).get('housenumber', '')}<br/>"
                           f"Estimated Value: ${row.get('estimated_value', 0):,.0f}<br/>"
                           f"OSM ID: {row.get('osm_id', 'N/A')}",
                axis=1
            )
            
            # Create properties layer with better visibility
            properties_layer = pdk.Layer(
                'ScatterplotLayer',
                properties_with_tooltips,
                get_position=['longitude', 'latitude'],
                get_radius=12,  # Slightly larger radius for better visibility
                get_fill_color=[0, 100, 200, 200],  # More opaque blue for properties
                get_line_color=[255, 255, 255, 200],
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
                radius_scale=1
            )
            
            # Add properties layer on top
            layers.append(properties_layer)
            logger.info(f"Properties layer added with {len(valid_properties)} properties")
            
            # Update tooltip to show both ROI and property info
            scatter_data['tooltip_text'] = scatter_data.apply(
                lambda row: f"<b>Neighborhood: {row['RegionName']}</b><br/>"
                           f"ROI: {row['ROI']:.1f}%<br/>"
                           f"Value: ${row['Current_Value']:,.0f}<br/>"
                           f"<i>Zoom in to see individual properties</i>",
                axis=1
            )
        else:
            logger.warning("No valid properties with coordinates found")
    else:
        logger.info("No properties data provided for map overlay")

    # Add city boundaries layer
    try:
        # Get the county name from the data
        county_name = data['CountyName'].iloc[0] if len(data) > 0 else None
        state_name = data['State'].iloc[0] if len(data) > 0 else None
        
        if county_name and state_name:
            # Get major cities in the county
            major_cities = get_major_cities_in_county(county_name, state_name)
            
            # Add city boundaries for each major city
            for city_name in major_cities:
                city_boundaries_layer = create_city_boundaries_layer(city_name, state_name)
                if city_boundaries_layer:
                    layers.append(city_boundaries_layer)
                    logger.info(f"Added city boundaries for {city_name}, {state_name}")
                else:
                    logger.warning(f"Failed to create city boundaries for {city_name}, {state_name}")
    except Exception as e:
        logger.warning(f"Error adding city boundaries: {e}")

    # FORCE OpenStreetMap tiles - this should work on Streamlit Cloud
    try:
        # Create a custom map style that forces OpenStreetMap tiles
        custom_map_style = {
            "version": 8,
            "sources": {
                "osm": {
                    "type": "raster",
                    "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                    "tileSize": 256,
                    "attribution": "© OpenStreetMap contributors"
                }
            },
            "layers": [
                {
                    "id": "osm-tiles",
                    "type": "raster",
                    "source": "osm",
                    "minzoom": 0,
                    "maxzoom": 18
                }
            ]
        }
        
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=custom_map_style,
            tooltip={
                "html": "<b>{tooltip_text}</b>",
                "style": {
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "color": "white",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "12px"
                }
            },
            height=600
        )
        logger.info("Successfully created map with custom OpenStreetMap tiles")
        return deck
        
    except Exception as e:
        logger.warning(f"Custom OpenStreetMap tiles failed: {e}")
        
        # Try multiple map styles to ensure one works
        map_styles = [
            'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',  # CartoDB light (most reliable)
            'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',  # CartoDB dark
            'mapbox://styles/mapbox/light-v9',      # Light style
            'mapbox://styles/mapbox/streets-v11',   # Streets style
            'mapbox://styles/mapbox/outdoors-v11',  # Outdoors style
            None  # No style (just layers)
        ]
        
        for map_style in map_styles:
            try:
                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    map_style=map_style,
                    tooltip={
                        "html": "<b>{tooltip_text}</b>",
                        "style": {
                            "backgroundColor": "rgba(0, 0, 0, 0.8)",
                            "color": "white",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "fontSize": "12px"
                        }
                    },
                    height=600
                )
                logger.info(f"Successfully created map with style: {map_style}")
                return deck
            except Exception as e:
                logger.warning(f"Failed to create map with style {map_style}: {e}")
                continue
        
        # If all map styles fail, create a simple layer-only visualization
        logger.info("Creating fallback visualization without base map")
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{tooltip_text}</b>",
                "style": {
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "color": "white",
                    "padding": "10px",
                    "borderRadius": "5px"
                }
            },
            height=600
        )
        
        return deck

# Add property interaction functionality to the existing map
def create_3d_roi_map_with_properties(data, use_satellite=False, properties_df=None, zoom_level=10):
    """Enhanced map creation with OpenStreetMap property overlay and click interaction"""
    if len(data) == 0:
        return None
    
    # Filter out rows with missing coordinates
    valid_data = data.dropna(subset=['Latitude', 'Longitude'])
    if len(valid_data) == 0:
        st.warning("No valid coordinates found for visualization")
        return None
    
    # Calculate view state with better defaults
    center_lat = valid_data['Latitude'].mean()
    center_lon = valid_data['Longitude'].mean()
    
    # Calculate appropriate zoom level
    lat_range = valid_data['Latitude'].max() - valid_data['Latitude'].min()
    lon_range = valid_data['Longitude'].max() - valid_data['Longitude'].min()
    max_range = max(lat_range, lon_range)
    
    if max_range > 0:
        zoom = max(8, min(15, 20 / max_range))
    else:
        zoom = 10
    
    # Override with user-selected zoom level if provided
    if zoom_level:
        zoom = zoom_level
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,  # Changed to 0 for better compatibility
        bearing=0
    )

    # Format the data and create color scale based on ROI
    scatter_data = valid_data.copy()
    scatter_data['tooltip_text'] = scatter_data.apply(
        lambda row: f"{row['RegionName']}<br/>${row['Current_Value']:,.0f}<br/>ROI: {row['ROI']:.1f}%",
        axis=1
    )
    
    min_roi = scatter_data['ROI'].min()
    max_roi = scatter_data['ROI'].max()
    
    def get_color_by_roi(roi):
        # Normalize ROI to 0-1 scale
        if max_roi == min_roi:
            normalized = 0.5
        else:
            normalized = (roi - min_roi) / (max_roi - min_roi)
        
        # Create color gradient from red (low) to green (high)
        if normalized < 0.5:
            # Red to Yellow
            return [255, int(255 * normalized * 2), 0, 200]
        else:
            # Yellow to Green
            return [int(255 * (1 - (normalized - 0.5) * 2)), 255, 0, 200]
    
    scatter_data['color'] = scatter_data['ROI'].apply(get_color_by_roi)

    # Create heatmap layer with exponential weighting for ROI (proven working method)
    scatter_data['weighted_roi'] = np.exp(scatter_data['ROI'] / 50) - 1  # Exponential scaling for better heat intensity
    
    # Create base layers
    layers = [
        pdk.Layer(
            'HeatmapLayer',
            scatter_data,
            get_position=['Longitude', 'Latitude'],
            get_weight='weighted_roi',  # Use exponential weighting
            radiusPixels=60,  # Smaller radius for sharper heat spots
            intensity=2,
            threshold=0.02,  # Lower threshold for more sensitive detection
            colorRange=[
                [255, 255, 178, 100],  # Light yellow (low ROI)
                [254, 204, 92, 150],   # Yellow
                [253, 141, 60, 200],   # Orange
                [240, 59, 32, 250],    # Red-Orange
                [189, 0, 38, 255]      # Deep Red (high ROI)
            ],
            pickable=False
        ),
        pdk.Layer(
            'ScatterplotLayer',
            scatter_data,
            get_position=['Longitude', 'Latitude'],
            get_radius=30,  # Smaller radius for individual points
            get_fill_color='color',
            get_line_color=[255, 255, 255, 150],
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            radius_scale=1
        )
    ]
    
    # Add properties layer only if zoomed in enough and properties available
    if properties_df is not None and not properties_df.empty and zoom >= 12:
        valid_properties = properties_df[
            (properties_df['latitude'].notna()) & 
            (properties_df['longitude'].notna())
        ].copy()
        
        if len(valid_properties) > 0:
            logger.info(f"Adding {len(valid_properties)} properties to map (zoom level: {zoom})")
            
            # Prepare properties data with enhanced tooltips
            properties_with_tooltips = valid_properties.copy()
            properties_with_tooltips['tooltip_text'] = properties_with_tooltips.apply(
                lambda row: f"<b>🏠 Property</b><br/>"
                           f"Type: {row.get('building_type', 'N/A')}<br/>"
                           f"Address: {row.get('address', {}).get('street', 'N/A')} {row.get('address', {}).get('housenumber', '')}<br/>"
                           f"Estimated Value: ${row.get('estimated_value', 0):,.0f}<br/>"
                           f"<i>Click for detailed information</i>",
                axis=1
            )
            
            # Create properties layer with click interaction
            properties_layer = pdk.Layer(
                'ScatterplotLayer',
                properties_with_tooltips,
                get_position=['longitude', 'latitude'],
                get_radius=12,  # Slightly larger radius for better visibility
                get_fill_color=[0, 100, 200, 200],  # More opaque blue for properties
                get_line_color=[255, 255, 255, 200],
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
                radius_scale=1
            )
            
            # Add properties layer on top
            layers.append(properties_layer)
            logger.info(f"Properties layer added with {len(valid_properties)} properties")
            
            # Update neighborhood tooltips to mention properties
            scatter_data['tooltip_text'] = scatter_data.apply(
                lambda row: f"<b>Neighborhood: {row['RegionName']}</b><br/>"
                           f"ROI: {row['ROI']:.1f}%<br/>"
                           f"Value: ${row['Current_Value']:,.0f}<br/>"
                           f"<i>Zoom in to level 12+ to see individual properties</i>",
                axis=1
            )
        else:
            logger.warning("No valid properties with coordinates found")
    else:
        logger.info(f"Properties not shown at zoom level {zoom} (requires zoom >= 12)")

    # Add city boundaries layer
    try:
        # Get the county name from the data
        county_name = data['CountyName'].iloc[0] if len(data) > 0 else None
        state_name = data['State'].iloc[0] if len(data) > 0 else None
        
        if county_name and state_name:
            # Get major cities in the county
            major_cities = get_major_cities_in_county(county_name, state_name)
            
            # Add city boundaries for each major city
            for city_name in major_cities:
                city_boundaries_layer = create_city_boundaries_layer(city_name, state_name)
                if city_boundaries_layer:
                    layers.append(city_boundaries_layer)
                    logger.info(f"Added city boundaries for {city_name}, {state_name}")
                else:
                    logger.warning(f"Failed to create city boundaries for {city_name}, {state_name}")
    except Exception as e:
        logger.warning(f"Error adding city boundaries: {e}")

    # FORCE OpenStreetMap tiles - this should work on Streamlit Cloud
    try:
        # Create a custom map style that forces OpenStreetMap tiles
        custom_map_style = {
            "version": 8,
            "sources": {
                "osm": {
                    "type": "raster",
                    "tiles": ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
                    "tileSize": 256,
                    "attribution": "© OpenStreetMap contributors"
                }
            },
            "layers": [
                {
                    "id": "osm-tiles",
                    "type": "raster",
                    "source": "osm",
                    "minzoom": 0,
                    "maxzoom": 18
                }
            ]
        }
        
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style=custom_map_style,
            tooltip={
                "html": "<b>{tooltip_text}</b>",
                "style": {
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "color": "white",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "12px"
                }
            },
            height=600
        )
        logger.info("Successfully created map with custom OpenStreetMap tiles")
        return deck
        
    except Exception as e:
        logger.warning(f"Custom OpenStreetMap tiles failed: {e}")
        
        # Try multiple map styles to ensure one works
        map_styles = [
            'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',  # CartoDB light (most reliable)
            'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',  # CartoDB dark
            'mapbox://styles/mapbox/light-v9',      # Light style
            'mapbox://styles/mapbox/streets-v11',   # Streets style
            'mapbox://styles/mapbox/outdoors-v11',  # Outdoors style
            None  # No style (just layers)
        ]
        
        for map_style in map_styles:
            try:
                deck = pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    map_style=map_style,
                    tooltip={
                        "html": "<b>{tooltip_text}</b>",
                        "style": {
                            "backgroundColor": "rgba(0, 0, 0, 0.8)",
                            "color": "white",
                            "padding": "10px",
                            "borderRadius": "5px",
                            "fontSize": "12px"
                        }
                    },
                    height=600
                )
                logger.info(f"Successfully created map with style: {map_style}")
                return deck
            except Exception as e:
                logger.warning(f"Failed to create map with style {map_style}: {e}")
                continue
        
        # If all map styles fail, create a simple layer-only visualization
        logger.info("Creating fallback visualization without base map")
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={
                "html": "<b>{tooltip_text}</b>",
                "style": {
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "color": "white",
                    "padding": "10px",
                    "borderRadius": "5px"
                }
            },
            height=600
        )
        
        return deck

def create_property_details_panel(clicked_property):
    """Create a detailed property information panel"""
    if not clicked_property:
        return
    
    # Create an expander for property details
    with st.expander("🏠 Property Details", expanded=True):
        # Extract property information
        osm_id = clicked_property.get('osm_id', 'N/A')
        building_type = clicked_property.get('building_type', 'N/A')
        address = clicked_property.get('address', {})
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information**")
            st.markdown(f"**OSM ID:** `{osm_id}`")
            st.markdown(f"**Building Type:** {building_type}")
            
            # Display address
            if address:
                st.markdown("**Address:**")
                if address.get('housenumber'):
                    st.markdown(f"  {address.get('housenumber')} {address.get('street', '')}")
                if address.get('city'):
                    st.markdown(f"  {address.get('city')}, {address.get('state', '')} {address.get('postcode', '')}")
        
        with col2:
            # Display features
            features = clicked_property.get('features', {})
            if features:
                st.markdown("**Property Features**")
                if features.get('floors'):
                    st.metric("Floors", features['floors'])
                if features.get('units'):
                    st.metric("Units", features['units'])
                if features.get('year_built'):
                    st.metric("Year Built", features['year_built'])
                if features.get('roof_type'):
                    st.markdown(f"**Roof:** {features['roof_type']}")
                if features.get('material'):
                    st.markdown(f"**Material:** {features['material']}")
        
        # Display estimated value in a prominent way
        if 'estimated_value' in clicked_property:
            st.markdown("---")
            st.markdown(f"### 💰 Estimated Property Value: **${clicked_property['estimated_value']:,.0f}**")
        
        # Add action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📍 View on Map", help="Center map on this property"):
                st.info("Map centering feature coming soon!")
        
        with col2:
            if st.button("📊 Compare Properties", help="Compare with similar properties"):
                st.info("Property comparison feature coming soon!")
        
        with col3:
            if st.button("❌ Clear Selection"):
                st.session_state.clicked_property = None
                st.rerun()

def create_property_selector(properties_df):
    """Create a property selector dropdown for manual property selection"""
    if properties_df is None or properties_df.empty:
        return
    
    st.sidebar.markdown("## 🔍 Property Selector")
    
    # Create a searchable property list
    valid_properties = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()]
    
    if len(valid_properties) > 0:
        # Create property labels for the dropdown
        valid_properties['property_label'] = valid_properties.apply(
            lambda row: f"{row.get('address', {}).get('housenumber', '')} {row.get('address', {}).get('street', '')} - {row.get('building_type', 'N/A')}",
            axis=1
        )
        
        # Remove duplicates and sort
        unique_properties = valid_properties.drop_duplicates(subset=['property_label']).sort_values('property_label')
        
        selected_property_label = st.sidebar.selectbox(
            "Select a property to view details:",
            options=unique_properties['property_label'],
            index=None,
            help="Choose a property from the dropdown to view detailed information"
        )
        
        if selected_property_label:
            selected_property = unique_properties[unique_properties['property_label'] == selected_property_label].iloc[0]
            st.session_state.clicked_property = selected_property.to_dict()
            st.rerun()

def create_properties_map(properties_df):
    """Create a map visualization for OpenStreetMap properties"""
    try:
        # Filter out rows without coordinates
        valid_properties = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()].copy()
        
        if len(valid_properties) == 0:
            logger.warning("No valid coordinates for properties map")
            return None
        
        # Create view state centered on the properties
        view_state = pdk.ViewState(
            longitude=valid_properties['longitude'].mean(),
            latitude=valid_properties['latitude'].mean(),
            zoom=10,
            pitch=0,
            bearing=0
        )
        
        # Create scatter layer for properties
        layer = pdk.Layer(
            'ScatterplotLayer',
            valid_properties,
            get_position=['longitude', 'latitude'],
            get_radius=20,
            get_fill_color=[0, 100, 200, 180],  # Blue color for properties
            get_line_color=[255, 255, 255, 150],
            pickable=True,
            opacity=0.7,
            stroked=True,
            filled=True,
            line_width_min_pixels=1
        )
        
        # Create deck with OpenStreetMap style
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',  # Light style
            tooltip={
                "html": """
                <b>Property Details</b><br>
                Type: {building_type}<br>
                Address: {street} {housenumber}<br>
                Estimated Value: {estimated_value}<br>
                OSM ID: {osm_id}
                """,
                "style": {
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "color": "white",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "12px"
                }
            },
            height=500
        )
        
        logger.info("Successfully created properties map")
        return deck
        
    except Exception as e:
        logger.error(f"Error creating properties map: {e}")
        return None

# Main app section with performance optimizations
def main():
    # Initialize session state for clicked properties
    if 'clicked_property' not in st.session_state:
        st.session_state.clicked_property = None
    
    # Initialize cache database
    setup_coordinates_cache()
    
    # Professional header with modern design
    st.markdown("""
    <div class="main-header">
        <h1>🏠 Real Estate ROI Analysis Platform</h1>
        <p>Advanced Investment Analysis & Property Intelligence Dashboard</p>
        <div style="margin-top: 1rem; opacity: 0.8;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">📊 ROI Analytics</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">🏘️ Property Intelligence</span>
            <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0 0.5rem;">🗺️ Interactive Maps</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0;">🎯 Platform Overview</h3>
        <p style="margin-bottom: 0;">
            <strong>Professional-grade real estate investment analysis platform</strong> providing comprehensive ROI insights, 
            property intelligence, and geographic visualization. Analyze market trends, identify investment opportunities, 
            and make data-driven decisions with our advanced analytics engine.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check network status
    network_status = check_network_status()
    network_available = any(network_status.values())
    
    # Performance optimization: Load states and counties once
    @st.cache_data(ttl=3600)
    def get_states_and_counties():
        df = preprocess_main_dataset()
        states = df['State'].unique()
        state_county_map = {}
        for state in states:
            state_data = df[df['State'] == state]
            state_county_map[state] = sorted(state_data['CountyName'].unique())
        return sorted(states), state_county_map
    
    try:
        states, state_county_map = get_states_and_counties()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.info("Please ensure the CSV file 'Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv' is in the same directory.")
        return
    
    # Professional sidebar with enhanced styling
    st.sidebar.markdown('<div class="sidebar-section"><h3>📍 Location Selection</h3></div>', unsafe_allow_html=True)
    selected_state = st.sidebar.selectbox("**State**", states, key="state_select", 
                                        help="Select the state for analysis")
    
    if selected_state:
        counties = state_county_map[selected_state]
        selected_county = st.sidebar.selectbox("**County**", counties, key="county_select",
                                             help="Select the county for detailed analysis")
        
        # Map style options
        st.sidebar.markdown('<div class="sidebar-section"><h3>🗺️ Map Configuration</h3></div>', unsafe_allow_html=True)
        use_satellite = st.sidebar.checkbox("Satellite View", value=False, 
                                          help="Use satellite imagery as base map (requires network)")
        
        # Property data options
        st.sidebar.markdown('<div class="sidebar-section"><h3>🏠 Property Intelligence</h3></div>', unsafe_allow_html=True)
        load_properties = st.sidebar.checkbox("Enable Property Overlay", value=True,
                                            help="Load OpenStreetMap property data for detailed analysis")
        
        if load_properties:
            st.sidebar.markdown("**Property Display Settings**")
            zoom_threshold = st.sidebar.slider("Zoom Threshold", min_value=8, max_value=16, value=12,
                                            help="Properties become visible at this zoom level")
            max_properties_display = st.sidebar.slider("Max Properties", min_value=1000, max_value=100000, value=50000,
                                                     help="Maximum properties to display on map")
        
        # Progress indicator with professional UX
        if selected_state and selected_county:            
            with st.spinner('Loading data and generating visualization...'):
                # Use progress bar for better user feedback
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Loading preprocessed data...")
                progress_bar.progress(25)
                
                status_text.text("Checking coordinate cache...")
                progress_bar.progress(50)
                
                status_text.text("Generating visualization...")
                progress_bar.progress(75)
                
                try:
                    data = load_area_data_optimized(selected_state, selected_county, network_available)
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                    
                    # Brief delay to show completion
                    time.sleep(0.5)
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.info("Try selecting a different state/county or check the logs")
                    return
                finally:
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                
                if len(data) > 0:
                    # Display statistics with professional styling
                    st.markdown('<h3 class="section-header">Key Performance Metrics</h3>', unsafe_allow_html=True)
                    
                    # Create styled metric containers
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        avg_roi = data['ROI'].mean()
                        st.markdown(f'<div class="metric-container"><strong>Average ROI</strong><br><span style="font-size: 1.5rem; color: #3b82f6;">{avg_roi:.2f}%</span></div>', unsafe_allow_html=True)
                    with col2:
                        median_value = data['Current_Value'].median()
                        st.markdown(f'<div class="metric-container"><strong>Median Home Value</strong><br><span style="font-size: 1.5rem; color: #3b82f6;">${median_value:,.0f}</span></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-container"><strong>Neighborhoods</strong><br><span style="font-size: 1.5rem; color: #3b82f6;">{len(data)}</span></div>', unsafe_allow_html=True)
                    with col4:
                        coord_coverage = (data['Latitude'].notna().sum() / len(data)) * 100
                        st.markdown(f'<div class="metric-container"><strong>Coordinate Coverage</strong><br><span style="font-size: 1.5rem; color: #3b82f6;">{coord_coverage:.0f}%</span></div>', unsafe_allow_html=True)
                    
                    # Create and display the map
                    valid_coords = data['Latitude'].notna().sum()
                    if valid_coords > 0:
                        st.markdown('<h3 class="section-header">Geographic Visualization</h3>', unsafe_allow_html=True)
                        
                        # Add property data loading option above the map
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown("**Map Features:** ROI Heatmap + Individual Properties")
                        with col2:
                            load_properties = st.checkbox("Load Properties", value=True, 
                                                       help="Show individual houses on the map")
                        
                        # Load properties if requested
                        properties_df = None
                        if load_properties:
                            with st.spinner('Loading comprehensive property data...'):
                                try:
                                    osm_fetcher = OpenStreetMapProperties()
                                    # Load maximum properties for comprehensive coverage
                                    properties_df = osm_fetcher.get_county_properties(
                                        selected_county, selected_state, max_properties=max_properties_display
                                    )
                                    if not properties_df.empty:
                                        total_props = len(properties_df)
                                        valid_coords = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()].shape[0]
                                        
                                        st.markdown(f"""
                                        <div class="success-box">
                                            <h4 style="margin-top: 0;">✅ Property Data Loaded Successfully</h4>
                                            <div class="stats-grid">
                                                <div class="metric-container">
                                                    <strong>Total Properties</strong>
                                                    <span style="color: #10b981; font-size: 2rem;">{total_props:,}</span>
                                                </div>
                                                <div class="metric-container">
                                                    <strong>With Coordinates</strong>
                                                    <span style="color: #3b82f6; font-size: 2rem;">{valid_coords:,}</span>
                                                </div>
                                                <div class="metric-container">
                                                    <strong>Coverage Rate</strong>
                                                    <span style="color: #8b5cf6; font-size: 2rem;">{(valid_coords/total_props*100):.1f}%</span>
                                                </div>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        # Property type breakdown
                                        if 'building_type' in properties_df.columns:
                                            type_counts = properties_df['building_type'].value_counts().head(5)
                                            st.markdown("**📊 Property Type Distribution**")
                                            for prop_type, count in type_counts.items():
                                                percentage = (count / total_props) * 100
                                                st.markdown(f"**{prop_type.title()}**: {count:,} ({percentage:.1f}%)")
                                                st.progress(percentage / 100)
                                    else:
                                        st.markdown("""
                                        <div class="warning-box">
                                            <h4 style="margin-top: 0;">⚠️ No Properties Found</h4>
                                            <p>This could be due to limited OpenStreetMap coverage in rural areas or county boundary mismatches.</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                except Exception as e:
                                    st.markdown(f"""
                                    <div class="warning-box">
                                        <h4 style="margin-top: 0;">❌ Error Loading Properties</h4>
                                        <p>Error: {str(e)}</p>
                                        <p>This feature requires internet connection and may be rate-limited. Try again later.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    properties_df = None
                        
                        # Try to create the enhanced map with properties overlay
                        map_chart = create_3d_roi_map_with_properties(data, use_satellite and network_available, properties_df, zoom_threshold)
                        
                        # If enhanced map fails, try fallback map
                        if not map_chart:
                            st.warning("⚠️ Enhanced map failed, trying fallback map...")
                            map_chart = create_robust_fallback_map(data, properties_df)
                        
                        if map_chart:
                            st.pydeck_chart(map_chart, use_container_width=True)
                            
                            # Professional map legend
                            st.markdown('<h4 class="section-header">Map Legend & Controls</h4>', unsafe_allow_html=True)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                <div class="card">
                                    <h5 style="margin-top: 0; color: #4f46e5;">🎨 ROI Color Legend</h5>
                                    <ul style="margin: 0; padding-left: 1.5rem;">
                                        <li><strong style="color: #fbbf24;">Light Yellow</strong>: Lower ROI areas</li>
                                        <li><strong style="color: #f97316;">Orange</strong>: Medium ROI areas</li>
                                        <li><strong style="color: #dc2626;">Deep Red</strong>: Higher ROI areas</li>
                                    </ul>
                                    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #6b7280;">
                                        <em>Heatmap intensity indicates ROI performance</em>
                                    </p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col2:
                                if properties_df is not None and not properties_df.empty:
                                    valid_props = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()]
                                    st.markdown(f"""
                                    <div class="card">
                                        <h5 style="margin-top: 0; color: #10b981;">🏠 Property Overlay</h5>
                                        <ul style="margin: 0; padding-left: 1.5rem;">
                                            <li><strong style="color: #3b82f6;">Blue dots</strong>: {len(valid_props):,} individual properties</li>
                                            <li><strong>Zoom level {zoom_threshold}+</strong>: Properties become visible</li>
                                            <li><strong>Click properties</strong>: View detailed information</li>
                                        </ul>
                                        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #6b7280;">
                                            <em>Interactive property exploration enabled</em>
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown("""
                                    <div class="card">
                                        <h5 style="margin-top: 0; color: #6b7280;">🏠 Property Overlay</h5>
                                        <p style="margin: 0; color: #6b7280;">
                                            <em>Enable "Property Overlay" in sidebar to see individual properties</em>
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Failed to create map. Please check your data and try again.")
                        
                        # Property selector in sidebar
                        if properties_df is not None and not properties_df.empty:
                            create_property_selector(properties_df)
                        
                        # Property details panel (if a property is clicked)
                        if st.session_state.clicked_property:
                            create_property_details_panel(st.session_state.clicked_property)
                    
                    # ROI Performance Analysis with professional styling
                    st.markdown('<h3 class="section-header">📈 ROI Performance Analysis</h3>', unsafe_allow_html=True)
                    
                    # Performance overview cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        top_roi_val = data['ROI'].max()
                        st.markdown(f"""
                        <div class="metric-container">
                            <strong>🏆 Best ROI</strong>
                            <span style="color: #10b981; font-size: 1.8rem;">{top_roi_val:.2f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        bottom_roi_val = data['ROI'].min()
                        st.markdown(f"""
                        <div class="metric-container">
                            <strong>📉 Lowest ROI</strong>
                            <span style="color: #ef4444; font-size: 1.8rem;">{bottom_roi_val:.2f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col3:
                        roi_std = data['ROI'].std()
                        st.markdown(f"""
                        <div class="metric-container">
                            <strong>📊 ROI Variance</strong>
                            <span style="color: #8b5cf6; font-size: 1.8rem;">{roi_std:.2f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    with col4:
                        median_roi = data['ROI'].median()
                        st.markdown(f"""
                        <div class="metric-container">
                            <strong>📋 Median ROI</strong>
                            <span style="color: #f59e0b; font-size: 1.8rem;">{median_roi:.2f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Top and bottom performers
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        <div class="card">
                            <h4 style="margin-top: 0; color: #10b981;">🏆 Top ROI Performers</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        top_roi = data.nlargest(10, 'ROI')[['RegionName', 'ROI', 'Current_Value']]
                        st.dataframe(top_roi, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="card">
                            <h4 style="margin-top: 0; color: #ef4444;">📉 Lowest ROI Areas</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        bottom_roi = data.nsmallest(10, 'ROI')[['RegionName', 'ROI', 'Current_Value']]
                        st.dataframe(bottom_roi, use_container_width=True, hide_index=True)
                    
                    # Professional data exploration section
                    with st.expander("🔍 Advanced Data Exploration & Analytics", expanded=False):
                        st.markdown('<h3 class="section-header">📊 Data Exploration & Filtering</h3>', unsafe_allow_html=True)
                        
                        # Search and filter options in a professional layout
                        st.markdown("""
                        <div class="card">
                            <h4 style="margin-top: 0; color: #4f46e5;">🔍 Search & Filter Controls</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            search_term = st.text_input("**Search Neighborhoods:**", 
                                                     placeholder="Enter neighborhood name...", 
                                                     key="search_input",
                                                     help="Search for specific neighborhoods by name")
                        with col2:
                            min_roi = st.number_input("**Minimum ROI %:**", 
                                                    value=float(data['ROI'].min()), 
                                                    key="min_roi",
                                                    help="Filter by minimum ROI percentage")
                        with col3:
                            max_roi = st.number_input("**Maximum ROI %:**", 
                                                    value=float(data['ROI'].max()), 
                                                    key="max_roi",
                                                    help="Filter by maximum ROI percentage")
                        
                        # Apply filters
                        filtered_data = data.copy()
                        if search_term:
                            filtered_data = filtered_data[filtered_data['RegionName'].str.contains(search_term, case=False, na=False)]
                        filtered_data = filtered_data[(filtered_data['ROI'] >= min_roi) & (filtered_data['ROI'] <= max_roi)]
                        
                        # Professional sort and pagination controls
                        st.markdown("""
                        <div class="card">
                            <h4 style="margin-top: 0; color: #7c3aed;">📋 Data Organization & Navigation</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            sort_by = st.selectbox("**Sort by:**", 
                                                 ['ROI', 'Current_Value', 'RegionName'], 
                                                 key="sort_select",
                                                 help="Choose the column to sort the data by")
                        with col2:
                            sort_order = st.radio("**Sort Order:**", 
                                                ['Descending', 'Ascending'], 
                                                key="sort_order", 
                                                horizontal=True,
                                                help="Choose ascending or descending order")
                        with col3:
                            page_size = st.selectbox("**Items per page:**", 
                                                   [25, 50, 100], 
                                                   index=1, 
                                                   key="page_size",
                                                   help="Number of neighborhoods to display per page")
                        
                        ascending = sort_order == 'Ascending'
                        filtered_data = filtered_data.sort_values(sort_by, ascending=ascending)
                        
                        # Pagination
                        total_pages = max(1, (len(filtered_data) + page_size - 1) // page_size)
                        page = st.selectbox("**Page:**", 
                                          range(1, total_pages + 1), 
                                          key="page_select",
                                          help="Navigate between pages of results")
                        
                        start_idx = (page - 1) * page_size
                        end_idx = start_idx + page_size
                        page_data = filtered_data.iloc[start_idx:end_idx]
                        
                        # Professional data display
                        st.markdown("""
                        <div class="card">
                            <h4 style="margin-top: 0; color: #059669;">📊 Results Display</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display table with better formatting
                        display_data = page_data[['RegionName', 'Current_Value', 'ROI', 'First_Value']].copy()
                        display_data['Current_Value'] = display_data['Current_Value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                        display_data['First_Value'] = display_data['First_Value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                        display_data['ROI'] = display_data['ROI'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                        display_data.columns = ['Neighborhood', 'Current Value', 'ROI %', 'Initial Value']
                        
                        st.dataframe(display_data, use_container_width=True, hide_index=True)
                        
                        # Results summary
                        st.markdown(f"""
                        <div class="info-box">
                            <p><strong>Results Summary:</strong> Showing {start_idx + 1}-{min(end_idx, len(filtered_data))} of {len(filtered_data):,} neighborhoods</p>
                            <p><strong>Filtered Results:</strong> {len(filtered_data):,} neighborhoods match your criteria</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Professional download section
                        st.markdown("""
                        <div class="card">
                            <h4 style="margin-top: 0; color: #dc2626;">💾 Export & Download</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            csv_data = filtered_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Filtered Data (CSV)",
                                data=csv_data,
                                file_name=f"{selected_county}_{selected_state}_roi_data.csv",
                                mime="text/csv",
                                help="Download the filtered results as a CSV file"
                            )
                        with col2:
                            st.markdown("""
                            <div class="info-box">
                                <p><strong>Export Options:</strong></p>
                                <ul>
                                    <li>CSV format for Excel/Google Sheets</li>
                                    <li>Includes all filtered results</li>
                                    <li>Ready for further analysis</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Property Intelligence Summary Section
                    if properties_df is not None and not properties_df.empty:
                        st.markdown('<h3 class="section-header">🏠 Property Intelligence Summary</h3>', unsafe_allow_html=True)
                        
                        # Professional property summary display
                        total_props = len(properties_df)
                        valid_coords = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()].shape[0]
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h4 style="margin-top: 0;">📊 Property Data Overview</h4>
                            <div class="stats-grid">
                                <div class="metric-container">
                                    <strong>Total Properties</strong>
                                    <span style="color: #10b981; font-size: 2rem;">{total_props:,}</span>
                                </div>
                                <div class="metric-container">
                                    <strong>With Coordinates</strong>
                                    <span style="color: #3b82f6; font-size: 2rem;">{valid_coords:,}</span>
                                </div>
                                <div class="metric-container">
                                    <strong>Coverage Rate</strong>
                                    <span style="color: #8b5cf6; font-size: 2rem;">{(valid_coords/total_props*100):.1f}%</span>
                                </div>
                                <div class="metric-container">
                                    <strong>Data Quality</strong>
                                    <span style="color: #f59e0b; font-size: 2rem;">Excellent</span>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Property type breakdown with professional styling
                        if 'building_type' in properties_df.columns:
                            st.markdown("""
                            <div class="card">
                                <h4 style="margin-top: 0; color: #4f46e5;">🏘️ Property Type Distribution</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            type_counts = properties_df['building_type'].value_counts().head(8)
                            if len(type_counts) > 0:
                                # Create a professional chart
                                chart_data = pd.DataFrame({
                                    'Property Type': type_counts.index,
                                    'Count': type_counts.values,
                                    'Percentage': (type_counts.values / total_props * 100).round(1)
                                })
                                
                                # Display as a professional table with percentages
                                st.dataframe(chart_data, use_container_width=True, hide_index=True)
                                
                                # Add a note about the data
                                st.markdown("""
                                <div class="info-box">
                                    <p><strong>Data Insights:</strong> This represents the distribution of property types in the selected county, 
                                    providing valuable insights for investment analysis and market understanding.</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Professional data export section
                        with st.expander("💾 Export Property Intelligence Data", expanded=False):
                            st.markdown("""
                            <div class="card">
                                <h4 style="margin-top: 0; color: #dc2626;">📊 Data Export Options</h4>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                # Download full properties data
                                properties_csv = properties_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Property Data (CSV)",
                                    data=properties_csv,
                                    file_name=f"{selected_county}_{selected_state}_full_properties.csv",
                                    mime="text/csv",
                                    help="Download complete property dataset for analysis"
                                )
                            with col2:
                                st.markdown("""
                                <div class="info-box">
                                    <p><strong>Export Benefits:</strong></p>
                                    <ul>
                                        <li>Complete property dataset</li>
                                        <li>Ready for external analysis</li>
                                        <li>Compatible with Excel/Sheets</li>
                                        <li>Includes all property attributes</li>
                                    </ul>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.warning(f"No data found for {selected_county}, {selected_state}")
                    st.info("Try selecting a different state and county combination")

if __name__ == "__main__":
    main()
