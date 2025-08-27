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
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    
    .metric-container {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #374151;
        margin: 2rem 0 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .sidebar-section {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
    }
    
    .data-table {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .success-box {
        background-color: #d1fae5;
        border: 1px solid #10b981;
        border-radius: 6px;
        padding: 0.75rem;
        color: #065f46;
    }
    
    .warning-box {
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        border-radius: 6px;
        padding: 0.75rem;
        color: #92400e;
    }
    
    .info-box {
        background-color: #dbeafe;
        border: 1px solid #3b82f6;
        border-radius: 6px;
        padding: 0.75rem;
        color: #1e40af;
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
    """Create a map with guaranteed OpenStreetMap background and optional property overlay"""
    if len(data) == 0:
        return None
    
    # Filter out rows with missing coordinates
    valid_data = data.dropna(subset=['Latitude', 'Longitude'])
    if len(valid_data) == 0:
        return None
    
    # Calculate view state
    center_lat = valid_data['Latitude'].mean()
    center_lon = valid_data['Longitude'].mean()
    
    # Calculate zoom level
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

    # Create simple scatter plot with better visibility
    scatter_data = valid_data.copy()
    scatter_data['tooltip_text'] = scatter_data.apply(
        lambda row: f"{row['RegionName']}<br/>${row['Current_Value']:,.0f}<br/>ROI: {row['ROI']:.1f}%",
        axis=1
    )
    
    # Color by ROI with better contrast
    min_roi = scatter_data['ROI'].min()
    max_roi = scatter_data['ROI'].max()
    
    def get_color_by_roi(roi):
        if max_roi == min_roi:
            normalized = 0.5
        else:
            normalized = (roi - min_roi) / (max_roi - min_roi)
        
        # Red to Green gradient with better visibility
        if normalized < 0.5:
            # Red to Yellow
            return [255, int(255 * normalized * 2), 0, 220]
        else:
            # Yellow to Green
            return [int(255 * (1 - (normalized - 0.5) * 2)), 255, 0, 220]
    
    scatter_data['color'] = scatter_data['ROI'].apply(get_color_by_roi)
    
    # Create heatmap layer with exponential weighting (proven working method)
    scatter_data['weighted_roi'] = np.exp(scatter_data['ROI'] / 50) - 1  # Exponential scaling
    
    # Create heatmap layer
    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        scatter_data,
        get_position=['Longitude', 'Latitude'],
        get_weight='weighted_roi',
        radiusPixels=60,
        intensity=2,
        threshold=0.02,
        colorRange=[
            [255, 255, 178, 100],  # Light yellow (low ROI)
            [254, 204, 92, 150],   # Yellow
            [253, 141, 60, 200],   # Orange
            [240, 59, 32, 250],    # Red-Orange
            [189, 0, 38, 255]      # Deep Red (high ROI)
        ],
        pickable=False
    )
    
    # Create scatter layer with better visibility
    scatter_layer = pdk.Layer(
        'ScatterplotLayer',
        scatter_data,
        get_position=['Longitude', 'Latitude'],
        get_radius=30,  # Smaller radius for individual points
        get_fill_color='color',
        get_line_color=[255, 255, 255, 200],
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        line_width_min_pixels=2,
        radius_scale=1
    )
    
    # Create coordinate grid lines for better orientation
    grid_data = []
    lat_min, lat_max = valid_data['Latitude'].min(), valid_data['Latitude'].max()
    lon_min, lon_max = valid_data['Longitude'].min(), valid_data['Longitude'].max()
    
    # Add latitude lines
    for lat in np.linspace(lat_min, lat_max, 5):
        grid_data.append({
            'path': [[lon_min, lat], [lon_max, lat]],
            'type': 'lat'
        })
    
    # Add longitude lines
    for lon in np.linspace(lon_min, lon_max, 5):
        grid_data.append({
            'path': [[lon, lat_min], [lon, lat_max]],
            'type': 'lon'
        })
    
    # Create grid layer
    grid_layer = pdk.Layer(
        'PathLayer',
        grid_data,
        get_path='path',
        get_color='[200, 200, 200, 100]',
        get_width=1,
        pickable=False
    )
    
    # Add properties layer if available
    all_layers = [grid_layer, heatmap_layer, scatter_layer]
    if properties_df is not None and not properties_df.empty:
        valid_properties = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()].copy()
        
        if len(valid_properties) > 0:
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
            
            properties_layer = pdk.Layer(
                'ScatterplotLayer',
                properties_with_tooltips,
                get_position=['longitude', 'latitude'],
                get_radius=8,
                get_fill_color=[0, 100, 200, 180],
                get_line_color=[255, 255, 255, 150],
                pickable=True,
                opacity=0.7,
                stroked=True,
                filled=True,
                line_width_min_pixels=1
            )
            all_layers.append(properties_layer)
    
    # Try to create deck with OpenStreetMap style
    try:
        deck = pdk.Deck(
            layers=all_layers,  # Include all layers including properties
            initial_view_state=view_state,
            map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json',
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
    except Exception as e:
        logger.warning(f"Failed to create map with CartoDB style: {e}")
        
        # Final fallback: no map style but with coordinate grid
        try:
            deck = pdk.Deck(
                layers=all_layers,  # Include all layers including properties
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
        except Exception as e2:
            logger.error(f"Failed to create any map: {e2}")
            return None

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
            
            # Create properties layer with smaller radius for individual houses
            properties_layer = pdk.Layer(
                'ScatterplotLayer',
                properties_with_tooltips,
                get_position=['longitude', 'latitude'],
                get_radius=8,  # Small radius for individual properties
                get_fill_color=[0, 100, 200, 180],  # Blue color for properties
                get_line_color=[255, 255, 255, 150],
                pickable=True,
                opacity=0.7,
                stroked=True,
                filled=True,
                line_width_min_pixels=1,
                radius_scale=1
            )
            
            # Add properties layer on top
            layers.append(properties_layer)
            
            # Update tooltip to show both ROI and property info
            scatter_data['tooltip_text'] = scatter_data.apply(
                lambda row: f"<b>Neighborhood: {row['RegionName']}</b><br/>"
                           f"ROI: {row['ROI']:.1f}%<br/>"
                           f"Value: ${row['Current_Value']:,.0f}<br/>"
                           f"<i>Zoom in to see individual properties</i>",
                axis=1
            )

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
    # Initialize cache database
    setup_coordinates_cache()
    
    # Title and description with professional styling
    st.markdown('<h1 class="main-header">Real Estate ROI Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    **Interactive visualization of real estate Return on Investment (ROI) patterns across neighborhoods**
    
    The dashboard provides comprehensive analysis of property value trends, ROI performance metrics, and geographic distribution patterns to support data-driven investment decisions.
    """)
    
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
    
    # Sidebar selection with professional labels
    selected_state = st.sidebar.selectbox("Select State", states, key="state_select")
    
    if selected_state:
        counties = state_county_map[selected_state]
        selected_county = st.sidebar.selectbox("Select County", counties, key="county_select")
        
        # Map style options
        st.sidebar.markdown("### Visualization Options")
        use_satellite = st.sidebar.checkbox("Satellite view", help="Use satellite imagery as base map (requires network)")
        
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
                            with st.spinner('Loading property data for map overlay...'):
                                try:
                                    osm_fetcher = OpenStreetMapProperties()
                                    properties_df = osm_fetcher.get_county_properties(
                                        selected_county, selected_state, max_properties=1000
                                    )
                                    if not properties_df.empty:
                                        st.success(f"✅ Loaded {len(properties_df)} properties")
                                    else:
                                        st.warning("⚠️ No properties found for this county")
                                except Exception as e:
                                    st.error(f"❌ Error loading properties: {str(e)}")
                                    properties_df = None
                        
                        # Try to create the enhanced map with properties overlay
                        map_chart = create_3d_roi_map_optimized(data, use_satellite and network_available, properties_df)
                        
                        # If enhanced map fails, try fallback map
                        if not map_chart:
                            map_chart = create_robust_fallback_map(data, properties_df)
                        
                        if map_chart:
                            st.pydeck_chart(map_chart, use_container_width=True)
                            
                            # Enhanced legend with property information
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("""
                                **ROI Color Legend:**
                                - **Light Yellow**: Lower ROI
                                - **Orange**: Medium ROI  
                                - **Deep Red**: Higher ROI
                                """)
                            with col2:
                                if properties_df is not None and not properties_df.empty:
                                    st.markdown("""
                                    **Property Overlay:**
                                    - **Blue dots**: Individual houses
                                    - **Zoom in** to see properties clearly
                                    - **Click properties** for details
                                    """)
                    
                    # ROI Distribution with professional styling
                    st.markdown('<h3 class="section-header">ROI Performance Analysis</h3>', unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Top performers
                        top_roi = data.nlargest(10, 'ROI')[['RegionName', 'ROI', 'Current_Value']]
                        st.markdown("**Top ROI Performers**")
                        st.dataframe(top_roi, use_container_width=True, hide_index=True)
                    
                    with col2:
                        # Bottom performers
                        bottom_roi = data.nsmallest(10, 'ROI')[['RegionName', 'ROI', 'Current_Value']]
                        st.markdown("**Lowest ROI Areas**")
                        st.dataframe(bottom_roi, use_container_width=True, hide_index=True)
                    
                    # Add data table with enhanced search and pagination
                    with st.expander("View Complete Data Table"):
                        st.markdown('<h3 class="section-header">Data Exploration & Filtering</h3>', unsafe_allow_html=True)
                        # Search and filter options
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            search_term = st.text_input("Search neighborhoods:", key="search_input")
                        with col2:
                            min_roi = st.number_input("Min ROI %:", value=float(data['ROI'].min()), key="min_roi")
                        with col3:
                            max_roi = st.number_input("Max ROI %:", value=float(data['ROI'].max()), key="max_roi")
                        
                        # Apply filters
                        filtered_data = data.copy()
                        if search_term:
                            filtered_data = filtered_data[filtered_data['RegionName'].str.contains(search_term, case=False, na=False)]
                        filtered_data = filtered_data[(filtered_data['ROI'] >= min_roi) & (filtered_data['ROI'] <= max_roi)]
                        
                        # Sort options
                        sort_by = st.selectbox("Sort by:", ['ROI', 'Current_Value', 'RegionName'], key="sort_select")
                        sort_order = st.radio("Order:", ['Descending', 'Ascending'], key="sort_order", horizontal=True)
                        ascending = sort_order == 'Ascending'
                        
                        filtered_data = filtered_data.sort_values(sort_by, ascending=ascending)
                        
                        # Pagination
                        page_size = st.selectbox("Items per page:", [25, 50, 100], index=1, key="page_size")
                        total_pages = max(1, (len(filtered_data) + page_size - 1) // page_size)
                        page = st.selectbox("Page:", range(1, total_pages + 1), key="page_select")
                        
                        start_idx = (page - 1) * page_size
                        end_idx = start_idx + page_size
                        page_data = filtered_data.iloc[start_idx:end_idx]
                        
                        # Display table with better formatting
                        display_data = page_data[['RegionName', 'Current_Value', 'ROI', 'First_Value']].copy()
                        display_data['Current_Value'] = display_data['Current_Value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                        display_data['First_Value'] = display_data['First_Value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
                        display_data['ROI'] = display_data['ROI'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                        display_data.columns = ['Neighborhood', 'Current Value', 'ROI %', 'Initial Value']
                        
                        st.dataframe(display_data, use_container_width=True, hide_index=True)
                        st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(filtered_data))} of {len(filtered_data)} neighborhoods")
                        
                        # Download data option
                        csv_data = filtered_data.to_csv(index=False)
                        st.download_button(
                            label="Download filtered data as CSV",
                            data=csv_data,
                            file_name=f"{selected_county}_{selected_state}_roi_data.csv",
                            mime="text/csv"
                        )
                    
                    # OpenStreetMap Properties Section
                    st.markdown('<h3 class="section-header">OpenStreetMap Property Data</h3>', unsafe_allow_html=True)
                    
                    # Add property data loading option
                    load_properties = st.checkbox("Load OpenStreetMap property data for this county", 
                                               help="Fetch detailed property information from OpenStreetMap")
                    
                    if load_properties:
                        with st.spinner('Loading OpenStreetMap property data...'):
                            try:
                                # Initialize OSM properties fetcher
                                osm_fetcher = OpenStreetMapProperties()
                                
                                # Get properties for the selected county
                                properties_df = osm_fetcher.get_county_properties(selected_county, selected_state, max_properties=500)
                                
                                if not properties_df.empty:
                                    # Display property summary
                                    property_summary = osm_fetcher.get_property_summary(properties_df)
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.markdown(f'<div class="metric-container"><strong>Total Properties</strong><br><span style="font-size: 1.5rem; color: #10b981;">{property_summary.get("total_properties", 0)}</span></div>', unsafe_allow_html=True)
                                    with col2:
                                        avg_value = property_summary.get("avg_estimated_value", 0)
                                        st.markdown(f'<div class="metric-container"><strong>Avg Estimated Value</strong><br><span style="font-size: 1.5rem; color: #10b981;">${avg_value:,.0f}</span></div>', unsafe_allow_html=True)
                                    with col3:
                                        coord_coverage = property_summary.get("coordinate_coverage", 0)
                                        st.markdown(f'<div class="metric-container"><strong>Coordinate Coverage</strong><br><span style="font-size: 1.5rem; color: #10b981;">{coord_coverage:.0f}%</span></div>', unsafe_allow_html=True)
                                    with col4:
                                        address_coverage = property_summary.get("address_coverage", 0)
                                        st.markdown(f'<div class="metric-container"><strong>Address Coverage</strong><br><span style="font-size: 1.5rem; color: #10b981;">{address_coverage:.0f}%</span></div>', unsafe_allow_html=True)
                                    
                                    # Property type distribution
                                    st.markdown('<h4 class="section-header">Property Type Distribution</h4>', unsafe_allow_html=True)
                                    property_types = property_summary.get("property_types", {})
                                    if property_types:
                                        type_df = pd.DataFrame(list(property_types.items()), columns=['Property Type', 'Count'])
                                        st.bar_chart(type_df.set_index('Property Type'))
                                    
                                    # Properties map
                                    if properties_df['latitude'].notna().sum() > 0:
                                        st.markdown('<h4 class="section-header">Properties Map</h4>', unsafe_allow_html=True)
                                        
                                        # Create properties map
                                        properties_map = create_properties_map(properties_df)
                                        if properties_map:
                                            st.pydeck_chart(properties_map, use_container_width=True)
                                    
                                    # Properties data table
                                    with st.expander("View Properties Data"):
                                        st.markdown('<h4 class="section-header">Properties Data Table</h4>', unsafe_allow_html=True)
                                        
                                        # Clean up the DataFrame for display
                                        display_properties = properties_df.copy()
                                        
                                        # Flatten address and features columns for better display
                                        if 'address' in display_properties.columns:
                                            address_df = pd.json_normalize(display_properties['address'])
                                            display_properties = pd.concat([display_properties.drop('address', axis=1), address_df], axis=1)
                                        
                                        if 'features' in display_properties.columns:
                                            features_df = pd.json_normalize(display_properties['features'])
                                            display_properties = pd.concat([display_properties.drop('features', axis=1), features_df], axis=1)
                                        
                                        # Select key columns for display
                                        key_columns = ['osm_id', 'building_type', 'latitude', 'longitude', 'estimated_value', 
                                                     'street', 'housenumber', 'city', 'state', 'postcode', 'floors', 'units']
                                        available_columns = [col for col in key_columns if col in display_properties.columns]
                                        
                                        # Format numeric columns
                                        if 'estimated_value' in display_properties.columns:
                                            display_properties['estimated_value'] = display_properties['estimated_value'].apply(
                                                lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A"
                                            )
                                        
                                        st.dataframe(display_properties[available_columns], use_container_width=True, hide_index=True)
                                        
                                        # Download properties data
                                        properties_csv = properties_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download properties data as CSV",
                                            data=properties_csv,
                                            file_name=f"{selected_county}_{selected_state}_properties.csv",
                                            mime="text/csv"
                                        )
                                        
                                else:
                                    st.warning("No OpenStreetMap properties found for this county. This could be due to:")
                                    st.info("""
                                    - Limited OpenStreetMap coverage in rural areas
                                    - County boundaries not matching OSM data
                                    - API rate limiting (try again in a few minutes)
                                    """)
                                    
                            except Exception as e:
                                st.error(f"Error loading OpenStreetMap properties: {str(e)}")
                                st.info("This feature requires internet connection and may be rate-limited. Try again later.")
                else:
                    st.warning(f"No data found for {selected_county}, {selected_state}")
                    st.info("Try selecting a different state and county combination")

if __name__ == "__main__":
    main()
