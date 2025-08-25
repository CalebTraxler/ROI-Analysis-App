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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(page_title="3D ROI Analysis", layout="wide")

# Constants
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
GEOCODE_CACHE_FILE = CACHE_DIR / "geocode_cache.pkl"
PROCESSED_DATA_CACHE = CACHE_DIR / "processed_data_cache.pkl"

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

# Batch geocoding with parallel processing and better error handling
def batch_geocode_parallel(locations, state, county, max_workers=5):
    """Parallel geocoding with rate limiting and better error handling"""
    geolocator = Nominatim(user_agent="my_roi_app_v2")
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
        return cached_locations
    
    logger.info(f"Found {len(cached_locations)} cached locations, geocoding {len(uncached_locations)} new locations")
    
    def geocode_single(loc):
        """Geocode a single location with retry logic"""
        location_key = f"{loc}_{county}_{state}"
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Add longer timeout and better error handling for Streamlit Cloud
                location = geolocator.geocode(f"{loc}, {county} County, {state}, USA", timeout=30)
                if location:
                    save_location_to_cache(location_key, location.latitude, location.longitude, state, county)
                    return loc, (location.latitude, location.longitude)
                else:
                    return loc, (None, None)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {loc}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Failed to geocode {loc} after {max_retries} attempts")
                    return loc, (None, None)
        
        return loc, (None, None)
    
    # Use fewer workers and longer delays for Streamlit Cloud compatibility
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_to_loc = {executor.submit(geocode_single, loc): loc for loc in uncached_locations}
        
        for future in as_completed(future_to_loc):
            loc = future_to_loc[future]
            try:
                result = future.result()
                if result:
                    results[result[0]] = result[1]
                # Add longer delay between requests for Streamlit Cloud
                time.sleep(2)
            except Exception as e:
                logger.error(f"Exception occurred while geocoding {loc}: {e}")
                results[loc] = (None, None)
    
    # Combine cached and new results
    results.update(cached_locations)
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
def load_area_data_optimized(state, county):
    """Optimized data loading with enhanced caching and fallback mechanisms"""
    cache_key = f"{state}_{county}_data"
    
    # Try to load from cache first
    cached_data = load_from_cache(cache_key)
    if cached_data is not None:
        logger.info(f"Loaded {len(cached_data)} rows from cache for {county}, {state}")
        return cached_data
    
    logger.info(f"Loading and preprocessing data for {county}, {state}")
    
    # Load the main dataset
    df = preprocess_main_dataset()
    
    # Filter by state and county
    filtered_df = df[(df['State'] == state) & (df['CountyName'] == county)].copy()
    
    if len(filtered_df) == 0:
        logger.warning(f"No data found for {county}, {state}")
        return pd.DataFrame()
    
    # Try to get coordinates from cache first
    coordinates = get_cached_coordinates(state, county)
    
    if coordinates is None or len(coordinates) == 0:
        # If no cached coordinates, try to geocode
        try:
            logger.info(f"Geocoding {len(filtered_df)} locations for {county}, {state}")
            coordinates = batch_geocode_parallel(filtered_df['RegionName'].unique(), state, county)
        except Exception as e:
            logger.error(f"Geocoding failed for {county}, {state}: {e}")
            # Fallback: use default coordinates or skip geocoding
            coordinates = {}
    
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
    
    # Cache the processed data
    cache_processed_data(cache_key, filtered_df)
    
    logger.info(f"Processed {len(filtered_df)} rows for {county}, {state}")
    return filtered_df

# Additional caching for coordinates
def get_cached_coordinates(cache_key):
    """Get cached coordinates for a state-county combination"""
    if PROCESSED_DATA_CACHE.exists():
        try:
            with open(PROCESSED_DATA_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data.get(cache_key)
        except Exception as e:
            logger.warning(f"Failed to load coordinate cache: {e}")
    return None

def cache_coordinates(cache_key, coordinates):
    """Cache coordinates for a state-county combination"""
    try:
        cache_data = {}
        if PROCESSED_DATA_CACHE.exists():
            with open(PROCESSED_DATA_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
        
        cache_data[cache_key] = coordinates
        
        with open(PROCESSED_DATA_CACHE, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Failed to save coordinate cache: {e}")

# Add missing caching functions
def load_from_cache(cache_key):
    """Load processed data from cache"""
    if PROCESSED_DATA_CACHE.exists():
        try:
            with open(PROCESSED_DATA_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data.get(cache_key)
        except Exception as e:
            logger.warning(f"Failed to load processed data cache: {e}")
    return None

def cache_processed_data(cache_key, data):
    """Cache processed data"""
    try:
        cache_data = {}
        if PROCESSED_DATA_CACHE.exists():
            with open(PROCESSED_DATA_CACHE, 'rb') as f:
                cache_data = pickle.load(f)
        
        cache_data[cache_key] = data
        
        with open(PROCESSED_DATA_CACHE, 'wb') as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        logger.warning(f"Failed to save processed data cache: {e}")

# Optimized 3D map creation
def create_3d_roi_map_optimized(data):
    """Optimized version of create_3d_roi_map"""
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
        pitch=45,
        bearing=0
    )

    # Format the data and create color scale based on ROI
    scatter_data = valid_data.copy()
    scatter_data['tooltip_text'] = scatter_data.apply(
        lambda row: f"{row['RegionName']}<br/>${row['Current_Value']:,.2f}<br/>ROI: {row['ROI']:.2f}%",
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
        # Create color gradient from light to dark orange
        return [
            255,  # R
            int(140 * (1 - normalized)),  # G (decreases with higher ROI)
            0,    # B
            180   # Alpha (transparency)
        ]
    
    scatter_data['color'] = scatter_data['ROI'].apply(get_color_by_roi)

    # Create heatmap layer with exponential weighting for ROI
    scatter_data['weighted_roi'] = np.exp(scatter_data['ROI'] / 50) - 1  # Exponential scaling
    
    # Create the deck with improved configuration
    deck = pdk.Deck(
        layers=[
            pdk.Layer(
                'HeatmapLayer',
                scatter_data,
                get_position=['Longitude', 'Latitude'],
                get_weight='weighted_roi',
                radiusPixels=80,
                intensity=1.5,
                threshold=0.01,
                colorRange=[
                    [255, 255, 178, 100],  # Light yellow
                    [254, 204, 92, 150],   # Yellow
                    [253, 141, 60, 200],   # Orange
                    [240, 59, 32, 250],    # Red-Orange
                    [189, 0, 38, 255]      # Deep Red
                ],
                pickable=False
            ),
            pdk.Layer(
                'ScatterplotLayer',
                scatter_data,
                get_position=['Longitude', 'Latitude'],
                get_radius=40,
                get_fill_color='color',
                get_line_color=[255, 255, 255, 100],
                pickable=True,
                opacity=0.9,
                stroked=True,
                filled=True,
                line_width_min_pixels=1
            )
        ],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={
            "html": "<b>{tooltip_text}</b>",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white",
                "padding": "10px",
                "borderRadius": "5px"
            }
        },
        height=600
    )

    return deck

# Main app section with performance optimizations
def main():
    # Title and description
    st.title("3D Neighborhood ROI Analysis")
    
    st.markdown("""
    This visualization shows Return on Investment (ROI) patterns across neighborhoods using a 3D map.
    - Heat intensity represents the ROI percentage
    - Color intensity indicates higher ROI values
    - Hover over points to see detailed information
    """)
    
    # Network status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Network Status**")
    
    # Check if we can reach external services
    try:
        import requests
        response = requests.get("https://nominatim.openstreetmap.org", timeout=5)
        if response.status_code == 200:
            st.sidebar.success("✅ External services accessible")
            network_available = True
        else:
            st.sidebar.warning("⚠️ Limited external access")
            network_available = False
    except:
        st.sidebar.error("❌ No external network access")
        st.sidebar.info("Using cached coordinates only")
        network_available = False
    
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
    
    states, state_county_map = get_states_and_counties()
    
    # Sidebar selection
    selected_state = st.sidebar.selectbox("Select State", states)
    
    if selected_state:
        counties = state_county_map[selected_state]
        selected_county = st.sidebar.selectbox("Select County", counties)
        
        # Performance metrics
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Performance Info**")
        
        # Progress indicator with better UX
        if selected_state and selected_county:
            # Check if data is already cached
            cache_key = f"{selected_state}_{selected_county}_data"
            
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
                    data = load_area_data_optimized(selected_state, selected_county)
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.info("Try selecting a different state/county or check the logs")
                    return
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if len(data) > 0:
                    # Display statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average ROI", f"{data['ROI'].mean():.2f}%")
                    with col2:
                        st.metric("Median Home Value", f"${data['Current_Value'].median():,.2f}")
                    with col3:
                        st.metric("Number of Neighborhoods", len(data))
                    
                    # Check if we have coordinates for visualization
                    if data['Latitude'].isna().sum() == len(data):
                        st.warning("⚠️ No coordinates available for visualization. The app may be experiencing network restrictions.")
                        st.info("Try refreshing the page or selecting a different location.")
                    else:
                        # Create and display the map
                        map_chart = create_3d_roi_map_optimized(data)
                        if map_chart:
                            st.pydeck_chart(map_chart, use_container_width=True)
                        else:
                            st.error("Failed to create map visualization")
                    
                    # Add data table with pagination for better performance
                    with st.expander("View Raw Data"):
                        # Add search functionality
                        search_term = st.text_input("Search neighborhoods:", key="search_input")
                        if search_term:
                            filtered_data = data[data['RegionName'].str.contains(search_term, case=False, na=False)]
                        else:
                            filtered_data = data
                        
                        # Pagination
                        page_size = 50
                        total_pages = (len(filtered_data) + page_size - 1) // page_size
                        page = st.selectbox("Page:", range(1, total_pages + 1), key="page_select")
                        
                        start_idx = (page - 1) * page_size
                        end_idx = start_idx + page_size
                        page_data = filtered_data.iloc[start_idx:end_idx]
                        
                        st.dataframe(
                            page_data[['RegionName', 'Current_Value', 'ROI']].sort_values('ROI', ascending=False),
                            use_container_width=True
                        )
                        
                        st.caption(f"Showing {start_idx + 1}-{min(end_idx, len(filtered_data))} of {len(filtered_data)} neighborhoods")
                else:
                    st.warning("No data found for the selected state and county combination.")

if __name__ == "__main__":
    main()
