import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import sqlite3
import time
from pathlib import Path

# Must be the first Streamlit command
st.set_page_config(page_title="3D ROI Analysis", layout="wide")

# Create a coordinates cache database
def setup_coordinates_cache():
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS coordinates
                 (location_key TEXT PRIMARY KEY, latitude REAL, longitude REAL)''')
    conn.commit()
    conn.close()

# Cache decorator for expensive computations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cached_location(location_key):
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('SELECT latitude, longitude FROM coordinates WHERE location_key = ?', (location_key,))
    result = c.fetchone()
    conn.close()
    return result

def save_location_to_cache(location_key, lat, lon):
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO coordinates (location_key, latitude, longitude) VALUES (?, ?, ?)',
              (location_key, lat, lon))
    conn.commit()
    conn.close()

@st.cache_data
def batch_geocode(locations, state, county):
    geolocator = Nominatim(user_agent="my_roi_app")
    results = []
    
    for loc in locations:
        # Create a unique key for this location
        location_key = f"{loc}_{county}_{state}"
        
        # Check cache first
        cached_result = get_cached_location(location_key)
        if cached_result:
            results.append(cached_result)
            continue
            
        try:
            # If not in cache, geocode and cache the result
            location = geolocator.geocode(f"{loc}, {county} County, {state}, USA", timeout=10)
            if location:
                save_location_to_cache(location_key, location.latitude, location.longitude)
                results.append((location.latitude, location.longitude))
            else:
                results.append((None, None))
            time.sleep(1)  # Respect Nominatim's usage policy
        except Exception as e:
            print(f"Error geocoding {loc}: {str(e)}")
            results.append((None, None))
    
    return results

@st.cache_data
def load_area_data(state, county=None):
    # Initialize the cache database if it doesn't exist
    setup_coordinates_cache()
    
    # Load and preprocess the main data
    df = pd.read_csv('Traxler-ROI/Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    
    if state:
        df = df[df['State'] == state]
    if county:
        df = df[df['CountyName'] == county]
        
        # Batch geocode all locations
        unique_locations = df['RegionName'].unique()
        coordinates = batch_geocode(unique_locations, state, county)
        
        # Create a mapping of locations to coordinates
        coord_dict = {loc: coord for loc, coord in zip(unique_locations, coordinates)}
        
        # Add coordinates to dataframe
        df['Latitude'] = df['RegionName'].map(lambda x: coord_dict[x][0] if coord_dict[x] else None)
        df['Longitude'] = df['RegionName'].map(lambda x: coord_dict[x][1] if coord_dict[x] else None)
    
    # Process time series data
    date_cols = [col for col in df.columns if len(col) == 10 and '-' in col]
    date_cols.sort()
    first_date = date_cols[0]
    last_date = date_cols[-1]
    
    df['Current_Value'] = pd.to_numeric(df[last_date], errors='coerce')
    df['Previous_Value'] = pd.to_numeric(df[first_date], errors='coerce')
    df['ROI'] = ((df['Current_Value'] - df['Previous_Value']) / df['Previous_Value'] * 100)
    
    return df.dropna(subset=['Current_Value', 'ROI', 'Latitude', 'Longitude'])

def create_3d_roi_map(data):
    # Calculate view state
    view_state = pdk.ViewState(
        latitude=data['Latitude'].mean(),
        longitude=data['Longitude'].mean(),
        zoom=min(20 / max(data['Latitude'].max() - data['Latitude'].min(), 
                         data['Longitude'].max() - data['Longitude'].min()), 12),
        pitch=45,
        bearing=0
    )

    # Format the data and create color scale based on ROI
    scatter_data = data.copy()
    scatter_data['tooltip_text'] = scatter_data.apply(
        lambda row: f"{row['RegionName']}<br/>${row['Current_Value']:,.2f}<br/>ROI: {row['ROI']:.2f}%",
        axis=1
    )
    
    def get_color_by_roi(roi):
        # Normalize ROI to 0-1 scale
        normalized = (roi - min_roi) / (max_roi - min_roi)
        # Create color gradient from light to dark orange
        return [
            255,  # R
            int(140 * (1 - normalized)),  # G (decreases with higher ROI)
            0,    # B
            180   # Alpha (transparency)
        ]
    
    min_roi = scatter_data['ROI'].min()
    max_roi = scatter_data['ROI'].max()
    scatter_data['color'] = scatter_data['ROI'].apply(get_color_by_roi)

    # Create heatmap layer with exponential weighting for ROI
    scatter_data['weighted_roi'] = np.exp(scatter_data['ROI'] / 50) - 1  # Exponential scaling
    
    heatmap_layer = pdk.Layer(
        'HeatmapLayer',
        scatter_data,
        get_position=['Longitude', 'Latitude'],
        get_weight='weighted_roi',
        radiusPixels=60,
        intensity=2,
        threshold=0.02,
        colorRange=[
            [255, 255, 178, 100],  # Light yellow
            [254, 204, 92, 150],   # Yellow
            [253, 141, 60, 200],   # Orange
            [240, 59, 32, 250],    # Red-Orange
            [189, 0, 38, 255]      # Deep Red
        ],
        pickable=False
    )

    # Create scatter layer for tooltips
    scatter_layer = pdk.Layer(
        'ScatterplotLayer',
        scatter_data,
        get_position=['Longitude', 'Latitude'],
        get_radius=30,
        get_fill_color='color',
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True
    )

    # Create the deck with tooltip configuration
    deck = pdk.Deck(
        layers=[heatmap_layer, scatter_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={
            "html": "<b>{tooltip_text}</b>",
            "style": {
                "backgroundColor": "steelblue",
                "color": "white"
            }
        }
    )

    return deck

# Main app section
@st.cache_data
def load_initial_data():
    return pd.read_csv('Real_Estate_Data/Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')

# Title and description
st.title("3D Neighborhood ROI Analysis")

st.markdown("""
This visualization shows Return on Investment (ROI) patterns across neighborhoods using a 3D map.
- Heat intensity represents the ROI percentage
- Color intensity indicates higher ROI values
- Hover over points to see detailed information
""")

# Load and filter data for dropdowns
initial_data = load_initial_data()
states = initial_data['State'].unique()
selected_state = st.sidebar.selectbox("Select State", sorted(states))

counties = initial_data[initial_data['State'] == selected_state]['CountyName'].unique()
selected_county = st.sidebar.selectbox("Select County", sorted(counties))

# Progress indicator
if selected_state and selected_county:
    with st.spinner('Loading data and generating visualization...'):
        data = load_area_data(selected_state, selected_county)
        
        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average ROI", f"{data['ROI'].mean():.2f}%")
        with col2:
            st.metric("Median Home Value", f"${data['Current_Value'].median():,.2f}")
        with col3:
            st.metric("Number of Neighborhoods", len(data))
        
        # Create and display the map
        st.pydeck_chart(create_3d_roi_map(data))
        
        # Add data table
        with st.expander("View Raw Data"):
            st.dataframe(data[['RegionName', 'Current_Value', 'ROI']].sort_values('ROI', ascending=False))
