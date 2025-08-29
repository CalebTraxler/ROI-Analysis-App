import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import sqlite3
import time
from pathlib import Path
import logging
from openstreetmap_properties import OpenStreetMapProperties

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Must be the first Streamlit command
st.set_page_config(page_title="Enhanced ROI Analysis with Property Data", layout="wide")

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
    df = pd.read_csv('Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    
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
    
    # Calculate ROI and current value
    df['Current_Value'] = df[last_date]
    df['Previous_Value'] = df[first_date]
    df['ROI'] = ((df['Current_Value'] - df['Previous_Value']) / df['Previous_Value']) * 100
    
    return df

def get_color_by_roi(roi):
    """Get color based on ROI value"""
    if pd.isna(roi):
        return [128, 128, 128]  # Gray for missing data
    
    if roi < 0:
        return [255, 0, 0]      # Red for negative ROI
    elif roi < 10:
        return [255, 255, 0]    # Yellow for low ROI
    elif roi < 20:
        return [0, 255, 0]      # Green for medium ROI
    else:
        return [0, 0, 255]      # Blue for high ROI

def create_clickable_property_map(data, properties_df, zoom_level=10):
    """Create a map specifically for property interaction with click handling"""
    
    # Filter data with valid coordinates
    valid_data = data[data['Latitude'].notna() & data['Longitude'].notna()].copy()
    
    if valid_data.empty:
        return None
    
    # Calculate center point
    center_lat = valid_data['Latitude'].mean()
    center_lon = valid_data['Longitude'].mean()
    
    # Create view state
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom_level,
        pitch=0,
        bearing=0
    )
    
    layers = []
    
    # Add ROI heatmap layer
    scatter_data = valid_data.copy()
    scatter_data['weighted_roi'] = np.exp(scatter_data['ROI'] / 50) - 1
    
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
    layers.append(heatmap_layer)
    
    # Add neighborhood points
    scatter_data['tooltip_text'] = scatter_data.apply(
        lambda row: f"<b>{row['RegionName']}</b><br/>ROI: {row['ROI']:.2f}%",
        axis=1
    )
    
    neighborhood_layer = pdk.Layer(
        'ScatterplotLayer',
        scatter_data,
        get_position=['Longitude', 'Latitude'],
        get_radius=20,
        get_fill_color=[255, 255, 255, 150],
        get_line_color=[0, 0, 0, 200],
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True
    )
    layers.append(neighborhood_layer)
    
    # Add properties layer if available and zoomed in
    if properties_df is not None and not properties_df.empty and zoom_level >= 12:
        valid_properties = properties_df[
            (properties_df['latitude'].notna()) & 
            (properties_df['longitude'].notna())
        ].copy()
        
        if len(valid_properties) > 0:
            # Add unique identifier and make properties clickable
            valid_properties['layer_id'] = 'property'
            valid_properties['clickable'] = True
            
            # Create enhanced tooltip for properties
            valid_properties['tooltip_text'] = valid_properties.apply(
                lambda row: f"<b>Property</b><br/>"
                           f"Type: {row.get('building_type', 'N/A')}<br/>"
                           f"Click for details",
                axis=1
            )
            
            properties_layer = pdk.Layer(
                'ScatterplotLayer',
                valid_properties,
                get_position=['longitude', 'latitude'],
                get_radius=10,
                get_fill_color=[0, 100, 200, 200],
                get_line_color=[255, 255, 255, 255],
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                line_width_min_pixels=2
            )
            
            layers.append(properties_layer)
    
    # Create the deck with click handling
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={
            "html": "<b>{tooltip_text}</b>",
            "style": {
                "backgroundColor": "rgba(0, 0, 0, 0.8)",
                "color": "white",
                "padding": "8px",
                "borderRadius": "4px",
                "fontSize": "12px"
            }
        }
    )
    
    return deck

def create_property_details_sidebar(clicked_property):
    """Create sidebar with detailed property information"""
    if not clicked_property:
        return
    
    st.sidebar.markdown("## 🏠 Property Details")
    
    # Extract property information
    osm_id = clicked_property.get('osm_id', 'N/A')
    building_type = clicked_property.get('building_type', 'N/A')
    address = clicked_property.get('address', {})
    
    # Display basic info
    st.sidebar.markdown(f"**OSM ID:** {osm_id}")
    st.sidebar.markdown(f"**Type:** {building_type}")
    
    # Display address
    if address:
        st.sidebar.markdown("**Address:**")
        if address.get('housenumber'):
            st.sidebar.markdown(f"  {address.get('housenumber')} {address.get('street', '')}")
        if address.get('city'):
            st.sidebar.markdown(f"  {address.get('city')}, {address.get('state', '')} {address.get('postcode', '')}")
    
    # Display features
    features = clicked_property.get('features', {})
    if features:
        st.sidebar.markdown("**Features:**")
        if features.get('floors'):
            st.sidebar.markdown(f"  • Floors: {features['floors']}")
        if features.get('units'):
            st.sidebar.markdown(f"  • Units: {features['units']}")
        if features.get('year_built'):
            st.sidebar.markdown(f"  • Year Built: {features['year_built']}")
        if features.get('roof_type'):
            st.sidebar.markdown(f"  • Roof: {features['roof_type']}")
        if features.get('material'):
            st.sidebar.markdown(f"  • Material: {features['material']}")
    
    # Display estimated value
    if 'estimated_value' in clicked_property:
        st.sidebar.markdown(f"**Estimated Value:** ${clicked_property['estimated_value']:,.0f}")
    
    # Add a clear button
    if st.sidebar.button("Clear Selection"):
        st.session_state.clicked_property = None
        st.rerun()

def main():
    # Initialize session state for clicked properties
    if 'clicked_property' not in st.session_state:
        st.session_state.clicked_property = None
    
    # Initialize cache database
    setup_coordinates_cache()
    
    # Title and description
    st.title("🏠 Enhanced Real Estate ROI Analysis")
    
    st.markdown("""
    **Interactive visualization with OpenStreetMap property data**
    
    - **Heatmap**: ROI patterns across neighborhoods
    - **Properties**: Individual houses (zoom in to see)
    - **Click Interaction**: Click on properties for details
    - **Zoom-based Loading**: Properties load automatically when zoomed in
    """)
    
    # Sidebar controls
    st.sidebar.markdown("## 📍 Location Selection")
    
    # Load and filter data for dropdowns
    @st.cache_data
    def load_initial_data():
        return pd.read_csv('Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    
    initial_data = load_initial_data()
    states = initial_data['State'].unique()
    selected_state = st.sidebar.selectbox("Select State", sorted(states))
    
    counties = initial_data[initial_data['State'] == selected_state]['CountyName'].unique()
    selected_county = st.sidebar.selectbox("Select County", sorted(counties))
    
    # Map controls
    st.sidebar.markdown("## 🗺️ Map Controls")
    zoom_level = st.sidebar.slider("Zoom Level", min_value=8, max_value=18, value=12, 
                                  help="Higher zoom levels show more property details")
    
    load_properties = st.sidebar.checkbox("Enable Property Loading", value=True,
                                        help="Load OpenStreetMap property data when zoomed in")
    
    # Property loading settings
    if load_properties:
        st.sidebar.markdown("### 🏘️ Property Settings")
        max_properties = st.sidebar.slider("Max Properties", min_value=100, max_value=10000, 
                                         value=2000, help="Maximum properties to load")
        
        property_types = st.sidebar.multiselect(
            "Property Types to Show",
            options=['house', 'residential', 'apartments', 'detached', 'semi-detached'],
            default=['house', 'residential'],
            help="Select which types of properties to display"
        )
    
    # Main content area
    if selected_state and selected_county:
        with st.spinner('Loading data and generating visualization...'):
            # Load neighborhood data
            data = load_area_data(selected_state, selected_county)
            
            # Load properties if enabled and zoom level is appropriate
            properties_df = None
            if load_properties and zoom_level >= 12:
                with st.spinner('Loading property data...'):
                    try:
                        osm_fetcher = OpenStreetMapProperties()
                        properties_df = osm_fetcher.get_county_properties(
                            selected_county, selected_state, max_properties=max_properties
                        )
                        
                        # Filter by selected property types
                        if property_types and not properties_df.empty:
                            properties_df = properties_df[
                                properties_df['building_type'].isin(property_types)
                            ]
                        
                        if not properties_df.empty:
                            st.success(f"✅ Loaded {len(properties_df)} properties")
                        else:
                            st.warning("⚠️ No properties found for this area")
                    except Exception as e:
                        st.error(f"❌ Error loading properties: {str(e)}")
                        properties_df = None
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Average ROI", f"{data['ROI'].mean():.2f}%")
            with col2:
                st.metric("Median Home Value", f"${data['Current_Value'].median():,.2f}")
            with col3:
                st.metric("Neighborhoods", len(data))
            with col4:
                if properties_df is not None and not properties_df.empty:
                    valid_props = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()]
                    st.metric("Properties Loaded", len(valid_props))
                else:
                    st.metric("Properties Loaded", 0)
            
            # Create and display the map
            st.markdown("## 🗺️ Interactive Map")
            
            # Map instructions
            if properties_df is not None and not properties_df.empty:
                st.info("💡 **Tip**: Zoom in to see individual properties. Click on properties for detailed information.")
            
            # Create the map
            map_chart = create_clickable_property_map(data, properties_df, zoom_level)
            
            if map_chart:
                # Display the map
                st.pydeck_chart(map_chart, use_container_width=True)
                
                # Legend
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **ROI Color Legend:**
                    - 🔴 **Red**: Negative ROI
                    - 🟡 **Yellow**: Low ROI (0-10%)
                    - 🟢 **Green**: Medium ROI (10-20%)
                    - 🔵 **Blue**: High ROI (20%+)
                    """)
                
                with col2:
                    if properties_df is not None and not properties_df.empty:
                        st.markdown("""
                        **Property Legend:**
                        - 🔵 **Blue dots**: Individual properties
                        - **Zoom level 12+**: Properties become visible
                        - **Click properties**: View detailed information
                        """)
                    else:
                        st.markdown("""
                        **Property Legend:**
                        - Properties not loaded
                        - Zoom in to level 12+ to load properties
                        """)
                
                # Property details sidebar
                if st.session_state.clicked_property:
                    create_property_details_sidebar(st.session_state.clicked_property)
                
                # Data table
                with st.expander("📊 View Raw Neighborhood Data"):
                    st.dataframe(data[['RegionName', 'Current_Value', 'ROI', 'Latitude', 'Longitude']].sort_values('ROI', ascending=False))
                
                # Property data table
                if properties_df is not None and not properties_df.empty:
                    with st.expander("🏠 View Property Data"):
                        # Show summary statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Property Type Distribution:**")
                            type_counts = properties_df['building_type'].value_counts()
                            st.write(type_counts)
                        
                        with col2:
                            st.markdown("**Coordinate Coverage:**")
                            coord_coverage = (properties_df['latitude'].notna().sum() / len(properties_df)) * 100
                            st.metric("Properties with Coordinates", f"{coord_coverage:.1f}%")
                        
                        # Show property data
                        display_cols = ['osm_id', 'building_type', 'latitude', 'longitude']
                        if 'address' in properties_df.columns:
                            # Flatten address data for display
                            address_df = properties_df['address'].apply(pd.Series)
                            properties_df['street_address'] = address_df['street'].fillna('') + ' ' + address_df['housenumber'].fillna('')
                            display_cols.append('street_address')
                        
                        st.dataframe(properties_df[display_cols].head(100))
            else:
                st.error("Failed to create map visualization")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Data Sources:**
    - Neighborhood ROI data from Zillow
    - Property data from OpenStreetMap
    - Geocoding via Nominatim
    """)

if __name__ == "__main__":
    main()
