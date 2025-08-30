#!/usr/bin/env python3
"""
Enhanced Real Estate Data Sources Integration
Comprehensive free data sources for real estate analysis:
1. OSMnx + Overpass API (OpenStreetMap)
2. Census/ACS Data (American Community Survey)
3. Geopandas + Shapely
4. Contextily + Folium
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union
import folium
import contextily as ctx
import cenpy
from geopy.distance import geodesic
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class AmenityData:
    """Structured amenity data container"""
    name: str
    amenity_type: str
    latitude: float
    longitude: float
    distance_miles: float
    rating: Optional[float] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None

@dataclass
class CensusData:
    """Structured census data container"""
    geography: str
    total_population: int
    median_household_income: float
    median_home_value: float
    median_rent: float
    education_bachelors_plus: float
    employment_rate: float
    poverty_rate: float
    median_age: float
    household_size: float

@dataclass
class PropertyBoundary:
    """Structured property boundary data"""
    property_id: str
    geometry: Polygon
    area_sqft: float
    address: str
    building_type: str
    year_built: Optional[int] = None
    stories: Optional[int] = None

class EnhancedRealEstateDataFetcher:
    """Comprehensive real estate data fetcher using free sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Enhanced-Real-Estate-Analysis/1.0'
        })
        
        # Configure OSMnx
        ox.config(use_cache=True, log_console=False)
        
        # Initialize Census API
        try:
            self.census_api = cenpy.products.ACS(2022)
        except Exception as e:
            logger.warning(f"Failed to initialize Census API: {e}")
            self.census_api = None
    
    def get_neighborhood_amenities(self, 
                                 center_lat: float, 
                                 center_lon: float, 
                                 radius_miles: float = 1.0,
                                 boundary_polygon: Optional[Polygon] = None) -> Dict[str, List[AmenityData]]:
        """
        Get neighborhood amenities using OSMnx and Overpass API
        Returns amenities categorized by type
        
        Args:
            center_lat: Center latitude (fallback if no boundary)
            center_lon: Center longitude (fallback if no boundary)
            radius_miles: Radius in miles (fallback if no boundary)
            boundary_polygon: Precise neighborhood boundary polygon (preferred)
        """
        if boundary_polygon:
            logger.info(f"Fetching amenities within defined neighborhood boundary")
            # Use the precise boundary polygon
            bbox = boundary_polygon.bounds  # Get bounding box
            center_lat = (bbox[1] + bbox[3]) / 2  # Average of min/max lat
            center_lon = (bbox[0] + bbox[2]) / 2  # Average of min/max lon
            # Calculate radius from boundary for fallback
            radius_miles = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 69  # Convert degrees to miles
        else:
            logger.info(f"Fetching amenities within {radius_miles} miles of ({center_lat}, {center_lon})")
        
        try:
            # Convert miles to meters
            radius_meters = radius_miles * 1609.34
            
            # Define amenity types to search for
            amenity_types = {
                'education': ['school', 'university', 'college', 'kindergarten'],
                'healthcare': ['hospital', 'clinic', 'pharmacy', 'doctor'],
                'shopping': ['supermarket', 'mall', 'shop', 'convenience'],
                'dining': ['restaurant', 'cafe', 'bar', 'fast_food'],
                'recreation': ['park', 'playground', 'sports_centre', 'gym'],
                'transport': ['bus_station', 'subway_station', 'train_station', 'parking'],
                'services': ['bank', 'post_office', 'library', 'police']
            }
            
            amenities_by_type = {}
            
            for category, types in amenity_types.items():
                category_amenities = []
                
                for amenity_type in types:
                    try:
                        # Use OSMnx to get amenities
                        amenities = ox.geometries_from_point(
                            (center_lat, center_lon),
                            tags={'amenity': amenity_type},
                            dist=radius_meters
                        )
                        
                        if not amenities.empty:
                            for idx, amenity in amenities.iterrows():
                                # Calculate distance
                                amenity_point = Point(amenity.geometry.x, amenity.geometry.y)
                                center_point = Point(center_lon, center_lat)
                                distance_miles = center_point.distance(amenity_point) * 69  # Rough conversion
                                
                                amenity_data = AmenityData(
                                    name=amenity.get('name', f'{amenity_type.title()}'),
                                    amenity_type=amenity_type,
                                    latitude=amenity.geometry.y,
                                    longitude=amenity.geometry.x,
                                    distance_miles=distance_miles,
                                    address=amenity.get('addr:street', None),
                                    phone=amenity.get('phone', None),
                                    website=amenity.get('website', None)
                                )
                                category_amenities.append(amenity_data)
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch {amenity_type} amenities: {e}")
                        continue
                
                amenities_by_type[category] = category_amenities
            
            logger.info(f"Found {sum(len(amenities) for amenities in amenities_by_type.values())} total amenities")
            return amenities_by_type
            
        except Exception as e:
            logger.error(f"Error fetching amenities: {e}")
            return {}
    
    def calculate_walkability_score(self, 
                                 center_lat: float, 
                                 center_lon: float, 
                                 radius_miles: float = 0.5) -> Dict[str, float]:
        """
        Calculate walkability score based on nearby amenities
        Returns scores for different categories
        """
        logger.info(f"Calculating walkability score for ({center_lat}, {center_lon})")
        
        try:
            amenities = self.get_neighborhood_amenities(center_lat, center_lon, radius_miles)
            
            # Define scoring weights
            amenity_weights = {
                'education': 0.15,
                'healthcare': 0.15,
                'shopping': 0.20,
                'dining': 0.20,
                'recreation': 0.10,
                'transport': 0.15,
                'services': 0.05
            }
            
            scores = {}
            total_score = 0
            
            for category, weight in amenity_weights.items():
                category_amenities = amenities.get(category, [])
                
                if category_amenities:
                    # Calculate score based on number and proximity of amenities
                    proximity_scores = []
                    for amenity in category_amenities:
                        if amenity.distance_miles <= 0.25:  # Within 1/4 mile
                            proximity_scores.append(100)
                        elif amenity.distance_miles <= 0.5:  # Within 1/2 mile
                            proximity_scores.append(75)
                        elif amenity.distance_miles <= 1.0:  # Within 1 mile
                            proximity_scores.append(50)
                        else:
                            proximity_scores.append(25)
                    
                    category_score = np.mean(proximity_scores) if proximity_scores else 0
                    scores[f'{category}_score'] = category_score
                    total_score += category_score * weight
                else:
                    scores[f'{category}_score'] = 0
            
            scores['overall_walkability'] = total_score
            scores['amenity_count'] = sum(len(amenities.get(cat, [])) for cat in amenity_weights.keys())
            
            logger.info(f"Walkability score: {total_score:.1f}/100")
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating walkability score: {e}")
            return {'overall_walkability': 0, 'amenity_count': 0}
    
    def get_census_data(self, 
                       state: str, 
                       county: str, 
                       tract: Optional[str] = None) -> Dict[str, CensusData]:
        """
        Get Census/ACS data for demographic and housing information
        """
        logger.info(f"Fetching Census data for {county}, {state}")
        
        try:
            if not self.census_api:
                logger.warning("Census API not available")
                return {}
            
            # Define variables to fetch
            variables = {
                'B01003_001E': 'total_population',
                'B19013_001E': 'median_household_income',
                'B25077_001E': 'median_home_value',
                'B25064_001E': 'median_rent',
                'B15003_022E': 'education_bachelors',
                'B15003_023E': 'education_masters',
                'B15003_024E': 'education_professional',
                'B15003_025E': 'education_doctorate',
                'B23025_002E': 'employed',
                'B23025_003E': 'unemployed',
                'B17001_002E': 'below_poverty',
                'B01002_001E': 'median_age',
                'B25010_001E': 'household_size'
            }
            
            # Fetch data
            if tract:
                # Tract-level data
                data = self.census_api.from_state(
                    state=state,
                    level='tract',
                    variables=list(variables.keys())
                )
                data = data[data['county'] == county]
                data = data[data['tract'] == tract]
            else:
                # County-level data
                data = self.census_api.from_state(
                    state=state,
                    level='county',
                    variables=list(variables.keys())
                )
                data = data[data['NAME'].str.contains(county, case=False)]
            
            if data.empty:
                logger.warning(f"No Census data found for {county}, {state}")
                return {}
            
            # Process the data
            census_data = {}
            for idx, row in data.iterrows():
                geography = row.get('NAME', f'{county}, {state}')
                
                # Calculate derived metrics
                total_pop = row.get('B01003_001E', 0)
                bachelors_plus = sum([
                    row.get('B15003_022E', 0),  # Bachelor's
                    row.get('B15003_023E', 0),  # Master's
                    row.get('B15003_024E', 0),  # Professional
                    row.get('B15003_025E', 0)   # Doctorate
                ])
                
                employed = row.get('B23025_002E', 0)
                unemployed = row.get('B23025_003E', 0)
                total_labor_force = employed + unemployed
                employment_rate = (employed / total_labor_force * 100) if total_labor_force > 0 else 0
                
                below_poverty = row.get('B17001_002E', 0)
                poverty_rate = (below_poverty / total_pop * 100) if total_pop > 0 else 0
                
                education_rate = (bachelors_plus / total_pop * 100) if total_pop > 0 else 0
                
                census_info = CensusData(
                    geography=geography,
                    total_population=total_pop,
                    median_household_income=row.get('B19013_001E', 0),
                    median_home_value=row.get('B25077_001E', 0),
                    median_rent=row.get('B25064_001E', 0),
                    education_bachelors_plus=education_rate,
                    employment_rate=employment_rate,
                    poverty_rate=poverty_rate,
                    median_age=row.get('B01002_001E', 0),
                    household_size=row.get('B25010_001E', 0)
                )
                
                census_data[geography] = census_info
            
            logger.info(f"Retrieved Census data for {len(census_data)} geographies")
            return census_data
            
        except Exception as e:
            logger.error(f"Error fetching Census data: {e}")
            return {}
    
    def get_property_boundaries(self, 
                               center_lat: float, 
                               center_lon: float, 
                               radius_miles: float = 0.5) -> List[PropertyBoundary]:
        """
        Get property boundaries using OSMnx and Overpass API
        """
        logger.info(f"Fetching property boundaries within {radius_miles} miles")
        
        try:
            # Convert miles to meters
            radius_meters = radius_miles * 1609.34
            
            # Get buildings
            buildings = ox.geometries_from_point(
                (center_lat, center_lon),
                tags={'building': True},
                dist=radius_meters
            )
            
            if buildings.empty:
                logger.warning("No buildings found in the specified area")
                return []
            
            property_boundaries = []
            
            for idx, building in buildings.iterrows():
                try:
                    # Extract building information
                    building_type = building.get('building', 'unknown')
                    address = building.get('addr:housenumber', '') + ' ' + building.get('addr:street', '')
                    year_built = building.get('start_date', None)
                    if year_built:
                        try:
                            year_built = int(str(year_built)[:4])
                        except:
                            year_built = None
                    
                    stories = building.get('building:levels', None)
                    if stories:
                        try:
                            stories = int(stories)
                        except:
                            stories = None
                    
                    # Calculate area
                    if hasattr(building.geometry, 'area'):
                        area_sqft = building.geometry.area * 10.764  # Convert sq meters to sq feet
                    else:
                        area_sqft = 0
                    
                    property_boundary = PropertyBoundary(
                        property_id=str(idx),
                        geometry=building.geometry,
                        area_sqft=area_sqft,
                        address=address.strip() if address.strip() else f"Building {idx}",
                        building_type=building_type,
                        year_built=year_built,
                        stories=stories
                    )
                    
                    property_boundaries.append(property_boundary)
                    
                except Exception as e:
                    logger.warning(f"Error processing building {idx}: {e}")
                    continue
            
            logger.info(f"Retrieved {len(property_boundaries)} property boundaries")
            return property_boundaries
            
        except Exception as e:
            logger.error(f"Error fetching property boundaries: {e}")
            return []
    
    def get_neighborhood_boundary(self, 
                                neighborhood_name: str, 
                                city: str, 
                                state: str) -> Optional[Polygon]:
        """
        Get precise neighborhood boundary from OpenStreetMap
        Similar to Zillow's neighborhood definitions
        """
        logger.info(f"Fetching neighborhood boundary for {neighborhood_name}, {city}, {state}")
        
        try:
            # Search for neighborhood boundary using OSM
            # Look for administrative boundaries and named places
            query = f"{neighborhood_name}, {city}, {state}, USA"
            
            # Try to get neighborhood boundary from OSM
            # This looks for administrative boundaries, named places, and residential areas
            boundary_tags = [
                {'boundary': 'administrative', 'admin_level': '10'},  # Neighborhood level
                {'place': 'neighbourhood'},  # Named neighborhoods
                {'landuse': 'residential'},  # Residential areas
                {'name': neighborhood_name}  # Exact name match
            ]
            
            for tags in boundary_tags:
                try:
                    # Search for boundaries with these tags
                    boundaries = ox.geometries_from_place(
                        query,
                        tags=tags
                    )
                    
                    if not boundaries.empty:
                        # Find the boundary that best matches our neighborhood
                        for idx, boundary in boundaries.iterrows():
                            if boundary.geometry and hasattr(boundary.geometry, 'area'):
                                # Check if this is a reasonable size for a neighborhood
                                area_sq_miles = boundary.geometry.area * 0.000000386102  # Convert sq meters to sq miles
                                
                                # Neighborhoods are typically 0.1 to 5 square miles
                                if 0.1 <= area_sq_miles <= 5.0:
                                    logger.info(f"Found neighborhood boundary: {area_sq_miles:.2f} sq miles")
                                    return boundary.geometry
                        
                        # If no reasonable size found, use the first boundary
                        if boundaries.iloc[0].geometry:
                            logger.info("Using first available boundary")
                            return boundaries.iloc[0].geometry
                
                except Exception as e:
                    logger.warning(f"Failed to fetch boundary with tags {tags}: {e}")
                    continue
            
            # Fallback: create a rough boundary using geocoding
            logger.info("Creating fallback boundary using geocoding")
            try:
                # Get the center point of the neighborhood
                geocoder_result = ox.geocoder.geocode(query)
                if geocoder_result:
                    lat, lon = geocoder_result
                    # Create a rough circular boundary (0.5 mile radius)
                    from shapely.geometry import Point
                    center_point = Point(lon, lat)
                    rough_boundary = center_point.buffer(0.008)  # Roughly 0.5 miles
                    logger.info("Created fallback circular boundary")
                    return rough_boundary
            except Exception as e:
                logger.warning(f"Fallback boundary creation failed: {e}")
            
            logger.warning(f"No boundary found for {neighborhood_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching neighborhood boundary: {e}")
            return None
    
    def get_street_network(self, 
                          center_lat: float, 
                          center_lon: float, 
                          radius_miles: float = 1.0) -> gpd.GeoDataFrame:
        """
        Get street network for accessibility analysis
        """
        logger.info(f"Fetching street network within {radius_miles} miles")
        
        try:
            # Convert miles to meters
            radius_meters = radius_miles * 1609.34
            
            # Get street network
            G = ox.graph_from_point(
                (center_lat, center_lon),
                dist=radius_meters,
                network_type='drive'
            )
            
            # Convert to GeoDataFrame
            edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            
            logger.info(f"Retrieved street network with {len(edges)} edges")
            return edges
            
        except Exception as e:
            logger.error(f"Error fetching street network: {e}")
            return gpd.GeoDataFrame()
    
    def calculate_transit_score(self, 
                              center_lat: float, 
                              center_lon: float, 
                              radius_miles: float = 1.0) -> float:
        """
        Calculate transit accessibility score
        """
        logger.info(f"Calculating transit score for ({center_lat}, {center_lon})")
        
        try:
            # Get transit amenities
            transit_amenities = self.get_neighborhood_amenities(
                center_lat, center_lon, radius_miles
            ).get('transport', [])
            
            if not transit_amenities:
                return 0
            
            # Calculate score based on proximity and variety
            proximity_scores = []
            for amenity in transit_amenities:
                if amenity.distance_miles <= 0.25:
                    proximity_scores.append(100)
                elif amenity.distance_miles <= 0.5:
                    proximity_scores.append(75)
                elif amenity.distance_miles <= 1.0:
                    proximity_scores.append(50)
                else:
                    proximity_scores.append(25)
            
            # Bonus for variety of transit options
            transit_types = set(amenity.amenity_type for amenity in transit_amenities)
            variety_bonus = min(len(transit_types) * 10, 30)  # Max 30 point bonus
            
            base_score = np.mean(proximity_scores) if proximity_scores else 0
            total_score = min(base_score + variety_bonus, 100)
            
            logger.info(f"Transit score: {total_score:.1f}/100")
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating transit score: {e}")
            return 0
    
    def create_interactive_map(self, 
                             center_lat: float, 
                             center_lon: float, 
                             amenities: Dict[str, List[AmenityData]] = None,
                             property_boundaries: List[PropertyBoundary] = None,
                             census_data: Dict[str, CensusData] = None) -> folium.Map:
        """
        Create an interactive map with all data layers
        """
        logger.info("Creating interactive map with data layers")
        
        try:
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # Add satellite layer
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satellite',
                overlay=False
            ).add_to(m)
            
            # Add amenities if provided
            if amenities:
                for category, category_amenities in amenities.items():
                    if category_amenities:
                        # Define colors for different amenity types
                        colors = {
                            'education': 'blue',
                            'healthcare': 'red',
                            'shopping': 'green',
                            'dining': 'orange',
                            'recreation': 'purple',
                            'transport': 'darkblue',
                            'services': 'darkgreen'
                        }
                        
                        color = colors.get(category, 'gray')
                        
                        for amenity in category_amenities:
                            popup_text = f"""
                            <b>{amenity.name}</b><br>
                            Type: {amenity.amenity_type}<br>
                            Distance: {amenity.distance_miles:.2f} miles<br>
                            """
                            if amenity.address:
                                popup_text += f"Address: {amenity.address}<br>"
                            if amenity.phone:
                                popup_text += f"Phone: {amenity.phone}<br>"
                            
                            folium.Marker(
                                [amenity.latitude, amenity.longitude],
                                popup=popup_text,
                                tooltip=f"{amenity.name} ({amenity.distance_miles:.2f} mi)",
                                icon=folium.Icon(color=color, icon='info-sign')
                            ).add_to(m)
            
            # Add property boundaries if provided
            if property_boundaries:
                for prop in property_boundaries:
                    if hasattr(prop.geometry, 'exterior'):
                        # Convert geometry to folium format
                        coords = list(prop.geometry.exterior.coords)
                        folium.Polygon(
                            locations=[[lat, lon] for lon, lat in coords],
                            popup=f"""
                            <b>Property Details</b><br>
                            Type: {prop.building_type}<br>
                            Area: {prop.area_sqft:.0f} sq ft<br>
                            Address: {prop.address}<br>
                            """,
                            tooltip=f"{prop.address} ({prop.area_sqft:.0f} sq ft)",
                            color='black',
                            weight=2,
                            fill=True,
                            fillColor='lightblue',
                            fillOpacity=0.3
                        ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            logger.info("Interactive map created successfully")
            return m
            
        except Exception as e:
            logger.error(f"Error creating interactive map: {e}")
            return None
    
    def get_comprehensive_osm_data_within_boundary(self, 
                                                 boundary_polygon: Polygon,
                                                 neighborhood_name: str = "Unknown") -> Dict[str, Any]:
        """
        Get comprehensive OSM data within a specific neighborhood boundary
        This loads much more detailed data within the precise area
        """
        logger.info(f"Loading comprehensive OSM data within {neighborhood_name} boundary")
        
        try:
            # Get the bounding box of the boundary
            bbox = boundary_polygon.bounds
            min_lon, min_lat, max_lon, max_lat = bbox
            
            # Calculate center for fallback operations
            center_lat = (min_lat + max_lat) / 2
            center_lon = (min_lon + max_lon) / 2
            
            comprehensive_data = {
                'boundary': boundary_polygon,
                'bbox': bbox,
                'center': (center_lat, center_lon),
                'amenities': {},
                'buildings': [],
                'streets': [],
                'landuse': [],
                'natural_features': [],
                'transport': [],
                'summary': {}
            }
            
            # 1. Get ALL buildings within the boundary
            logger.info("Loading buildings within boundary...")
            try:
                buildings = ox.geometries_from_bbox(
                    north=max_lat, south=min_lat, 
                    east=max_lon, west=min_lon,
                    tags={'building': True}
                )
                
                if not buildings.empty:
                    for idx, building in buildings.iterrows():
                        if building.geometry and boundary_polygon.contains(building.geometry):
                            building_data = {
                                'osm_id': idx,
                                'building_type': building.get('building', 'unknown'),
                                'geometry': building.geometry,
                                'area_sqft': building.geometry.area * 10.764 if hasattr(building.geometry, 'area') else 0,
                                'address': {
                                    'housenumber': building.get('addr:housenumber', ''),
                                    'street': building.get('addr:street', ''),
                                    'city': building.get('addr:city', ''),
                                    'state': building.get('addr:state', ''),
                                    'postcode': building.get('addr:postcode', '')
                                },
                                'year_built': building.get('start_date', None),
                                'stories': building.get('building:levels', None),
                                'roof_type': building.get('roof:type', None),
                                'material': building.get('building:material', None)
                            }
                            comprehensive_data['buildings'].append(building_data)
                
                logger.info(f"Loaded {len(comprehensive_data['buildings'])} buildings")
                
            except Exception as e:
                logger.warning(f"Failed to load buildings: {e}")
            
            # 2. Get ALL amenities within the boundary
            logger.info("Loading amenities within boundary...")
            amenity_types = {
                'education': ['school', 'university', 'college', 'kindergarten', 'library'],
                'healthcare': ['hospital', 'clinic', 'pharmacy', 'doctor', 'dentist', 'veterinary'],
                'shopping': ['supermarket', 'mall', 'shop', 'convenience', 'department_store', 'clothes'],
                'dining': ['restaurant', 'cafe', 'bar', 'fast_food', 'bakery', 'ice_cream'],
                'recreation': ['park', 'playground', 'sports_centre', 'gym', 'swimming_pool', 'tennis_court'],
                'transport': ['bus_station', 'subway_station', 'train_station', 'parking', 'taxi', 'bicycle_parking'],
                'services': ['bank', 'post_office', 'police', 'fire_station', 'townhall', 'courthouse'],
                'entertainment': ['cinema', 'theatre', 'museum', 'gallery', 'bowling_alley', 'casino']
            }
            
            for category, types in amenity_types.items():
                category_amenities = []
                
                for amenity_type in types:
                    try:
                        amenities = ox.geometries_from_bbox(
                            north=max_lat, south=min_lat, 
                            east=max_lon, west=min_lon,
                            tags={'amenity': amenity_type}
                        )
                        
                        if not amenities.empty:
                            for idx, amenity in amenities.iterrows():
                                if amenity.geometry and boundary_polygon.contains(amenity.geometry):
                                    amenity_data = AmenityData(
                                        name=amenity.get('name', f'{amenity_type.title()}'),
                                        amenity_type=amenity_type,
                                        latitude=amenity.geometry.y,
                                        longitude=amenity.geometry.x,
                                        distance_miles=0,  # Within boundary
                                        address=amenity.get('addr:street', None),
                                        phone=amenity.get('phone', None),
                                        website=amenity.get('website', None)
                                    )
                                    category_amenities.append(amenity_data)
                        
                        time.sleep(0.05)  # Rate limiting
                        
                    except Exception as e:
                        logger.warning(f"Failed to fetch {amenity_type} amenities: {e}")
                        continue
                
                comprehensive_data['amenities'][category] = category_amenities
            
            # 3. Get street network within boundary
            logger.info("Loading street network within boundary...")
            try:
                streets = ox.geometries_from_bbox(
                    north=max_lat, south=min_lat, 
                    east=max_lon, west=min_lon,
                    tags={'highway': True}
                )
                
                if not streets.empty:
                    for idx, street in streets.iterrows():
                        if street.geometry and boundary_polygon.intersects(street.geometry):
                            street_data = {
                                'osm_id': idx,
                                'name': street.get('name', 'Unnamed Street'),
                                'highway_type': street.get('highway', 'unknown'),
                                'geometry': street.geometry,
                                'lanes': street.get('lanes', None),
                                'surface': street.get('surface', None),
                                'speed_limit': street.get('maxspeed', None)
                            }
                            comprehensive_data['streets'].append(street_data)
                
                logger.info(f"Loaded {len(comprehensive_data['streets'])} street segments")
                
            except Exception as e:
                logger.warning(f"Failed to load streets: {e}")
            
            # 4. Get land use and natural features
            logger.info("Loading land use and natural features...")
            try:
                landuse = ox.geometries_from_bbox(
                    north=max_lat, south=min_lat, 
                    east=max_lon, west=min_lon,
                    tags={'landuse': True}
                )
                
                if not landuse.empty:
                    for idx, feature in landuse.iterrows():
                        if feature.geometry and boundary_polygon.intersects(feature.geometry):
                            landuse_data = {
                                'osm_id': idx,
                                'landuse_type': feature.get('landuse', 'unknown'),
                                'geometry': feature.geometry,
                                'name': feature.get('name', None)
                            }
                            comprehensive_data['landuse'].append(landuse_data)
                
                # Natural features
                natural = ox.geometries_from_bbox(
                    north=max_lat, south=min_lat, 
                    east=max_lon, west=min_lon,
                    tags={'natural': True}
                )
                
                if not natural.empty:
                    for idx, feature in natural.iterrows():
                        if feature.geometry and boundary_polygon.intersects(feature.geometry):
                            natural_data = {
                                'osm_id': idx,
                                'natural_type': feature.get('natural', 'unknown'),
                                'geometry': feature.geometry,
                                'name': feature.get('name', None)
                            }
                            comprehensive_data['natural_features'].append(natural_data)
                
                logger.info(f"Loaded {len(comprehensive_data['landuse'])} land use areas and {len(comprehensive_data['natural_features'])} natural features")
                
            except Exception as e:
                logger.warning(f"Failed to load land use/natural features: {e}")
            
            # 5. Calculate summary statistics
            total_buildings = len(comprehensive_data['buildings'])
            total_amenities = sum(len(amenities) for amenities in comprehensive_data['amenities'].values())
            total_streets = len(comprehensive_data['streets'])
            
            comprehensive_data['summary'] = {
                'total_buildings': total_buildings,
                'total_amenities': total_amenities,
                'total_streets': total_streets,
                'boundary_area_sq_miles': boundary_polygon.area * 0.000000386102,
                'building_density': total_buildings / (boundary_polygon.area * 0.000000386102) if boundary_polygon.area > 0 else 0,
                'amenity_density': total_amenities / (boundary_polygon.area * 0.000000386102) if boundary_polygon.area > 0 else 0
            }
            
            logger.info(f"Comprehensive OSM data loaded successfully for {neighborhood_name}")
            logger.info(f"Summary: {total_buildings} buildings, {total_amenities} amenities, {total_streets} streets")
            
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error loading comprehensive OSM data: {e}")
            return {}
    
    def get_comprehensive_neighborhood_data(self, 
                                          center_lat: float, 
                                          center_lon: float, 
                                          state: str, 
                                          county: str) -> Dict[str, Any]:
        """
        Get comprehensive neighborhood data from all sources
        """
        logger.info(f"Getting comprehensive data for neighborhood at ({center_lat}, {center_lon})")
        
        try:
            # Get all data sources
            amenities = self.get_neighborhood_amenities(center_lat, center_lon)
            walkability_scores = self.calculate_walkability_score(center_lat, center_lon)
            transit_score = self.calculate_transit_score(center_lat, center_lon)
            property_boundaries = self.get_property_boundaries(center_lat, center_lon)
            census_data = self.get_census_data(state, county)
            street_network = self.get_street_network(center_lat, center_lon)
            
            # Compile comprehensive data
            comprehensive_data = {
                'location': {
                    'latitude': center_lat,
                    'longitude': center_lon,
                    'state': state,
                    'county': county
                },
                'amenities': amenities,
                'scores': {
                    'walkability': walkability_scores,
                    'transit': transit_score,
                    'overall_accessibility': (walkability_scores.get('overall_walkability', 0) + transit_score) / 2
                },
                'property_boundaries': property_boundaries,
                'census_data': census_data,
                'street_network': street_network,
                'summary': {
                    'total_amenities': sum(len(amenities.get(cat, [])) for cat in amenities.keys()),
                    'total_properties': len(property_boundaries),
                    'walkability_score': walkability_scores.get('overall_walkability', 0),
                    'transit_score': transit_score
                }
            }
            
            logger.info(f"Comprehensive data compiled successfully")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error getting comprehensive neighborhood data: {e}")
            return {}
    
    def get_neighborhood_with_comprehensive_data(self, 
                                               neighborhood_name: str, 
                                               city: str, 
                                               state: str) -> Dict[str, Any]:
        """
        Get neighborhood boundary and comprehensive OSM data
        This is the main method to use for Zillow-like neighborhood analysis
        """
        logger.info(f"Getting comprehensive data for {neighborhood_name}, {city}, {state}")
        
        try:
            # Step 1: Get the neighborhood boundary
            boundary = self.get_neighborhood_boundary(neighborhood_name, city, state)
            
            if not boundary:
                logger.warning(f"No boundary found for {neighborhood_name}, using fallback")
                return {}
            
            # Step 2: Load comprehensive OSM data within the boundary
            comprehensive_data = self.get_comprehensive_osm_data_within_boundary(
                boundary, neighborhood_name
            )
            
            # Step 3: Add boundary information
            comprehensive_data['neighborhood_info'] = {
                'name': neighborhood_name,
                'city': city,
                'state': state,
                'boundary_polygon': boundary,
                'boundary_area_sq_miles': boundary.area * 0.000000386102
            }
            
            logger.info(f"Successfully loaded comprehensive data for {neighborhood_name}")
            return comprehensive_data
            
        except Exception as e:
            logger.error(f"Error getting neighborhood with comprehensive data: {e}")
            return {}
    
    # Example usage and testing
    if __name__ == "__main__":
        # Test the enhanced data fetcher
        fetcher = EnhancedRealEstateDataFetcher()
        
        # Test coordinates (San Francisco)
        test_lat, test_lon = 37.7749, -122.4194
        
        print("Testing Enhanced Real Estate Data Fetcher...")
        
        # Test amenities
        amenities = fetcher.get_neighborhood_amenities(test_lat, test_lon, 0.5)
        print(f"Found {sum(len(amenities.get(cat, [])) for cat in amenities.keys())} amenities")
        
        # Test walkability
        walkability = fetcher.calculate_walkability_score(test_lat, test_lon)
        print(f"Walkability score: {walkability.get('overall_walkability', 0):.1f}/100")
        
        # Test transit
        transit = fetcher.calculate_transit_score(test_lat, test_lon)
        print(f"Transit score: {transit:.1f}/100")
        
        # Test comprehensive data
        comprehensive = fetcher.get_comprehensive_neighborhood_data(test_lat, test_lon, 'CA', 'San Francisco')
        print(f"Comprehensive data summary: {comprehensive.get('summary', {})}")
        
        # Test new neighborhood boundary functionality
        print("\nTesting Neighborhood Boundary Functionality...")
        neighborhood_data = fetcher.get_neighborhood_with_comprehensive_data("North Beach", "San Francisco", "CA")
        if neighborhood_data:
            print(f"✅ Successfully loaded data for North Beach neighborhood")
            print(f"   Buildings: {neighborhood_data.get('summary', {}).get('total_buildings', 0)}")
            print(f"   Amenities: {neighborhood_data.get('summary', {}).get('total_amenities', 0)}")
            print(f"   Streets: {neighborhood_data.get('summary', {}).get('total_streets', 0)}")
        else:
            print("❌ Failed to load neighborhood data")
