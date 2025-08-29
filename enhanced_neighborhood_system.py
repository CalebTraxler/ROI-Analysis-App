#!/usr/bin/env python3
"""
Enhanced Neighborhood System
Handles neighborhood boundary definition, continuous heat mapping, and house clustering
"""

import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union, voronoi_polygons
from shapely.validation import make_valid
import pydeck as pdk
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeighborhoodBoundaryManager:
    """Manages neighborhood boundaries and continuous coverage"""
    
    def __init__(self):
        self.neighborhood_boundaries = {}
        self.neighborhood_centroids = {}
        self.voronoi_polygons = {}
        
    def define_neighborhood_boundaries(self, 
                                    neighborhood_data: pd.DataFrame,
                                    buffer_miles: float = 0.5) -> Dict[str, Polygon]:
        """
        Define neighborhood boundaries using multiple methods:
        1. Voronoi tessellation for continuous coverage
        2. Buffer around neighborhood centroids
        3. Natural boundary detection
        """
        logger.info("Defining neighborhood boundaries for continuous coverage")
        
        try:
            # Get valid coordinates
            valid_data = neighborhood_data.dropna(subset=['Latitude', 'Longitude'])
            
            if len(valid_data) == 0:
                logger.warning("No valid coordinates for boundary definition")
                return {}
            
            # Method 1: Voronoi tessellation for continuous coverage
            self._create_voronoi_boundaries(valid_data)
            
            # Method 2: Enhanced buffer boundaries with overlap
            self._create_buffer_boundaries(valid_data, buffer_miles)
            
            # Method 3: Natural boundary detection using OSM data
            self._detect_natural_boundaries(valid_data)
            
            # Combine all methods for optimal coverage
            self._combine_boundary_methods()
            
            logger.info(f"Created boundaries for {len(self.neighborhood_boundaries)} neighborhoods")
            return self.neighborhood_boundaries
            
        except Exception as e:
            logger.error(f"Error defining neighborhood boundaries: {e}")
            return {}
    
    def _create_voronoi_boundaries(self, data: pd.DataFrame):
        """Create Voronoi polygons for continuous neighborhood coverage"""
        try:
            # Extract coordinates
            coords = np.array([[row['Longitude'], row['Latitude']] for _, row in data.iterrows()])
            
            # Create Voronoi diagram
            vor = Voronoi(coords)
            
            # Convert to polygons
            for i, (_, row) in enumerate(data.iterrows()):
                neighborhood_name = row['RegionName']
                
                # Get Voronoi region for this point
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                
                if -1 not in region:  # Skip unbounded regions
                    # Get vertices for this region
                    vertices = [vor.vertices[j] for j in region]
                    
                    if len(vertices) >= 3:
                        # Create polygon and buffer slightly for overlap
                        poly = Polygon(vertices)
                        buffered_poly = poly.buffer(0.001)  # Small buffer for overlap
                        
                        self.voronoi_polygons[neighborhood_name] = buffered_poly
                        
        except Exception as e:
            logger.warning(f"Voronoi boundary creation failed: {e}")
    
    def _create_buffer_boundaries(self, data: pd.DataFrame, buffer_miles: float):
        """Create buffer boundaries around neighborhood centroids"""
        try:
            # Convert miles to degrees (approximate)
            buffer_degrees = buffer_miles / 69.0
            
            for _, row in data.iterrows():
                neighborhood_name = row['RegionName']
                lat, lon = row['Latitude'], row['Longitude']
                
                # Create buffer around centroid
                centroid = Point(lon, lat)
                buffer_poly = centroid.buffer(buffer_degrees)
                
                # Store buffer boundary
                if neighborhood_name not in self.neighborhood_boundaries:
                    self.neighborhood_boundaries[neighborhood_name] = []
                self.neighborhood_boundaries[neighborhood_name].append(buffer_poly)
                
        except Exception as e:
            logger.warning(f"Buffer boundary creation failed: {e}")
    
    def _detect_natural_boundaries(self, data: pd.DataFrame):
        """Detect natural boundaries using OSM data"""
        try:
            # Get center point for the entire area
            center_lat = data['Latitude'].mean()
            center_lon = data['Longitude'].mean()
            
            # Search for natural boundaries (rivers, major roads, etc.)
            natural_boundaries = ox.geometries_from_point(
                (center_lat, center_lon),
                tags={'natural': ['water', 'waterway']},
                dist=5000  # 5km radius
            )
            
            # Also get major roads
            major_roads = ox.geometries_from_point(
                (center_lat, center_lon),
                tags={'highway': ['motorway', 'trunk', 'primary']},
                dist=5000
            )
            
            # Store natural boundaries for later use
            self.natural_boundaries = {
                'water': natural_boundaries,
                'roads': major_roads
            }
            
        except Exception as e:
            logger.warning(f"Natural boundary detection failed: {e}")
    
    def _combine_boundary_methods(self):
        """Combine all boundary methods for optimal coverage"""
        try:
            for neighborhood_name in self.voronoi_polygons.keys():
                if neighborhood_name in self.neighborhood_boundaries:
                    # Combine Voronoi with buffer boundaries
                    combined_poly = unary_union([
                        self.voronoi_polygons[neighborhood_name]
                    ] + self.neighborhood_boundaries[neighborhood_name])
                    
                    # Clean up geometry
                    if combined_poly.is_valid:
                        self.neighborhood_boundaries[neighborhood_name] = combined_poly
                    else:
                        self.neighborhood_boundaries[neighborhood_name] = make_valid(combined_poly)
                else:
                    # Use Voronoi only
                    self.neighborhood_boundaries[neighborhood_name] = self.voronoi_polygons[neighborhood_name]
                    
        except Exception as e:
            logger.warning(f"Boundary combination failed: {e}")
    
    def get_neighborhood_houses(self, 
                              neighborhood_name: str,
                              center_lat: float,
                              center_lon: float,
                              radius_miles: float = 1.0) -> List[Dict]:
        """
        Load houses within a specific neighborhood boundary
        """
        logger.info(f"Loading houses for neighborhood: {neighborhood_name}")
        
        try:
            # Get neighborhood boundary
            if neighborhood_name not in self.neighborhood_boundaries:
                logger.warning(f"No boundary defined for {neighborhood_name}")
                return []
            
            boundary = self.neighborhood_boundaries[neighborhood_name]
            
            # Convert miles to meters
            radius_meters = radius_miles * 1609.34
            
            # Get buildings within the boundary area
            buildings = ox.geometries_from_point(
                (center_lat, center_lon),
                tags={'building': True},
                dist=radius_meters
            )
            
            if buildings.empty:
                logger.warning(f"No buildings found in {neighborhood_name}")
                return []
            
            # Filter buildings to only those within the neighborhood boundary
            houses = []
            for idx, building in buildings.iterrows():
                try:
                    # Check if building is within neighborhood boundary
                    if boundary.contains(building.geometry):
                        house_data = self._extract_house_data(building, idx)
                        houses.append(house_data)
                except Exception as e:
                    logger.warning(f"Error processing building {idx}: {e}")
                    continue
            
            logger.info(f"Found {len(houses)} houses in {neighborhood_name}")
            return houses
            
        except Exception as e:
            logger.error(f"Error loading neighborhood houses: {e}")
            return []
    
    def _extract_house_data(self, building, idx) -> Dict:
        """Extract comprehensive house data from OSM building"""
        try:
            # Basic building info
            building_type = building.get('building', 'unknown')
            address = building.get('addr:housenumber', '') + ' ' + building.get('addr:street', '')
            
            # Year built
            year_built = building.get('start_date', None)
            if year_built:
                try:
                    year_built = int(str(year_built)[:4])
                except:
                    year_built = None
            
            # Stories and height
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
            
            # Get coordinates
            if hasattr(building.geometry, 'centroid'):
                centroid = building.geometry.centroid
                lat, lon = centroid.y, centroid.x
            else:
                lat, lon = None, None
            
            return {
                'osm_id': idx,
                'building_type': building_type,
                'address': address.strip() if address.strip() else f"Building {idx}",
                'year_built': year_built,
                'stories': stories,
                'area_sqft': area_sqft,
                'latitude': lat,
                'longitude': lon,
                'geometry': building.geometry,
                'roof_type': building.get('roof:type', None),
                'material': building.get('building:material', None),
                'amenities': self._get_nearby_amenities(lat, lon) if lat and lon else {}
            }
            
        except Exception as e:
            logger.warning(f"Error extracting house data: {e}")
            return {}
    
    def _get_nearby_amenities(self, lat: float, lon: float, radius_miles: float = 0.5) -> Dict:
        """Get nearby amenities for a specific house"""
        try:
            # Convert miles to meters
            radius_meters = radius_miles * 1609.34
            
            # Get amenities
            amenities = ox.geometries_from_point(
                (lat, lon),
                tags={'amenity': True},
                dist=radius_meters
            )
            
            if amenities.empty:
                return {}
            
            # Categorize amenities
            amenity_categories = {}
            for _, amenity in amenities.iterrows():
                amenity_type = amenity.get('amenity', 'other')
                if amenity_type not in amenity_categories:
                    amenity_categories[amenity_type] = []
                
                amenity_categories[amenity_type].append({
                    'name': amenity.get('name', amenity_type),
                    'distance': self._calculate_distance(lat, lon, amenity.geometry.y, amenity.geometry.x)
                })
            
            return amenity_categories
            
        except Exception as e:
            logger.warning(f"Error getting nearby amenities: {e}")
            return {}
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in miles"""
        try:
            from geopy.distance import geodesic
            return geodesic((lat1, lon1), (lat2, lon2)).miles
        except:
            # Fallback calculation
            return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 69.0

class ContinuousHeatMapGenerator:
    """Generates continuous heat maps covering entire neighborhoods"""
    
    def __init__(self):
        self.heat_data = {}
        self.interpolation_grid = None
    
    def generate_continuous_heatmap(self, 
                                  neighborhood_data: pd.DataFrame,
                                  resolution: int = 100) -> np.ndarray:
        """
        Generate a continuous heat map covering all neighborhoods
        """
        logger.info("Generating continuous heat map")
        
        try:
            # Create interpolation grid
            self._create_interpolation_grid(neighborhood_data, resolution)
            
            # Interpolate ROI values across the grid
            self._interpolate_roi_values(neighborhood_data)
            
            # Apply smoothing for continuous coverage
            self._apply_smoothing()
            
            logger.info("Continuous heat map generated successfully")
            return self.interpolation_grid
            
        except Exception as e:
            logger.error(f"Error generating continuous heat map: {e}")
            return None
    
    def _create_interpolation_grid(self, data: pd.DataFrame, resolution: int):
        """Create a high-resolution grid for interpolation"""
        try:
            # Get bounds
            min_lat, max_lat = data['Latitude'].min(), data['Latitude'].max()
            min_lon, max_lon = data['Longitude'].min(), data['Longitude'].max()
            
            # Add buffer
            lat_buffer = (max_lat - min_lat) * 0.1
            lon_buffer = (max_lon - min_lon) * 0.1
            
            # Create grid
            lat_grid = np.linspace(min_lat - lat_buffer, max_lat + lat_buffer, resolution)
            lon_grid = np.linspace(min_lon - lon_buffer, max_lon + lon_buffer, resolution)
            
            self.lat_grid = lat_grid
            self.lon_grid = lon_grid
            self.interpolation_grid = np.zeros((resolution, resolution))
            
        except Exception as e:
            logger.warning(f"Grid creation failed: {e}")
    
    def _interpolate_roi_values(self, data: pd.DataFrame):
        """Interpolate ROI values across the grid"""
        try:
            from scipy.interpolate import griddata
            
            # Prepare data for interpolation
            points = np.array([[row['Longitude'], row['Latitude']] for _, row in data.iterrows()])
            values = np.array([row['ROI'] for _, row in data.iterrows()])
            
            # Create grid coordinates
            lon_mesh, lat_mesh = np.meshgrid(self.lon_grid, self.lat_grid)
            grid_coords = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
            
            # Interpolate
            interpolated = griddata(points, values, grid_coords, method='cubic', fill_value=np.nan)
            
            # Reshape to grid
            self.interpolation_grid = interpolated.reshape(self.interpolation_grid.shape)
            
        except Exception as e:
            logger.warning(f"ROI interpolation failed: {e}")
            # Fallback to nearest neighbor
            self._fallback_interpolation(data)
    
    def _fallback_interpolation(self, data: pd.DataFrame):
        """Fallback interpolation method"""
        try:
            for i, lat in enumerate(self.lat_grid):
                for j, lon in enumerate(self.lon_grid):
                    # Find nearest neighborhood
                    distances = []
                    for _, row in data.iterrows():
                        dist = np.sqrt((lat - row['Latitude'])**2 + (lon - row['Longitude'])**2)
                        distances.append(dist)
                    
                    nearest_idx = np.argmin(distances)
                    self.interpolation_grid[i, j] = data.iloc[nearest_idx]['ROI']
                    
        except Exception as e:
            logger.warning(f"Fallback interpolation failed: {e}")
    
    def _apply_smoothing(self):
        """Apply smoothing for continuous coverage"""
        try:
            from scipy.ndimage import gaussian_filter
            
            # Apply Gaussian smoothing
            self.interpolation_grid = gaussian_filter(self.interpolation_grid, sigma=1.0)
            
            # Fill any remaining NaN values
            self.interpolation_grid = np.nan_to_num(self.interpolation_grid, nan=0.0)
            
        except Exception as e:
            logger.warning(f"Smoothing failed: {e}")

class EnhancedHouseVisualizer:
    """Enhanced house visualization with better markers and clustering"""
    
    def __init__(self):
        self.house_styles = {}
        self.cluster_settings = {}
    
    def create_enhanced_house_layer(self, 
                                  houses: List[Dict],
                                  neighborhood_name: str) -> pdk.Layer:
        """
        Create enhanced house visualization layer
        """
        logger.info(f"Creating enhanced house layer for {neighborhood_name}")
        
        try:
            # Filter valid houses
            valid_houses = [h for h in houses if h.get('latitude') and h.get('longitude')]
            
            if not valid_houses:
                logger.warning(f"No valid houses for {neighborhood_name}")
                return None
            
            # Create house data with enhanced styling
            house_data = []
            for house in valid_houses:
                house_data.append({
                    'latitude': house['latitude'],
                    'longitude': house['longitude'],
                    'osm_id': house.get('osm_id', 'N/A'),
                    'building_type': house.get('building_type', 'unknown'),
                    'area_sqft': house.get('area_sqft', 0),
                    'year_built': house.get('year_built', 'N/A'),
                    'address': house.get('address', 'N/A'),
                    'roi_score': self._calculate_roi_score(house),
                    'size_category': self._categorize_house_size(house.get('area_sqft', 0)),
                    'age_category': self._categorize_house_age(house.get('year_built')),
                    'tooltip_text': self._create_house_tooltip(house)
                })
            
            # Create enhanced scatter layer
            layer = pdk.Layer(
                'ScatterplotLayer',
                house_data,
                get_position=['longitude', 'latitude'],
                get_radius='size_category',
                get_fill_color='roi_score',
                get_line_color=[255, 255, 255, 200],
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
                radius_scale=1,
                radius_min_pixels=8,
                radius_max_pixels=20
            )
            
            logger.info(f"Enhanced house layer created with {len(house_data)} houses")
            return layer
            
        except Exception as e:
            logger.error(f"Error creating enhanced house layer: {e}")
            return None
    
    def _calculate_roi_score(self, house: Dict) -> List[int]:
        """Calculate ROI-based color score for house"""
        try:
            # Base color (blue for houses)
            base_color = [59, 130, 246]
            
            # Adjust based on house characteristics
            area = house.get('area_sqft', 1000)
            year = house.get('year_built', 2000)
            
            # Size adjustment (larger houses get darker blue)
            size_factor = min(area / 2000, 1.0)
            color_adjustment = int(50 * size_factor)
            
            # Age adjustment (newer houses get brighter)
            if year and year > 2000:
                age_factor = 0.8
            else:
                age_factor = 1.0
            
            final_color = [
                int(base_color[0] * age_factor),
                int(base_color[1] * age_factor),
                int(base_color[2] * age_factor),
                220  # Alpha
            ]
            
            return final_color
            
        except Exception as e:
            logger.warning(f"ROI score calculation failed: {e}")
            return [59, 130, 246, 220]  # Default blue
    
    def _categorize_house_size(self, area_sqft: float) -> int:
        """Categorize house by size for radius scaling"""
        if area_sqft < 1000:
            return 8
        elif area_sqft < 2000:
            return 12
        elif area_sqft < 3000:
            return 16
        else:
            return 20
    
    def _categorize_house_age(self, year_built) -> str:
        """Categorize house by age"""
        if not year_built or year_built == 'N/A':
            return 'Unknown'
        elif year_built >= 2000:
            return 'Modern'
        elif year_built >= 1980:
            return 'Contemporary'
        elif year_built >= 1960:
            return 'Mid-Century'
        else:
            return 'Historic'
    
    def _create_house_tooltip(self, house: Dict) -> str:
        """Create enhanced tooltip for house"""
        try:
            return f"""
            <div style="padding: 12px; background: rgba(0,0,0,0.95); border-radius: 10px; color: white; min-width: 280px;">
                <h4 style="margin: 0 0 8px 0; color: #3b82f6;">🏠 {house.get('building_type', 'Property').title()}</h4>
                <p style="margin: 4px 0;"><strong>Address:</strong> {house.get('address', 'N/A')}</p>
                <p style="margin: 4px 0;"><strong>Type:</strong> {house.get('building_type', 'N/A')}</p>
                <p style="margin: 4px 0;"><strong>Area:</strong> {house.get('area_sqft', 0):,.0f} sq ft</p>
                <p style="margin: 4px 0;"><strong>Year Built:</strong> {house.get('year_built', 'N/A')}</p>
                <p style="margin: 4px 0;"><strong>Stories:</strong> {house.get('stories', 'N/A')}</p>
                <p style="margin: 4px 0;"><strong>OSM ID:</strong> <code>{house.get('osm_id', 'N/A')}</code></p>
                <p style="margin: 8px 0 0 0; font-size: 11px; opacity: 0.8;">
                    <em>Click for detailed investment analysis</em>
                </p>
            </div>
            """
        except Exception as e:
            logger.warning(f"Tooltip creation failed: {e}")
            return f"<b>🏠 {house.get('building_type', 'Property')}</b>"

# Example usage
if __name__ == "__main__":
    # Test the enhanced neighborhood system
    print("Testing Enhanced Neighborhood System...")
    
    # Create managers
    boundary_manager = NeighborhoodBoundaryManager()
    heatmap_generator = ContinuousHeatMapGenerator()
    house_visualizer = EnhancedHouseVisualizer()
    
    print("✅ Enhanced neighborhood system components created successfully")
