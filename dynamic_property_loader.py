#!/usr/bin/env python3
"""
Dynamic Property Loader for ROI Analysis Platform
Loads OpenStreetMap properties based on zoom level and map view area
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point, Polygon
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MapViewState:
    """Current map view state for dynamic loading"""
    center_lat: float
    center_lon: float
    zoom_level: float
    bounds: Dict[str, float]  # min_lat, max_lat, min_lon, max_lon
    viewport_width: float
    viewport_height: float

@dataclass
class PropertyCluster:
    """Property cluster for different zoom levels"""
    center_lat: float
    center_lon: float
    property_count: int
    average_value: Optional[float] = None
    cluster_radius: float = 0.001  # degrees

class DynamicPropertyLoader:
    """Dynamic property loading based on zoom level and map view"""
    
    def __init__(self):
        self.cache = {}  # Simple in-memory cache
        self.max_cache_size = 1000
        self.cache_ttl = 300  # 5 minutes
        
    def calculate_map_bounds(self, center_lat: float, center_lon: float, zoom_level: float) -> Dict[str, float]:
        """Calculate map bounds based on center and zoom level"""
        # Zoom-based radius calculation
        # Higher zoom = smaller area = more detail
        if zoom_level <= 8:
            radius_degrees = 2.0  # Very wide view
        elif zoom_level <= 10:
            radius_degrees = 1.0  # Wide view
        elif zoom_level <= 12:
            radius_degrees = 0.5  # Medium view
        elif zoom_level <= 14:
            radius_degrees = 0.25  # Close view
        elif zoom_level <= 16:
            radius_degrees = 0.1   # Very close view
        else:
            radius_degrees = 0.05  # Street level view
        
        bounds = {
            'min_lat': center_lat - radius_degrees,
            'max_lat': center_lat + radius_degrees,
            'min_lon': center_lon - radius_degrees,
            'max_lon': center_lon + radius_degrees
        }
        
        return bounds
    
    def get_properties_for_area(self, bounds: Dict[str, float], zoom_level: float) -> pd.DataFrame:
        """Get properties for a specific map area and zoom level"""
        cache_key = f"{bounds['min_lat']:.4f}_{bounds['min_lon']:.4f}_{bounds['max_lat']:.4f}_{bounds['max_lon']:.4f}_{zoom_level:.1f}"
        
        # Check cache first
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                logger.info(f"Using cached properties for area: {len(cached_data['properties'])} properties")
                return cached_data['properties']
        
        try:
            # Calculate area size for property limit
            area_size = (bounds['max_lat'] - bounds['min_lat']) * (bounds['max_lon'] - bounds['min_lon'])
            
            # Adjust property limit based on zoom level and area size
            if zoom_level <= 10:
                max_properties = 100  # Few properties for wide view
            elif zoom_level <= 12:
                max_properties = 500  # More properties for medium view
            elif zoom_level <= 14:
                max_properties = 1000  # Many properties for close view
            else:
                max_properties = 2000  # Maximum detail for street level
            
            # Use OSMnx to get buildings in the area
            buildings = ox.geometries_from_bbox(
                bounds['max_lat'], bounds['min_lat'],
                bounds['min_lon'], bounds['max_lon'],
                tags={'building': True}
            )
            
            if buildings.empty:
                logger.info(f"No buildings found in area at zoom level {zoom_level}")
                return pd.DataFrame()
            
            # Convert to DataFrame and process
            properties_df = self._process_buildings_to_properties(buildings, zoom_level)
            
            # Limit properties based on zoom level
            if len(properties_df) > max_properties:
                # Sample properties for performance
                properties_df = properties_df.sample(n=max_properties, random_state=42)
                logger.info(f"Sampled {max_properties} properties from {len(properties_df)} total")
            
            # Cache the results
            self.cache[cache_key] = {
                'properties': properties_df,
                'timestamp': time.time()
            }
            
            # Clean cache if too large
            if len(self.cache) > self.max_cache_size:
                self._clean_cache()
            
            logger.info(f"Loaded {len(properties_df)} properties for area at zoom level {zoom_level}")
            return properties_df
            
        except Exception as e:
            logger.error(f"Error loading properties for area: {e}")
            return pd.DataFrame()
    
    def _process_buildings_to_properties(self, buildings: gpd.GeoDataFrame, zoom_level: float) -> pd.DataFrame:
        """Process OSM buildings into property DataFrame"""
        properties = []
        
        for idx, building in buildings.iterrows():
            try:
                # Extract building information
                building_type = building.get('building', 'unknown')
                address = building.get('addr:housenumber', '') + ' ' + building.get('addr:street', '')
                
                # Calculate area in square feet
                if hasattr(building.geometry, 'area'):
                    area_sqft = building.geometry.area * 10.764  # Convert sq meters to sq feet
                else:
                    area_sqft = 0
                
                # Get coordinates
                if hasattr(building.geometry, 'centroid'):
                    centroid = building.geometry.centroid
                    lat, lon = centroid.y, centroid.x
                else:
                    continue
                
                # Create property data
                property_data = {
                    'osm_id': str(idx),
                    'latitude': lat,
                    'longitude': lon,
                    'building_type': building_type,
                    'address': address.strip() if address.strip() else f"Building {idx}",
                    'area_sqft': area_sqft,
                    'year_built': building.get('start_date', None),
                    'stories': building.get('building:levels', None),
                    'roof_type': building.get('roof:type', None),
                    'material': building.get('building:material', None)
                }
                
                properties.append(property_data)
                
            except Exception as e:
                logger.warning(f"Error processing building {idx}: {e}")
                continue
        
        return pd.DataFrame(properties)
    
    def create_property_clusters(self, properties_df: pd.DataFrame, zoom_level: float) -> List[PropertyCluster]:
        """Create property clusters for lower zoom levels"""
        if properties_df.empty:
            return []
        
        if zoom_level >= 14:
            # No clustering needed at high zoom
            return []
        
        # Simple clustering based on zoom level
        cluster_radius = 0.01 if zoom_level <= 10 else 0.005
        
        clusters = []
        processed_properties = set()
        
        for idx, prop in properties_df.iterrows():
            if idx in processed_properties:
                continue
            
            # Find nearby properties
            nearby_props = []
            for other_idx, other_prop in properties_df.iterrows():
                if other_idx in processed_properties:
                    continue
                
                distance = np.sqrt(
                    (prop['latitude'] - other_prop['latitude'])**2 + 
                    (prop['longitude'] - other_prop['longitude'])**2
                )
                
                if distance <= cluster_radius:
                    nearby_props.append(other_prop)
                    processed_properties.add(other_idx)
            
            if nearby_props:
                # Calculate cluster center
                cluster_lat = np.mean([p['latitude'] for p in nearby_props])
                cluster_lon = np.mean([p['longitude'] for p in nearby_props])
                
                cluster = PropertyCluster(
                    center_lat=cluster_lat,
                    center_lon=cluster_lon,
                    property_count=len(nearby_props),
                    cluster_radius=cluster_radius
                )
                
                clusters.append(cluster)
        
        return clusters
    
    def get_zoom_appropriate_properties(self, center_lat: float, center_lon: float, zoom_level: float) -> Dict[str, Any]:
        """Get properties appropriate for the current zoom level"""
        bounds = self.calculate_map_bounds(center_lat, center_lon, zoom_level)
        
        # Get raw properties
        properties_df = self.get_properties_for_area(bounds, zoom_level)
        
        # Create clusters for lower zoom levels
        clusters = self.create_property_clusters(properties_df, zoom_level)
        
        return {
            'properties': properties_df,
            'clusters': clusters,
            'bounds': bounds,
            'zoom_level': zoom_level,
            'total_properties': len(properties_df),
            'total_clusters': len(clusters)
        }
    
    def _clean_cache(self):
        """Clean old cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, data in self.cache.items()
            if current_time - data['timestamp'] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
    
    def get_property_summary(self, properties_df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for properties"""
        if properties_df.empty:
            return {}
        
        summary = {
            'total_properties': len(properties_df),
            'building_types': properties_df['building_type'].value_counts().to_dict(),
            'avg_area_sqft': properties_df['area_sqft'].mean() if 'area_sqft' in properties_df.columns else 0,
            'total_area_sqft': properties_df['area_sqft'].sum() if 'area_sqft' in properties_df.columns else 0,
            'coordinate_coverage': properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()].shape[0] / len(properties_df) * 100
        }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Test the dynamic property loader
    loader = DynamicPropertyLoader()
    
    # Test coordinates (San Francisco)
    test_lat, test_lon = 37.7749, -122.4194
    
    print("Testing Dynamic Property Loader...")
    
    # Test different zoom levels
    for zoom in [8, 10, 12, 14, 16]:
        print(f"\nZoom Level {zoom}:")
        result = loader.get_zoom_appropriate_properties(test_lat, test_lon, zoom)
        print(f"  Properties: {result['total_properties']}")
        print(f"  Clusters: {result['total_clusters']}")
        print(f"  Bounds: {result['bounds']}")
