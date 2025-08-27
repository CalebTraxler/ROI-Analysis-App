"""
OpenStreetMap Property Data Fetcher

This module fetches property data from OpenStreetMap based on county boundaries
and integrates with the existing ROI analysis dashboard.
"""

import requests
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import logging
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenStreetMapProperties:
    """Fetches and processes property data from OpenStreetMap"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="roi_analysis_app")
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.properties_cache_file = self.cache_dir / "osm_properties_cache.pkl"
        
    def get_county_boundaries(self, county_name: str, state_name: str) -> Optional[Dict]:
        """Get county boundaries using Nominatim geocoding"""
        try:
            # Try different search formats for county names
            search_queries = [
                f"{county_name} County, {state_name}, USA",
                f"{county_name}, {state_name}, USA",
                f"{county_name} County, {state_name}",
                f"{county_name}, {state_name}"
            ]
            
            for search_query in search_queries:
                try:
                    logger.info(f"Trying to geocode: {search_query}")
                    location = self.geolocator.geocode(search_query, timeout=15)
                    
                    if location:
                        # Get bounding box for the county
                        bbox = location.raw.get('boundingbox', [])
                        if bbox:
                            logger.info(f"Found boundaries for {search_query}: {bbox}")
                            return {
                                'min_lat': float(bbox[0]),
                                'max_lat': float(bbox[1]),
                                'min_lon': float(bbox[2]),
                                'max_lon': float(bbox[3]),
                                'center_lat': location.latitude,
                                'center_lon': location.longitude
                            }
                    
                    # Rate limiting between attempts
                    time.sleep(2)
                    
                except Exception as e:
                    logger.warning(f"Geocoding failed for {search_query}: {e}")
                    continue
            
            # If all geocoding attempts fail, try to use approximate coordinates
            logger.warning(f"All geocoding attempts failed for {county_name}, {state_name}")
            return self._get_approximate_county_coordinates(county_name, state_name)
            
        except Exception as e:
            logger.error(f"Error getting county boundaries: {e}")
            return self._get_approximate_county_coordinates(county_name, state_name)
    
    def _get_approximate_county_coordinates(self, county_name: str, state_name: str) -> Optional[Dict]:
        """Get approximate county coordinates when geocoding fails"""
        # Known county centers for major counties
        county_coordinates = {
            'CA': {
                'Alameda': (37.7652, -122.2416),
                'Los Angeles': (34.0522, -118.2437),
                'San Diego': (32.7157, -117.1611),
                'Orange': (33.7175, -117.8311),
                'Santa Clara': (37.3541, -121.9552),
                'San Francisco': (37.7749, -122.4194),
                'Marin': (37.9735, -122.5311),
                'Contra Costa': (37.9191, -122.3281),
                'San Mateo': (37.4969, -122.3330),
                'Ventura': (34.3705, -119.1391)
            },
            'TX': {
                'Harris': (29.7604, -95.3698),
                'Dallas': (32.7767, -96.7970),
                'Tarrant': (32.7555, -97.3308),
                'Bexar': (29.4241, -98.4936),
                'Travis': (30.2672, -97.7431)
            },
            'NY': {
                'Kings': (40.6782, -73.9442),
                'Queens': (40.7282, -73.7949),
                'New York': (40.7128, -74.0060),
                'Bronx': (40.8448, -73.8648),
                'Richmond': (40.5795, -74.1502)
            },
            'FL': {
                'Miami-Dade': (25.7617, -80.1918),
                'Broward': (26.1224, -80.1373),
                'Palm Beach': (26.7153, -80.0534),
                'Hillsborough': (27.9904, -82.3018),
                'Orange': (28.5383, -81.3792)
            }
        }
        
        # Try to find exact county match
        if state_name in county_coordinates:
            for county, coords in county_coordinates[state_name].items():
                if county.lower() in county_name.lower() or county_name.lower() in county.lower():
                    lat, lon = coords
                    # Create a bounding box around the county center
                    bbox_size = 0.1  # Approximately 6-7 miles
                    return {
                        'min_lat': lat - bbox_size,
                        'max_lat': lat + bbox_size,
                        'min_lon': lon - bbox_size,
                        'max_lon': lon + bbox_size,
                        'center_lat': lat,
                        'center_lon': lon
                    }
        
        # Fallback to state center with large bounding box
        state_centers = {
            'CA': (36.7783, -119.4179),
            'TX': (31.9686, -99.9018),
            'FL': (27.7663, -82.6404),
            'NY': (42.1657, -74.9481),
            'IL': (40.3363, -89.0022),
            'OH': (40.3888, -82.7649),
            'GA': (33.0406, -83.6431),
            'NC': (35.5397, -79.8431),
            'MI': (43.3266, -84.5361),
            'PA': (41.2033, -77.1945)
        }
        
        if state_name in state_centers:
            lat, lon = state_centers[state_name]
            bbox_size = 0.2  # Larger bounding box for state-level fallback
            return {
                'min_lat': lat - bbox_size,
                'max_lat': lat + bbox_size,
                'min_lon': lon - bbox_size,
                'max_lon': lon + bbox_size,
                'center_lat': lat,
                'center_lon': lon
            }
        
        # Final fallback to US center
        return {
            'min_lat': 39.8283 - 0.3,
            'max_lat': 39.8283 + 0.3,
            'min_lon': -98.5795 - 0.3,
            'max_lon': -98.5795 + 0.3,
            'center_lat': 39.8283,
            'center_lon': -98.5795
        }
    
    def fetch_properties_in_area(self, bbox: Dict, property_types: List[str] = None) -> List[Dict]:
        """Fetch properties from OpenStreetMap within the bounding box"""
        if property_types is None:
            property_types = ['residential', 'house', 'apartment', 'condo']
        
        properties = []
        
        try:
            # Create a comprehensive Overpass query for the bounding box
            query = f"""
            [out:json][timeout:30];
            (
              way["building"="residential"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="house"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="apartment"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="detached"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["landuse"="residential"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              node["amenity"="house"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
            );
            out body;
            >;
            out skel qt;
            """
            
            # Use Overpass API
            url = "https://overpass-api.de/api/interpreter"
            response = requests.post(url, data=query, timeout=45)
            
            if response.status_code == 200:
                data = response.json()
                properties = self._parse_overpass_response(data)
                logger.info(f"Found {len(properties)} properties in bounding box")
            else:
                logger.warning(f"Overpass API returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching properties: {e}")
        
        return properties
    
    def _parse_overpass_response(self, data: Dict) -> List[Dict]:
        """Parse Overpass API response into property objects"""
        properties = []
        
        try:
            elements = data.get('elements', [])
            
            for element in elements:
                if element.get('type') == 'way' and 'tags' in element:
                    tags = element['tags']
                    
                    # Extract property information
                    prop = {
                        'osm_id': element.get('id'),
                        'type': 'way',
                        'building_type': tags.get('building', 'unknown'),
                        'landuse': tags.get('landuse', 'unknown'),
                        'address': {
                            'street': tags.get('addr:street'),
                            'housenumber': tags.get('addr:housenumber'),
                            'city': tags.get('addr:city'),
                            'state': tags.get('addr:state'),
                            'postcode': tags.get('addr:postcode')
                        },
                        'features': {
                            'floors': tags.get('building:levels'),
                            'units': tags.get('building:units'),
                            'year_built': tags.get('start_date'),
                            'roof_type': tags.get('roof:type'),
                            'material': tags.get('building:material')
                        }
                    }
                    
                    # Get coordinates for the way
                    if 'center' in element:
                        prop['latitude'] = element['center']['lat']
                        prop['longitude'] = element['center']['lon']
                    elif 'lat' in element and 'lon' in element:
                        prop['latitude'] = element['lat']
                        prop['longitude'] = element['lon']
                    
                    properties.append(prop)
                    
        except Exception as e:
            logger.error(f"Error parsing Overpass response: {e}")
        
        return properties
    
    def enrich_with_public_data(self, properties: List[Dict], county: str, state: str) -> List[Dict]:
        """Enrich property data with additional public information"""
        enriched_properties = []
        
        for prop in properties:
            enriched_prop = prop.copy()
            
            # Add estimated property value based on location and type
            if 'latitude' in prop and 'longitude' in prop:
                enriched_prop['estimated_value'] = self._estimate_property_value(prop)
                enriched_prop['property_tax_rate'] = self._get_tax_rate(county, state)
            
            enriched_properties.append(enriched_prop)
        
        return enriched_properties
    
    def _estimate_property_value(self, property_data: Dict) -> Optional[float]:
        """Estimate property value based on available data"""
        base_value = 250000  # Base value for residential properties
        
        # Adjust based on building features
        if 'features' in property_data:
            features = property_data['features']
            
            if features.get('floors'):
                try:
                    floors = int(features['floors'])
                    base_value *= (1 + (floors - 1) * 0.15)
                except:
                    pass
            
            if features.get('units'):
                try:
                    units = int(features['units'])
                    base_value *= (1 + (units - 1) * 0.2)
                except:
                    pass
        
        return round(base_value, -3)  # Round to nearest thousand
    
    def _get_tax_rate(self, county: str, state: str) -> float:
        """Get estimated property tax rate for the county"""
        tax_rates = {
            'CA': 1.25,  # California average
            'TX': 1.80,  # Texas average
            'NY': 1.68,  # New York average
            'FL': 0.98,  # Florida average
            'default': 1.20
        }
        
        return tax_rates.get(state, tax_rates['default'])
    
    def get_county_properties(self, county_name: str, state_name: str, 
                            max_properties: int = 1000) -> pd.DataFrame:
        """Main method to get all properties for a county"""
        try:
            # Check cache first
            cache_key = f"{county_name}_{state_name}_properties"
            cached_data = self._load_from_cache(cache_key)
            
            if cached_data is not None:
                logger.info(f"Loaded {len(cached_data)} properties from cache for {county_name}, {state_name}")
                return cached_data
            
            # Get county boundaries
            bbox = self.get_county_boundaries(county_name, state_name)
            if not bbox:
                logger.error(f"Could not get boundaries for {county_name}, {state_name}")
                return pd.DataFrame()
            
            # Fetch properties
            logger.info(f"Fetching properties for {county_name}, {state_name}")
            properties = self.fetch_properties_in_area(bbox)
            
            if not properties:
                logger.warning(f"No properties found for {county_name}, {state_name}")
                return pd.DataFrame()
            
            # Limit properties if too many
            if len(properties) > max_properties:
                properties = properties[:max_properties]
                logger.info(f"Limited to {max_properties} properties")
            
            # Enrich with additional data
            enriched_properties = self.enrich_with_public_data(properties, county_name, state_name)
            
            # Convert to DataFrame
            df = pd.DataFrame(enriched_properties)
            
            # Save to cache
            self._save_to_cache(cache_key, df)
            
            logger.info(f"Successfully fetched {len(df)} properties for {county_name}, {state_name}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting county properties: {e}")
            return pd.DataFrame()
    
    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Load data from cache"""
        try:
            if self.properties_cache_file.exists():
                cache_data = pd.read_pickle(self.properties_cache_file)
                if key in cache_data:
                    return cache_data[key]
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
        return None
    
    def _save_to_cache(self, key: str, data: pd.DataFrame):
        """Save data to cache"""
        try:
            cache_data = {}
            if self.properties_cache_file.exists():
                cache_data = pd.read_pickle(self.properties_cache_file)
            
            cache_data[key] = data
            pd.to_pickle(cache_data, self.properties_cache_file)
            
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")
    
    def get_property_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for properties"""
        if df.empty:
            return {}
        
        summary = {
            'total_properties': len(df),
            'property_types': df.get('building_type', pd.Series()).value_counts().to_dict(),
            'avg_estimated_value': df.get('estimated_value', pd.Series()).mean(),
            'median_estimated_value': df.get('estimated_value', pd.Series()).median(),
            'coordinate_coverage': (df['latitude'].notna().sum() / len(df)) * 100,
            'address_coverage': (df['address'].apply(lambda x: x.get('street') is not None).sum() / len(df)) * 100
        }
        
        return summary
