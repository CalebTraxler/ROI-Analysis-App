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
        """Get approximate county coordinates when geocoding fails - universal approach"""
        try:
            # Try multiple geocoding strategies for any county
            search_strategies = [
                f"{county_name} County, {state_name}, USA",
                f"{county_name}, {state_name}, USA",
                f"{county_name} County, {state_name}",
                f"{county_name}, {state_name}",
                f"{county_name} County",
                f"{county_name}"
            ]
            
            for strategy in search_strategies:
                try:
                    logger.info(f"Trying fallback geocoding: {strategy}")
                    location = self.geolocator.geocode(strategy, timeout=10)
                    
                    if location:
                        # Create a reasonable bounding box around the found location
                        bbox_size = 0.15  # About 10-12 miles radius
                        return {
                            'min_lat': location.latitude - bbox_size,
                            'max_lat': location.latitude + bbox_size,
                            'min_lon': location.longitude - bbox_size,
                            'max_lon': location.longitude + bbox_size,
                            'center_lat': location.latitude,
                            'center_lon': location.longitude
                        }
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Fallback geocoding failed for {strategy}: {e}")
                    continue
            
            # If all geocoding fails, use state-based fallback
            logger.warning(f"All geocoding strategies failed for {county_name}, {state_name}")
            return self._get_state_based_fallback(state_name)
            
        except Exception as e:
            logger.error(f"Error in fallback coordinate lookup: {e}")
            return self._get_state_based_fallback(state_name)
    
    def _get_state_based_fallback(self, state_name: str) -> Dict:
        """Get state-based fallback coordinates when county lookup fails"""
        # Comprehensive state center coordinates
        state_centers = {
            'AL': (32.3182, -86.9023), 'AK': (63.5887, -154.4931), 'AZ': (33.7298, -111.4312),
            'AR': (35.2010, -91.8318), 'CA': (36.7783, -119.4179), 'CO': (39.5501, -105.7821),
            'CT': (41.6032, -73.0877), 'DE': (38.9108, -75.5277), 'FL': (27.7663, -82.6404),
            'GA': (33.0406, -83.6431), 'HI': (19.8968, -155.5828), 'ID': (44.2405, -114.4788),
            'IL': (40.3363, -89.0022), 'IN': (39.8494, -86.2583), 'IA': (42.0115, -93.2105),
            'KS': (38.5266, -96.7265), 'KY': (37.6681, -84.6701), 'LA': (31.1695, -91.8678),
            'ME': (44.6939, -69.3819), 'MD': (39.0639, -76.8021), 'MA': (42.2304, -71.5301),
            'MI': (43.3266, -84.5361), 'MN': (46.7296, -94.6859), 'MS': (32.7416, -89.6787),
            'MO': (38.4561, -92.2884), 'MT': (46.8797, -110.3626), 'NE': (41.4925, -99.9018),
            'NV': (38.8026, -116.4194), 'NH': (43.1939, -71.5724), 'NJ': (40.0583, -74.4057),
            'NM': (34.5199, -105.8701), 'NY': (42.1657, -74.9481), 'NC': (35.7596, -79.0193),
            'ND': (47.5515, -101.0020), 'OH': (40.3888, -82.7649), 'OK': (35.0078, -97.0929),
            'OR': (44.5720, -122.0709), 'PA': (41.2033, -77.1945), 'RI': (41.6809, -71.5118),
            'SC': (33.8569, -80.9450), 'SD': (44.2998, -99.4388), 'TN': (35.7478, -86.6923),
            'TX': (31.9686, -99.9018), 'UT': (39.3210, -111.0937), 'VT': (44.0459, -72.7107),
            'VA': (37.4316, -78.6569), 'WA': (47.7511, -120.7401), 'WV': (38.5976, -80.4549),
            'WI': (43.7844, -88.7879), 'WY': (42.7475, -107.2085)
        }
        
        if state_name in state_centers:
            lat, lon = state_centers[state_name]
            bbox_size = 0.25  # Larger bounding box for state-level fallback
            logger.info(f"Using state center fallback for {state_name}")
            return {
                'min_lat': lat - bbox_size,
                'max_lat': lat + bbox_size,
                'min_lon': lon - bbox_size,
                'max_lon': lon + bbox_size,
                'center_lat': lat,
                'center_lon': lon
            }
        
        # Final fallback to US center
        logger.warning("Using US center fallback")
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
        properties = []
        
        try:
            # Expand the bounding box if it's too small
            lat_range = bbox['max_lat'] - bbox['min_lat']
            lon_range = bbox['max_lon'] - bbox['min_lon']
            
            # If bounding box is too small, expand it
            if lat_range < 0.01 or lon_range < 0.01:
                expansion = 0.01  # About 0.7 miles
                bbox['min_lat'] -= expansion
                bbox['max_lat'] += expansion
                bbox['min_lon'] -= expansion
                bbox['max_lon'] += expansion
                logger.info(f"Expanded bounding box to: {bbox}")
            
            # Determine the best approach based on area size
            area_size = lat_range * lon_range
            
            if area_size > 0.01:  # Large area (county level)
                logger.info(f"Large area detected ({area_size:.4f}), using targeted sampling approach")
                properties = self._fetch_large_area_properties(bbox)
            elif area_size > 0.001:  # Medium area (city level)
                logger.info(f"Medium area detected ({area_size:.6f}), using standard approach")
                properties = self._fetch_standard_properties(bbox)
            else:  # Small area (neighborhood level)
                logger.info(f"Small area detected ({area_size:.8f}), using comprehensive approach")
                properties = self._fetch_comprehensive_properties(bbox)
            
            logger.info(f"Found {len(properties)} properties in bounding box")
            
            # If no properties found, try a broader search
            if len(properties) == 0:
                logger.info("No properties found, trying broader search...")
                properties = self._try_broader_search(bbox)
                    
        except Exception as e:
            logger.error(f"Error fetching properties: {e}")
            # Try fallback method
            properties = self._try_broader_search(bbox)
        
        return properties
    
    def _fetch_comprehensive_properties(self, bbox: Dict) -> List[Dict]:
        """Fetch properties using comprehensive query for small areas"""
        try:
            # For small areas, we can do a comprehensive search
            query = f"""
            [out:json][timeout:30];
            (
              way["building"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              node["building"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["landuse"="residential"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["amenity"="house"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              node["amenity"="house"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="house"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="residential"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="apartments"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="detached"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="yes"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
            );
            out center;
            """
            
            logger.info(f"Using comprehensive query for small area")
            
            url = "https://overpass-api.de/api/interpreter"
            response = requests.post(url, data=query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_overpass_response(data)
            else:
                logger.warning(f"Comprehensive query failed with status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error in comprehensive property fetch: {e}")
            return []
    
    def _fetch_standard_properties(self, bbox: Dict) -> List[Dict]:
        """Fetch properties using standard Overpass query for smaller areas"""
        try:
            # Create a targeted query for smaller areas
            query = f"""
            [out:json][timeout:30];
            (
              way["building"="house"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="residential"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="apartments"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
              way["building"="detached"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
            );
            out center;
            """
            
            logger.info(f"Using standard query for smaller area")
            
            url = "https://overpass-api.de/api/interpreter"
            response = requests.post(url, data=query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_overpass_response(data)
            else:
                logger.warning(f"Standard query failed with status {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error in standard property fetch: {e}")
            return []
    
    def _fetch_large_area_properties(self, bbox: Dict) -> List[Dict]:
        """Fetch properties from large areas using intelligent sampling approach"""
        properties = []
        
        try:
            # Calculate area dimensions
            lat_range = bbox['max_lat'] - bbox['min_lat']
            lon_range = bbox['max_lon'] - bbox['min_lon']
            
            # Determine optimal chunk size based on area
            if lat_range > 0.5 or lon_range > 0.5:  # Very large area
                chunk_count = 6  # 6x6 = 36 chunks
                logger.info(f"Very large area detected, using {chunk_count}x{chunk_count} chunks")
            elif lat_range > 0.2 or lon_range > 0.2:  # Large area
                chunk_count = 4  # 4x4 = 16 chunks
                logger.info(f"Large area detected, using {chunk_count}x{chunk_count} chunks")
            else:  # Medium-large area
                chunk_count = 3  # 3x3 = 9 chunks
                logger.info(f"Medium-large area detected, using {chunk_count}x{chunk_count} chunks")
            
            # Calculate chunk sizes
            lat_step = lat_range / chunk_count
            lon_step = lon_range / chunk_count
            
            logger.info(f"Dividing area into {chunk_count}x{chunk_count} chunks of size {lat_step:.4f}x{lon_step:.4f}")
            
            # Sample from different parts of the area
            for i in range(chunk_count):
                for j in range(chunk_count):
                    chunk_bbox = {
                        'min_lat': bbox['min_lat'] + i * lat_step,
                        'max_lat': bbox['min_lat'] + (i + 1) * lat_step,
                        'min_lon': bbox['min_lon'] + j * lon_step,
                        'max_lon': bbox['min_lon'] + (j + 1) * lon_step
                    }
                    
                    try:
                        # Use different query strategies for different chunk types
                        if i == 0 and j == 0:  # First chunk - comprehensive
                            query = f"""
                            [out:json][timeout:20];
                            (
                              way["building"="house"]({chunk_bbox['min_lat']},{chunk_bbox['min_lon']},{chunk_bbox['max_lat']},{chunk_bbox['max_lon']});
                              way["building"="residential"]({chunk_bbox['min_lat']},{chunk_bbox['min_lon']},{chunk_bbox['max_lat']},{chunk_bbox['max_lon']});
                              way["building"="apartments"]({chunk_bbox['min_lat']},{chunk_bbox['min_lon']},{chunk_bbox['max_lat']},{chunk_bbox['max_lon']});
                            );
                            out center;
                            """
                        else:  # Other chunks - focused on houses
                            query = f"""
                            [out:json][timeout:15];
                            way["building"="house"]({chunk_bbox['min_lat']},{chunk_bbox['min_lon']},{chunk_bbox['max_lat']},{chunk_bbox['max_lon']});
                            out center;
                            """
                        
                        url = "https://overpass-api.de/api/interpreter"
                        response = requests.post(url, data=query, timeout=20)
                        
                        if response.status_code == 200:
                            data = response.json()
                            chunk_properties = self._parse_overpass_response(data)
                            properties.extend(chunk_properties)
                            logger.info(f"Chunk {i},{j}: Found {len(chunk_properties)} properties")
                        else:
                            logger.warning(f"Chunk {i},{j} failed with status {response.status_code}")
                        
                        # Rate limiting between chunks
                        time.sleep(0.5)
                        
                    except Exception as e:
                        logger.warning(f"Error in chunk {i},{j}: {e}")
                        continue
            
            logger.info(f"Large area sampling found {len(properties)} total properties")
            
        except Exception as e:
            logger.error(f"Error in large area property fetch: {e}")
        
        return properties
    
    def _try_broader_search(self, bbox: Dict) -> List[Dict]:
        """Try a broader search when the main query returns no results"""
        properties = []
        
        try:
            # Expand the bounding box significantly
            expansion = 0.05  # About 3.5 miles
            expanded_bbox = {
                'min_lat': bbox['min_lat'] - expansion,
                'max_lat': bbox['max_lat'] + expansion,
                'min_lon': bbox['min_lon'] - expansion,
                'max_lon': bbox['max_lon'] + expansion
            }
            
            logger.info(f"Trying broader search with expanded bbox: {expanded_bbox}")
            
            # Use the same efficient approach for broader search
            lat_range = expanded_bbox['max_lat'] - expanded_bbox['min_lat']
            lon_range = expanded_bbox['max_lon'] - expanded_bbox['min_lon']
            
            if lat_range > 0.1 or lon_range > 0.1:
                logger.info("Broader search using large area approach")
                properties = self._fetch_large_area_properties(expanded_bbox)
            else:
                logger.info("Broader search using standard approach")
                properties = self._fetch_standard_properties(expanded_bbox)
            
            logger.info(f"Broader search found {len(properties)} properties")
            
        except Exception as e:
            logger.error(f"Error in broader search: {e}")
        
        return properties
    
    def _parse_overpass_response(self, data: Dict) -> List[Dict]:
        """Parse Overpass API response into property objects"""
        properties = []
        
        try:
            elements = data.get('elements', [])
            logger.info(f"Parsing {len(elements)} elements from Overpass response")
            
            # Count different types for debugging
            element_types = {}
            building_types = {}
            
            for element in elements:
                element_type = element.get('type')
                tags = element.get('tags', {})
                
                # Track element types
                element_types[element_type] = element_types.get(element_type, 0) + 1
                
                # Handle both ways and nodes
                if element_type in ['way', 'node'] and tags:
                    # Check if this is a building or residential area
                    building_type = tags.get('building')
                    landuse = tags.get('landuse')
                    
                    # Track building types
                    if building_type:
                        building_types[building_type] = building_types.get(building_type, 0) + 1
                    
                    # More inclusive filtering - include any building or residential landuse
                    if building_type or landuse == 'residential':
                        prop = {
                            'osm_id': element.get('id'),
                            'type': element_type,
                            'building_type': building_type or 'unknown',
                            'landuse': landuse or 'unknown',
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
                        
                        # Get coordinates - try multiple methods
                        coordinates_found = False
                        
                        # For ways with center (from out center)
                        if element_type == 'way' and 'center' in element:
                            prop['latitude'] = element['center']['lat']
                            prop['longitude'] = element['center']['lon']
                            coordinates_found = True
                        # For nodes with direct lat/lon
                        elif 'lat' in element and 'lon' in element:
                            prop['latitude'] = element['lat']
                            prop['longitude'] = element['lon']
                            coordinates_found = True
                        # For ways without center, try to calculate from bounds
                        elif element_type == 'way' and 'bounds' in element:
                            bounds = element['bounds']
                            prop['latitude'] = (bounds['minlat'] + bounds['maxlat']) / 2
                            prop['longitude'] = (bounds['minlon'] + bounds['maxlon']) / 2
                            coordinates_found = True
                        
                        # Only add if we have coordinates
                        if coordinates_found:
                            properties.append(prop)
                        else:
                            logger.debug(f"Skipping property without coordinates: {prop['osm_id']}")
            
            # Log summary for debugging
            logger.info(f"Element type breakdown: {element_types}")
            logger.info(f"Building type breakdown: {building_types}")
            logger.info(f"Successfully parsed {len(properties)} properties")
            
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
    
    def get_county_properties(self, county_name: str, state_name: str, max_properties: int = 50000) -> pd.DataFrame:
        """Get properties for a specific county with caching"""
        cache_key = f"{county_name}_{state_name}"
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded {len(cached_data)} properties from cache for {county_name}, {state_name}")
            return cached_data.head(max_properties)
        
        # Fetch fresh data
        logger.info(f"Fetching properties for {county_name}, {state_name}")
        properties = self.fetch_properties_for_county(county_name, state_name)
        
        if properties:
            # Convert to DataFrame
            df = pd.DataFrame(properties)
            
            # Enrich with additional data
            df = self._enrich_property_dataframe(df, county_name, state_name)
            
            # Cache the results
            self._save_to_cache(cache_key, df)
            
            logger.info(f"Successfully fetched {len(df)} properties for {county_name}, {state_name}")
            return df.head(max_properties)
        else:
            logger.warning(f"No properties found for {county_name}, {state_name}")
            return pd.DataFrame()
    
    def get_city_properties(self, city_name: str, county_name: str, state_name: str, max_properties: int = 50000) -> pd.DataFrame:
        """Get properties for a specific city/neighborhood with city-specific boundaries"""
        cache_key = f"{city_name}_{county_name}_{state_name}"
        
        # Try to load from cache first
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded {len(cached_data)} properties from cache for {city_name}, {county_name}, {state_name}")
            return cached_data.head(max_properties)
        
        # Fetch fresh data for the specific city
        logger.info(f"Fetching properties for {city_name}, {county_name}, {state_name}")
        properties = self.fetch_properties_for_city(city_name, county_name, state_name)
        
        if properties:
            # Convert to DataFrame
            df = pd.DataFrame(properties)
            
            # Enrich with additional data
            df = self._enrich_property_dataframe(df, county_name, state_name)
            
            # Cache the results
            self._save_to_cache(cache_key, df)
            
            logger.info(f"Successfully fetched {len(df)} properties for {city_name}, {county_name}, {state_name}")
            return df.head(max_properties)
        else:
            logger.warning(f"No properties found for {city_name}, {county_name}, {state_name}")
            return pd.DataFrame()
    
    def fetch_properties_for_city(self, city_name: str, county_name: str, state_name: str) -> List[Dict]:
        """Fetch properties specifically for a city/neighborhood using city boundaries"""
        properties = []
        
        try:
            # Get city-specific boundaries using more targeted geocoding
            city_bbox = self._get_city_boundaries(city_name, county_name, state_name)
            
            if city_bbox:
                logger.info(f"Found city boundaries for {city_name}: {city_bbox}")
                
                # Use comprehensive search for city-level detail
                city_properties = self._fetch_comprehensive_properties(city_bbox)
                properties.extend(city_properties)
                
                if city_properties:
                    logger.info(f"Found {len(city_properties)} properties in {city_name}")
                else:
                    logger.info(f"No properties found in city boundaries, trying broader search for {city_name}")
                    # Fallback to broader search if city boundaries are too restrictive
                    broader_bbox = self._expand_bbox_for_city(city_bbox)
                    broader_properties = self._fetch_comprehensive_properties(broader_bbox)
                    properties.extend(broader_properties)
                    logger.info(f"Broader search found {len(broader_properties)} properties for {city_name}")
            else:
                logger.warning(f"Could not determine city boundaries for {city_name}, using county-level search")
                # Fallback to county-level search
                county_bbox = self.get_county_boundaries(county_name, state_name)
                if county_bbox:
                    # Filter properties by city name in address data
                    county_properties = self._fetch_comprehensive_properties(county_bbox)
                    city_properties = self._filter_properties_by_city(county_properties, city_name)
                    properties.extend(city_properties)
                    logger.info(f"County-level search with city filtering found {len(city_properties)} properties for {city_name}")
        
        except Exception as e:
            logger.error(f"Error fetching city properties for {city_name}, {county_name}, {state_name}: {e}")
            # Fallback to county-level search
            try:
                county_bbox = self.get_county_boundaries(county_name, state_name)
                if county_bbox:
                    county_properties = self._fetch_comprehensive_properties(county_bbox)
                    city_properties = self._filter_properties_by_city(county_properties, city_name)
                    properties.extend(city_properties)
                    logger.info(f"Fallback county search found {len(city_properties)} properties for {city_name}")
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
        
        return properties
    
    def _get_city_boundaries(self, city_name: str, county_name: str, state_name: str) -> Optional[Dict]:
        """Get city-specific boundaries using targeted geocoding"""
        try:
            # Try multiple search strategies for city boundaries
            search_queries = [
                f"{city_name}, {county_name} County, {state_name}, USA",
                f"{city_name}, {state_name}, USA",
                f"{city_name}, {county_name}, {state_name}",
                f"{city_name}, {state_name}",
                f"{city_name}"
            ]
            
            for search_query in search_queries:
                try:
                    logger.info(f"Trying to geocode city: {search_query}")
                    location = self.geolocator.geocode(search_query, timeout=15)
                    
                    if location:
                        # Create a city-appropriate bounding box
                        # Cities typically have smaller bounding boxes than counties
                        bbox_size = 0.02  # About 1-2 miles radius for city centers
                        
                        city_bbox = {
                            'min_lat': location.latitude - bbox_size,
                            'max_lat': location.latitude + bbox_size,
                            'min_lon': location.longitude - bbox_size,
                            'max_lon': location.longitude + bbox_size,
                            'center_lat': location.latitude,
                            'center_lon': location.longitude
                        }
                        
                        logger.info(f"Found city boundaries for {search_query}: {city_bbox}")
                        return city_bbox
                    
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"City geocoding failed for {search_query}: {e}")
                    continue
            
            logger.warning(f"Could not geocode city boundaries for {city_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error in city boundary lookup: {e}")
            return None
    
    def _expand_bbox_for_city(self, city_bbox: Dict) -> Dict:
        """Expand city bounding box if it's too restrictive"""
        expansion_factor = 2.5  # Expand by 2.5x if city boundaries are too small
        
        lat_range = city_bbox['max_lat'] - city_bbox['min_lat']
        lon_range = city_bbox['max_lon'] - city_bbox['min_lon']
        
        # If the bounding box is very small, expand it
        if lat_range < 0.01 or lon_range < 0.01:
            lat_expansion = (0.01 - lat_range) / 2
            lon_expansion = (0.01 - lon_range) / 2
            
            expanded_bbox = {
                'min_lat': city_bbox['min_lat'] - lat_expansion,
                'max_lat': city_bbox['max_lat'] + lat_expansion,
                'min_lon': city_bbox['min_lon'] - lon_expansion,
                'max_lon': city_bbox['max_lon'] + lon_expansion,
                'center_lat': city_bbox['center_lat'],
                'center_lon': city_bbox['center_lon']
            }
            
            logger.info(f"Expanded city bounding box from {city_bbox} to {expanded_bbox}")
            return expanded_bbox
        
        return city_bbox
    
    def _filter_properties_by_city(self, properties: List[Dict], city_name: str) -> List[Dict]:
        """Filter properties to only include those in the specified city"""
        city_properties = []
        
        for prop in properties:
            # Check if property has city information
            address = prop.get('address', {})
            prop_city = address.get('city', '').lower()
            
            # Check if the property is in the target city
            if (prop_city and city_name.lower() in prop_city) or \
               (not prop_city and city_name.lower() in str(prop.get('osm_id', '')).lower()):
                city_properties.append(prop)
        
        logger.info(f"Filtered {len(city_properties)} properties for city {city_name} from {len(properties)} total properties")
        return city_properties

    def fetch_properties_for_county(self, county_name: str, state_name: str) -> List[Dict]:
        """Fetch properties for a specific county using the main fetch_properties_in_area"""
        bbox = self.get_county_boundaries(county_name, state_name)
        if not bbox:
            logger.error(f"Could not get boundaries for {county_name}, {state_name}")
            return []
        return self.fetch_properties_in_area(bbox)
    
    def _enrich_property_dataframe(self, df: pd.DataFrame, county: str, state: str) -> pd.DataFrame:
        """Enrich the DataFrame with additional public data (estimated value, tax rate)"""
        enriched_df = df.copy()
        
        # Ensure 'latitude' and 'longitude' columns exist
        if 'latitude' not in enriched_df.columns or 'longitude' not in enriched_df.columns:
            logger.warning("Latitude or longitude columns not found in DataFrame. Cannot enrich.")
            return enriched_df
        
        # Add estimated value
        enriched_df['estimated_value'] = enriched_df.apply(
            lambda row: self._estimate_property_value(row.to_dict()) if pd.notna(row['latitude']) and pd.notna(row['longitude']) else None,
            axis=1
        )
        
        # Add property tax rate
        enriched_df['property_tax_rate'] = enriched_df.apply(
            lambda row: self._get_tax_rate(county, state) if pd.notna(row['latitude']) and pd.notna(row['longitude']) else None,
            axis=1
        )
        
        return enriched_df
    
    def _try_alternative_approaches(self, county_name: str, state_name: str, original_bbox: Dict) -> List[Dict]:
        """Try alternative approaches when the main method fails"""
        logger.info(f"Trying alternative approaches for {county_name}, {state_name}")
        
        # Try 1: Broader search area
        try:
            logger.info("Alternative 1: Broader search area")
            expanded_bbox = {
                'min_lat': original_bbox['min_lat'] - 0.1,
                'max_lat': original_bbox['max_lat'] + 0.1,
                'min_lon': original_bbox['min_lon'] - 0.1,
                'max_lon': original_bbox['max_lon'] + 0.1
            }
            properties = self._fetch_standard_properties(expanded_bbox)
            if properties:
                logger.info(f"Alternative 1 successful: found {len(properties)} properties")
                return properties
        except Exception as e:
            logger.debug(f"Alternative 1 failed: {e}")
        
        # Try 2: Focus on major cities in the county
        try:
            logger.info("Alternative 2: Focus on major cities")
            properties = self._search_major_cities_in_county(county_name, state_name)
            if properties:
                logger.info(f"Alternative 2 successful: found {len(properties)} properties")
                return properties
        except Exception as e:
            logger.debug(f"Alternative 2 failed: {e}")
        
        # Try 3: Use state center with very broad search
        try:
            logger.info("Alternative 3: State center broad search")
            state_bbox = self._get_state_based_fallback(state_name)
            # Expand state bbox significantly
            state_bbox['min_lat'] -= 0.2
            state_bbox['max_lat'] += 0.2
            state_bbox['min_lon'] -= 0.2
            state_bbox['max_lon'] += 0.2
            properties = self._fetch_standard_properties(state_bbox)
            if properties:
                logger.info(f"Alternative 3 successful: found {len(properties)} properties")
                return properties
        except Exception as e:
            logger.debug(f"Alternative 3 failed: {e}")
        
        logger.warning("All alternative approaches failed")
        return []
    
    def _search_major_cities_in_county(self, county_name: str, state_name: str) -> List[Dict]:
        """Search for properties in major cities within the county"""
        properties = []
        
        # Common major cities that might be in the county
        major_cities = [
            f"{county_name} City",
            f"{county_name} Town",
            f"{county_name} Village"
        ]
        
        for city_name in major_cities:
            try:
                # Try to geocode the city
                location = self.geolocator.geocode(f"{city_name}, {state_name}, USA", timeout=10)
                if location:
                    # Create a small bounding box around the city
                    city_bbox = {
                        'min_lat': location.latitude - 0.05,
                        'max_lat': location.latitude + 0.05,
                        'min_lon': location.longitude - 0.05,
                        'max_lon': location.longitude + 0.05,
                        'center_lat': location.latitude,
                        'center_lon': location.longitude
                    }
                    
                    # Search for properties in the city
                    city_properties = self._fetch_comprehensive_properties(city_bbox)
                    properties.extend(city_properties)
                    
                    if city_properties:
                        logger.info(f"Found {len(city_properties)} properties in {city_name}")
                    
                    time.sleep(1)  # Rate limiting
                    
            except Exception as e:
                logger.debug(f"City search failed for {city_name}: {e}")
                continue
        
        return properties
    
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
    
    def test_overpass_query(self, county_name: str, state_name: str):
        """Test function to debug Overpass API queries"""
        try:
            logger.info(f"Testing Overpass API for {county_name}, {state_name}")
            
            # Get county boundaries
            bbox = self.get_county_boundaries(county_name, state_name)
            if not bbox:
                logger.error("No bounding box found")
                return
            
            logger.info(f"Bounding box: {bbox}")
            
            # Test simple query
            test_query = f"""
            [out:json][timeout:30];
            way["building"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
            out body;
            >;
            out skel qt;
            """
            
            url = "https://overpass-api.de/api/interpreter"
            response = requests.post(url, data=test_query, timeout=30)
            
            logger.info(f"Test query status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                elements = data.get('elements', [])
                logger.info(f"Test query found {len(elements)} elements")
                
                # Show first few elements
                for i, element in enumerate(elements[:5]):
                    logger.info(f"Element {i}: {element.get('type')} - {element.get('tags', {}).get('building', 'no building tag')}")
            else:
                logger.warning(f"Test query failed: {response.text[:200]}")
                
        except Exception as e:
            logger.error(f"Test query error: {e}")
