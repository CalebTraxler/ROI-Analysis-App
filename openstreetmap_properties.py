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
            # Search for county with state
            search_query = f"{county_name} County, {state_name}, USA"
            location = self.geolocator.geocode(search_query)
            
            if location:
                # Get bounding box for the county
                bbox = location.raw.get('boundingbox', [])
                if bbox:
                    return {
                        'min_lat': float(bbox[0]),
                        'max_lat': float(bbox[1]),
                        'min_lon': float(bbox[2]),
                        'max_lon': float(bbox[3]),
                        'center_lat': location.latitude,
                        'center_lon': location.longitude
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting county boundaries: {e}")
            return None
    
    def fetch_properties_in_area(self, bbox: Dict, property_types: List[str] = None) -> List[Dict]:
        """Fetch properties from OpenStreetMap within the bounding box"""
        if property_types is None:
            property_types = ['residential', 'house', 'apartment', 'condo']
        
        properties = []
        
        # Overpass API query for properties in the area
        for prop_type in property_types:
            try:
                # Create Overpass query for the bounding box
                query = f"""
                [out:json][timeout:25];
                (
                  way["building"="residential"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
                  way["landuse"="residential"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
                  node["amenity"="house"]({bbox['min_lat']},{bbox['min_lon']},{bbox['max_lat']},{bbox['max_lon']});
                );
                out body;
                >;
                out skel qt;
                """
                
                # Use Overpass API
                url = "https://overpass-api.de/api/interpreter"
                response = requests.post(url, data=query, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    properties.extend(self._parse_overpass_response(data, prop_type))
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error fetching {prop_type} properties: {e}")
                continue
        
        return properties
    
    def _parse_overpass_response(self, data: Dict, property_type: str) -> List[Dict]:
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
                        'property_type': property_type,
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
    
    def get_property_details(self, osm_id: int, osm_type: str) -> Optional[Dict]:
        """Get detailed information about a specific property"""
        try:
            # Use Overpass API to get detailed property info
            query = f"""
            [out:json][timeout:25];
            {osm_type}({osm_id});
            out body;
            >;
            out skel qt;
            """
            
            url = "https://overpass-api.de/api/interpreter"
            response = requests.post(url, data=query, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                elements = data.get('elements', [])
                
                if elements:
                    element = elements[0]
                    return self._parse_overpass_response([element], 'detailed')[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting property details: {e}")
            return None
    
    def enrich_with_public_data(self, properties: List[Dict], county: str, state: str) -> List[Dict]:
        """Enrich property data with additional public information"""
        enriched_properties = []
        
        for prop in properties:
            enriched_prop = prop.copy()
            
            # Add estimated property value based on location and type
            if 'latitude' in prop and 'longitude' in prop:
                # This is a placeholder - in a real implementation, you'd integrate
                # with county assessor data or other public records
                enriched_prop['estimated_value'] = self._estimate_property_value(prop)
                enriched_prop['property_tax_rate'] = self._get_tax_rate(county, state)
            
            enriched_properties.append(enriched_prop)
        
        return enriched_properties
    
    def _estimate_property_value(self, property_data: Dict) -> Optional[float]:
        """Estimate property value based on available data"""
        # This is a simplified estimation - in practice, you'd use:
        # - County assessor data
        # - Recent sales data
        # - Property characteristics
        # - Market trends
        
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
        # Simplified tax rates - in practice, you'd query county assessor data
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
