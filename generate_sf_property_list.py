#!/usr/bin/env python3
"""
San Francisco Property List Generator

This script generates a comprehensive list of SF properties by:
1. Creating realistic coordinates across all SF neighborhoods
2. Reverse geocoding to get real addresses
3. Generating a complete property list for research
"""

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import logging
from pathlib import Path
import random
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sf_property_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SFPropertyGenerator:
    """Generates comprehensive list of San Francisco properties"""
    
    def __init__(self):
        # Initialize geocoder
        self.geolocator = Nominatim(user_agent="sf_property_generator")
        
        # San Francisco coordinate boundaries
        self.sf_bounds = {
            'lat_min': 37.7,
            'lat_max': 37.8,
            'lon_min': -122.5,
            'lon_max': -122.2
        }
        
        # SF neighborhoods with coordinate ranges
        self.sf_neighborhoods = {
            'Mission': {
                'lat_range': (37.744, 37.760),
                'lon_range': (-122.420, -122.400),
                'density': 0.25  # Higher density for popular areas
            },
            'Financial District': {
                'lat_range': (37.790, 37.800),
                'lon_range': (-122.410, -122.390),
                'density': 0.20
            },
            'Pacific Heights': {
                'lat_range': (37.790, 37.800),
                'lon_range': (-122.440, -122.420),
                'density': 0.15
            },
            'Marina': {
                'lat_range': (37.800, 37.810),
                'lon_range': (-122.440, -122.420),
                'density': 0.15
            },
            'North Beach': {
                'lat_range': (37.800, 37.810),
                'lon_range': (-122.410, -122.390),
                'density': 0.15
            },
            'Chinatown': {
                'lat_range': (37.790, 37.800),
                'lon_range': (-122.410, -122.390),
                'density': 0.20
            },
            'South of Market': {
                'lat_range': (37.770, 37.790),
                'lon_range': (-122.410, -122.390),
                'density': 0.25
            },
            'Hayes Valley': {
                'lat_range': (37.770, 37.780),
                'lon_range': (-122.420, -122.400),
                'density': 0.20
            },
            'Castro': {
                'lat_range': (37.750, 37.760),
                'lon_range': (-122.440, -122.420),
                'density': 0.20
            },
            'Noe Valley': {
                'lat_range': (37.750, 37.760),
                'lon_range': (-122.430, -122.410),
                'density': 0.15
            },
            'Haight-Ashbury': {
                'lat_range': (37.760, 37.770),
                'lon_range': (-122.450, -122.430),
                'density': 0.20
            },
            'Outer Sunset': {
                'lat_range': (37.720, 37.740),
                'lon_range': (-122.500, -122.480),
                'density': 0.10
            },
            'Inner Sunset': {
                'lat_range': (37.740, 37.760),
                'lon_range': (-122.480, -122.460),
                'density': 0.15
            },
            'Laurel Heights': {
                'lat_range': (37.780, 37.790),
                'lon_range': (-122.440, -122.420),
                'density': 0.15
            },
            'Western Addition': {
                'lat_range': (37.780, 37.790),
                'lon_range': (-122.420, -122.400),
                'density': 0.20
            },
            'Mission Dolores': {
                'lat_range': (37.740, 37.750),
                'lon_range': (-122.420, -122.400),
                'density': 0.20
            }
        }
    
    def generate_coordinates_for_neighborhood(self, neighborhood: str, num_properties: int) -> list:
        """Generate realistic coordinates for a specific neighborhood"""
        if neighborhood not in self.sf_neighborhoods:
            logger.warning(f"Unknown neighborhood: {neighborhood}")
            return []
        
        coords = []
        neighborhood_info = self.sf_neighborhoods[neighborhood]
        
        for i in range(num_properties):
            # Generate random coordinates within neighborhood bounds
            lat = random.uniform(
                neighborhood_info['lat_range'][0],
                neighborhood_info['lat_range'][1]
            )
            lon = random.uniform(
                neighborhood_info['lon_range'][0],
                neighborhood_info['lon_range'][1]
            )
            
            coords.append({
                'neighborhood': neighborhood,
                'latitude': round(lat, 6),
                'longitude': round(lon, 6),
                'property_id': f"{neighborhood.lower().replace(' ', '_')}_{i+1:03d}"
            })
        
        return coords
    
    def generate_all_sf_coordinates(self, properties_per_neighborhood: int = 20) -> list:
        """Generate coordinates for all SF neighborhoods"""
        logger.info(f"Generating {properties_per_neighborhood} properties per neighborhood...")
        
        all_coords = []
        
        for neighborhood in self.sf_neighborhoods.keys():
            logger.info(f"Generating coordinates for {neighborhood}...")
            coords = self.generate_coordinates_for_neighborhood(
                neighborhood, properties_per_neighborhood
            )
            all_coords.extend(coords)
            time.sleep(0.1)  # Small delay to be respectful
        
        logger.info(f"Generated {len(all_coords)} total coordinate sets")
        return all_coords
    
    def reverse_geocode_coordinates(self, coords_list: list, max_properties: int = 100) -> list:
        """Reverse geocode coordinates to get addresses"""
        logger.info(f"Starting reverse geocoding for {len(coords_list)} coordinates...")
        
        properties = []
        success_count = 0
        
        for i, coord in enumerate(coords_list[:max_properties]):
            try:
                logger.info(f"Geocoding {i+1}/{min(len(coords_list), max_properties)}: {coord['neighborhood']}")
                
                # Reverse geocode
                location = self.geolocator.reverse(
                    (coord['latitude'], coord['longitude']), 
                    timeout=10
                )
                
                if location and location.address:
                    # Check if it's actually in San Francisco
                    if 'San Francisco' in location.address or 'CA 941' in location.address:
                        property_data = {
                            'property_id': coord['property_id'],
                            'address': location.address,
                            'latitude': coord['latitude'],
                            'longitude': coord['longitude'],
                            'neighborhood': coord['neighborhood'],
                            'geocoding_success': True
                        }
                        properties.append(property_data)
                        success_count += 1
                        logger.info(f"✓ Success: {location.address}")
                    else:
                        logger.warning(f"Location outside SF: {location.address}")
                else:
                    logger.warning(f"No address found for coordinates")
                
                # Rate limiting
                time.sleep(1)  # 1 second delay between requests
                
            except GeocoderTimedOut:
                logger.warning(f"Geocoding timeout for {coord['neighborhood']}")
                continue
            except Exception as e:
                logger.error(f"Error geocoding {coord['neighborhood']}: {e}")
                continue
        
        logger.info(f"Reverse geocoding completed. Success: {success_count}/{min(len(coords_list), max_properties)}")
        return properties
    
    def create_csv_template(self, properties: list, output_file: str = "sf_properties_for_research.csv") -> str:
        """Create CSV template with addresses and empty data fields"""
        logger.info(f"Creating CSV template with {len(properties)} properties...")
        
        # Create template data
        template_data = []
        for prop in properties:
            row = {
                'Address': prop['address'],
                'Latitude': prop['latitude'],
                'Longitude': prop['longitude'],
                'Neighborhood': prop['neighborhood'],
                'Created_At': '',
                'Property_Type': '',
                'Style': '',
                'Year_Built': '',
                'Square_Footage': '',
                'Bedrooms': '',
                'Bathrooms': '',
                'Lot_Size': '',
                'Price': '',
                'Property_Tax': '',
                'HOA_Fees': '',
                'Parking_Spaces': '',
                'Heating': '',
                'Cooling': '',
                'Appliances': '',
                'Features': '',
                'School_District': '',
                'Walk_Score': '',
                'Transit_Score': '',
                'Last_Sold_Date': '',
                'Last_Sold_Price': '',
                'Estimated_Monthly_Payment': ''
            }
            template_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(template_data)
        df.to_csv(output_file, index=False)
        
        logger.info(f"CSV template saved to {output_file}")
        return output_file
    
    def generate_property_list(self, properties_per_neighborhood: int = 20, max_geocode: int = 100) -> str:
        """Main function to generate complete SF property list"""
        logger.info("Starting San Francisco property list generation...")
        
        # Step 1: Generate coordinates
        coords = self.generate_all_sf_coordinates(properties_per_neighborhood)
        
        # Step 2: Reverse geocode to addresses
        properties = self.reverse_geocode_coordinates(coords, max_geocode)
        
        # Step 3: Create CSV template
        output_file = self.create_csv_template(properties)
        
        logger.info(f"Property list generation completed!")
        logger.info(f"Generated {len(properties)} properties with addresses")
        logger.info(f"Output file: {output_file}")
        
        return output_file

def main():
    """Main function to run the SF property generator"""
    try:
        # Initialize generator
        generator = SFPropertyGenerator()
        
        # Generate property list
        # Start with smaller numbers for testing
        output_file = generator.generate_property_list(
            properties_per_neighborhood=10,  # 10 properties per neighborhood
            max_geocode=50  # Limit geocoding to 50 for testing
        )
        
        print(f"\n✅ SF Property List Generation Completed!")
        print(f"📁 Output file: {output_file}")
        print(f"🔍 Ready for property research and data filling!")
        
        # Show sample of generated data
        df = pd.read_csv(output_file)
        print(f"\n📊 Sample of generated properties:")
        print(f"Total properties: {len(df)}")
        print(f"Neighborhoods: {df['Neighborhood'].nunique()}")
        print(f"\nFirst 5 properties:")
        print(df[['Address', 'Neighborhood', 'Latitude', 'Longitude']].head())
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main()
