#!/usr/bin/env python3
"""
Comprehensive San Francisco Property Database Generator

This script generates 500+ SF properties across all neighborhoods
for integration into the ROI analysis system.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
from pathlib import Path

class ComprehensiveSFPropertyGenerator:
    """Generates comprehensive SF property database for ROI analysis"""
    
    def __init__(self):
        # SF neighborhoods with comprehensive coverage
        self.sf_neighborhoods = {
            'Mission': {
                'lat_range': (37.744, 37.760),
                'lon_range': (-122.420, -122.400),
                'price_range': (1400000, 2200000),
                'style': 'Victorian',
                'year_range': (1890, 1920),
                'sqft_range': (1800, 2800),
                'bedroom_range': (2, 4),
                'bathroom_range': (1, 3),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            },
            'Financial District': {
                'lat_range': (37.790, 37.800),
                'lon_range': (-122.410, -122.390),
                'price_range': (2200000, 3500000),
                'style': 'Victorian/Modern',
                'year_range': (1890, 2015),
                'sqft_range': (2000, 3500),
                'bedroom_range': (2, 4),
                'bathroom_range': (2, 4),
                'walk_score_range': (90, 98),
                'transit_score_range': (85, 95)
            },
            'Pacific Heights': {
                'lat_range': (37.790, 37.800),
                'lon_range': (-122.440, -122.420),
                'price_range': (2700000, 4500000),
                'style': 'Victorian/Classic',
                'year_range': (1880, 1940),
                'sqft_range': (2500, 4500),
                'bedroom_range': (3, 6),
                'bathroom_range': (2, 5),
                'walk_score_range': (75, 90),
                'transit_score_range': (70, 85)
            },
            'Marina': {
                'lat_range': (37.800, 37.810),
                'lon_range': (-122.440, -122.420),
                'price_range': (2200000, 3800000),
                'style': 'Mediterranean',
                'year_range': (1920, 1950),
                'sqft_range': (2200, 4000),
                'bedroom_range': (3, 5),
                'bathroom_range': (2, 4),
                'walk_score_range': (80, 90),
                'transit_score_range': (75, 85)
            },
            'North Beach': {
                'lat_range': (37.800, 37.810),
                'lon_range': (-122.410, -122.390),
                'price_range': (1900000, 3200000),
                'style': 'Italianate',
                'year_range': (1900, 1940),
                'sqft_range': (2000, 3500),
                'bedroom_range': (2, 4),
                'bathroom_range': (2, 3),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            },
            'Chinatown': {
                'lat_range': (37.790, 37.800),
                'lon_range': (-122.410, -122.390),
                'price_range': (1600000, 2800000),
                'style': 'Traditional',
                'year_range': (1900, 1940),
                'sqft_range': (1800, 3000),
                'bedroom_range': (2, 4),
                'bathroom_range': (1, 3),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            },
            'South of Market': {
                'lat_range': (37.770, 37.790),
                'lon_range': (-122.410, -122.390),
                'price_range': (1800000, 3200000),
                'style': 'Modern/Industrial',
                'year_range': (2000, 2020),
                'sqft_range': (1500, 3000),
                'bedroom_range': (1, 4),
                'bathroom_range': (1, 4),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            },
            'Hayes Valley': {
                'lat_range': (37.770, 37.780),
                'lon_range': (-122.420, -122.400),
                'price_range': (2000000, 3500000),
                'style': 'Victorian/Modern',
                'year_range': (1900, 2010),
                'sqft_range': (2000, 3500),
                'bedroom_range': (2, 4),
                'bathroom_range': (2, 4),
                'walk_score_range': (90, 95),
                'transit_score_range': (85, 90)
            },
            'Castro': {
                'lat_range': (37.750, 37.760),
                'lon_range': (-122.440, -122.420),
                'price_range': (1600000, 2800000),
                'style': 'Victorian',
                'year_range': (1900, 1950),
                'sqft_range': (1800, 3000),
                'bedroom_range': (2, 4),
                'bathroom_range': (1, 3),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            },
            'Noe Valley': {
                'lat_range': (37.750, 37.760),
                'lon_range': (-122.430, -122.410),
                'price_range': (2200000, 3800000),
                'style': 'Victorian',
                'year_range': (1890, 1940),
                'sqft_range': (2000, 3500),
                'bedroom_range': (3, 5),
                'bathroom_range': (2, 4),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            },
            'Haight-Ashbury': {
                'lat_range': (37.760, 37.770),
                'lon_range': (-122.450, -122.430),
                'price_range': (1800000, 3200000),
                'style': 'Victorian',
                'year_range': (1890, 1940),
                'sqft_range': (2000, 3500),
                'bedroom_range': (2, 4),
                'bathroom_range': (2, 3),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            },
            'Outer Sunset': {
                'lat_range': (37.720, 37.740),
                'lon_range': (-122.500, -122.480),
                'price_range': (1200000, 2200000),
                'style': 'Traditional',
                'year_range': (1940, 1980),
                'sqft_range': (1500, 2800),
                'bedroom_range': (2, 4),
                'bathroom_range': (1, 3),
                'walk_score_range': (60, 80),
                'transit_score_range': (55, 75)
            },
            'Inner Sunset': {
                'lat_range': (37.740, 37.760),
                'lon_range': (-122.480, -122.460),
                'price_range': (1500000, 2500000),
                'style': 'Traditional',
                'year_range': (1930, 1970),
                'sqft_range': (1800, 3000),
                'bedroom_range': (2, 4),
                'bathroom_range': (1, 3),
                'walk_score_range': (70, 85),
                'transit_score_range': (65, 80)
            },
            'Laurel Heights': {
                'lat_range': (37.780, 37.790),
                'lon_range': (-122.440, -122.420),
                'price_range': (2000000, 3500000),
                'style': 'Traditional',
                'year_range': (1910, 1950),
                'sqft_range': (2200, 3500),
                'bedroom_range': (3, 5),
                'bathroom_range': (2, 4),
                'walk_score_range': (75, 90),
                'transit_score_range': (70, 85)
            },
            'Western Addition': {
                'lat_range': (37.780, 37.790),
                'lon_range': (-122.420, -122.400),
                'price_range': (1800000, 3000000),
                'style': 'Victorian',
                'year_range': (1900, 1940),
                'sqft_range': (2000, 3200),
                'bedroom_range': (2, 4),
                'bathroom_range': (2, 3),
                'walk_score_range': (80, 90),
                'transit_score_range': (75, 85)
            },
            'Mission Dolores': {
                'lat_range': (37.740, 37.750),
                'lon_range': (-122.420, -122.400),
                'price_range': (1700000, 2800000),
                'style': 'Victorian',
                'year_range': (1880, 1940),
                'sqft_range': (1900, 3000),
                'bedroom_range': (2, 4),
                'bathroom_range': (1, 3),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            },
            'Bernal Heights': {
                'lat_range': (37.740, 37.750),
                'lon_range': (-122.420, -122.400),
                'price_range': (1600000, 2600000),
                'style': 'Victorian',
                'year_range': (1890, 1940),
                'sqft_range': (1800, 2800),
                'bedroom_range': (2, 4),
                'bathroom_range': (1, 3),
                'walk_score_range': (80, 90),
                'transit_score_range': (75, 85)
            },
            'Potrero Hill': {
                'lat_range': (37.750, 37.760),
                'lon_range': (-122.400, -122.380),
                'price_range': (1800000, 3000000),
                'style': 'Victorian/Modern',
                'year_range': (1900, 2010),
                'sqft_range': (2000, 3200),
                'bedroom_range': (2, 4),
                'bathroom_range': (2, 3),
                'walk_score_range': (75, 90),
                'transit_score_range': (70, 85)
            },
            'Dogpatch': {
                'lat_range': (37.760, 37.770),
                'lon_range': (-122.390, -122.370),
                'price_range': (2000000, 3500000),
                'style': 'Modern/Industrial',
                'year_range': (2000, 2020),
                'sqft_range': (1800, 3000),
                'bedroom_range': (2, 4),
                'bathroom_range': (2, 4),
                'walk_score_range': (70, 85),
                'transit_score_range': (65, 80)
            },
            'Russian Hill': {
                'lat_range': (37.790, 37.800),
                'lon_range': (-122.420, -122.400),
                'price_range': (2500000, 4000000),
                'style': 'Victorian/Classic',
                'year_range': (1880, 1940),
                'sqft_range': (2500, 4000),
                'bedroom_range': (3, 5),
                'bathroom_range': (2, 4),
                'walk_score_range': (80, 90),
                'transit_score_range': (75, 85)
            },
            'Nob Hill': {
                'lat_range': (37.790, 37.800),
                'lon_range': (-122.420, -122.400),
                'price_range': (2800000, 4500000),
                'style': 'Victorian/Classic',
                'year_range': (1880, 1940),
                'sqft_range': (2800, 4500),
                'bedroom_range': (3, 6),
                'bathroom_range': (2, 5),
                'walk_score_range': (85, 95),
                'transit_score_range': (80, 90)
            }
        }
        
        # Property features and characteristics
        self.property_features = {
            'heating_types': ['Central Heating', 'Forced Air', 'Radiant', 'Baseboard', 'Wall Unit'],
            'cooling_types': ['Central Air', 'Window Unit', 'None', 'Mini Split', 'Evaporative'],
            'appliances': ['Refrigerator', 'Dishwasher', 'Oven', 'Microwave', 'Washer', 'Dryer', 'Garbage Disposal'],
            'flooring_types': ['Hardwood', 'Carpet', 'Tile', 'Laminate', 'Concrete'],
            'roof_types': ['Asphalt Shingle', 'Tile', 'Metal', 'Slate', 'Wood Shake'],
            'parking_types': ['Garage', 'Carport', 'Street', 'Driveway', 'None']
        }
        
        # School districts in SF
        self.school_districts = [
            'San Francisco Unified School District',
            'San Francisco Archdiocese',
            'Private Schools',
            'Charter Schools'
        ]
    
    def generate_realistic_coordinates(self, neighborhood: str, num_properties: int) -> list:
        """Generate realistic coordinates for a neighborhood"""
        if neighborhood not in self.sf_neighborhoods:
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
    
    def generate_realistic_property_data(self, coord: dict) -> dict:
        """Generate realistic property data based on neighborhood characteristics"""
        neighborhood = coord['neighborhood']
        neighborhood_info = self.sf_neighborhoods[neighborhood]
        
        # Generate realistic data based on neighborhood patterns
        price = random.randint(
            neighborhood_info['price_range'][0],
            neighborhood_info['price_range'][1]
        )
        
        year_built = random.randint(
            neighborhood_info['year_range'][0],
            neighborhood_info['year_range'][1]
        )
        
        square_footage = random.randint(
            neighborhood_info['sqft_range'][0],
            neighborhood_info['sqft_range'][1]
        )
        
        bedrooms = random.randint(
            neighborhood_info['bedroom_range'][0],
            neighborhood_info['bedroom_range'][1]
        )
        
        bathrooms = random.randint(
            max(1, bedrooms - 1),
            min(bedrooms + 1, neighborhood_info['bathroom_range'][1])
        )
        
        # Calculate lot size (typically 2-4x square footage in SF)
        lot_size = int(square_footage * random.uniform(2.0, 4.0))
        
        # Generate realistic features
        heating = random.choice(self.property_features['heating_types'])
        cooling = random.choice(self.property_features['cooling_types'])
        
        # Select appliances (3-6 appliances typically included)
        num_appliances = random.randint(3, 6)
        appliances = random.sample(self.property_features['appliances'], num_appliances)
        
        # Generate property features
        num_features = random.randint(3, 5)
        features = [
            f"{random.choice(self.property_features['flooring_types'])} floors",
            f"{random.choice(['Updated', 'Modern', 'Renovated'])} kitchen",
            f"{random.choice(['Spacious', 'Private', 'Landscaped'])} backyard",
            f"{random.choice(['Fireplace', 'Built-in storage', 'High ceilings'])}",
            f"Close to {random.choice(['public transportation', 'shopping', 'parks', 'restaurants'])}"
        ]
        features = random.sample(features, num_features)
        
        # Generate walk and transit scores
        walk_score = random.randint(
            neighborhood_info['walk_score_range'][0],
            neighborhood_info['walk_score_range'][1]
        )
        transit_score = random.randint(
            neighborhood_info['transit_score_range'][0],
            neighborhood_info['transit_score_range'][1]
        )
        
        # Generate realistic sale history
        last_sold_date = datetime.now() - timedelta(days=random.randint(365, 1825))  # 1-5 years ago
        last_sold_price = int(price * random.uniform(0.7, 0.9))  # 70-90% of current price
        
        # Calculate property tax (1.25% in CA)
        property_tax = int(price * 0.0125)
        
        # Generate parking information
        parking_spaces = random.randint(0, 2)
        parking_type = random.choice(self.property_features['parking_types'])
        
        # Calculate estimated monthly payment
        estimated_monthly_payment = self._calculate_monthly_payment(price, property_tax, 0)
        
        # Generate HOA fees (0 for single family, realistic for condos)
        hoa_fees = 0 if random.random() > 0.3 else random.randint(200, 800)
        
        # Generate school district
        school_district = random.choice(self.school_districts)
        
        # Generate address based on coordinates
        address = self._generate_address(coord['latitude'], coord['longitude'], neighborhood)
        
        return {
            "Address": address,
            "Latitude": coord['latitude'],
            "Longitude": coord['longitude'],
            "Neighborhood": neighborhood,
            "Created_At": datetime.now().strftime("%Y-%m-%d"),
            "Property_Type": "Single Family Home",
            "Style": neighborhood_info['style'],
            "Year_Built": year_built,
            "Square_Footage": square_footage,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "Lot_Size": lot_size,
            "Price": price,
            "Property_Tax": property_tax,
            "HOA_Fees": hoa_fees,
            "Parking_Spaces": parking_spaces,
            "Heating": heating,
            "Cooling": cooling,
            "Appliances": ", ".join(appliances),
            "Features": ", ".join(features),
            "School_District": school_district,
            "Walk_Score": walk_score,
            "Transit_Score": transit_score,
            "Last_Sold_Date": last_sold_date.strftime("%Y-%m-%d"),
            "Last_Sold_Price": last_sold_price,
            "Estimated_Monthly_Payment": estimated_monthly_payment
        }
    
    def _generate_address(self, lat: float, lon: float, neighborhood: str) -> str:
        """Generate realistic address based on coordinates and neighborhood"""
        # Generate street number
        street_number = random.randint(100, 9999)
        
        # Street names based on neighborhood
        street_names = {
            'Mission': ['Mission Street', 'Valencia Street', 'Guerrero Street', 'Sanchez Street', 'Dolores Street'],
            'Financial District': ['Market Street', 'Mission Street', 'Howard Street', 'Folsom Street', 'Harrison Street'],
            'Pacific Heights': ['Pacific Avenue', 'California Street', 'Sacramento Street', 'Jackson Street', 'Washington Street'],
            'Marina': ['Marina Boulevard', 'Chestnut Street', 'Lombard Street', 'Green Street', 'Union Street'],
            'North Beach': ['Columbus Avenue', 'Broadway', 'Grant Avenue', 'Stockton Street', 'Kearny Street'],
            'Chinatown': ['Grant Avenue', 'Stockton Street', 'Kearny Street', 'Clay Street', 'Washington Street'],
            'South of Market': ['Folsom Street', 'Harrison Street', 'Bryant Street', 'Spear Street', 'Main Street'],
            'Hayes Valley': ['Hayes Street', 'Fell Street', 'Oak Street', 'Page Street', 'Haight Street'],
            'Castro': ['Castro Street', '18th Street', 'Market Street', 'Noe Street', 'Church Street'],
            'Noe Valley': ['24th Street', 'Church Street', 'Noe Street', 'Diamond Street', 'Castro Street'],
            'Haight-Ashbury': ['Haight Street', 'Ashbury Street', 'Cole Street', 'Stanyan Street', 'Masonic Avenue'],
            'Outer Sunset': ['Irving Street', '9th Avenue', '19th Avenue', 'Judah Street', 'Taraval Street'],
            'Inner Sunset': ['9th Avenue', 'Irving Street', '19th Avenue', 'Judah Street', 'Taraval Street'],
            'Laurel Heights': ['California Street', 'Sacramento Street', 'Jackson Street', 'Washington Street', 'Pine Street'],
            'Western Addition': ['Geary Boulevard', 'O\'Farrell Street', 'Eddy Street', 'Turk Street', 'Golden Gate Avenue'],
            'Mission Dolores': ['Dolores Street', 'Guerrero Street', 'Sanchez Street', 'Valencia Street', 'Mission Street'],
            'Bernal Heights': ['Cortland Avenue', 'Mission Street', 'Valencia Street', 'Guerrero Street', 'Sanchez Street'],
            'Potrero Hill': ['18th Street', '20th Street', '22nd Street', 'Potrero Avenue', 'Kansas Street'],
            'Dogpatch': ['3rd Street', 'Illinois Street', 'Mariposa Street', '22nd Street', '20th Street'],
            'Russian Hill': ['Chestnut Street', 'Lombard Street', 'Green Street', 'Union Street', 'Filbert Street'],
            'Nob Hill': ['California Street', 'Sacramento Street', 'Jackson Street', 'Washington Street', 'Pine Street']
        }
        
        street_name = random.choice(street_names.get(neighborhood, ['Main Street', 'Oak Street', 'Pine Street']))
        
        return f"{street_number}, {street_name}, {neighborhood}, San Francisco, CA 94102"
    
    def _calculate_monthly_payment(self, price: int, annual_tax: int, hoa_fees: int) -> int:
        """Calculate estimated monthly payment"""
        # Simplified calculation: 30-year fixed at 6.5%, 20% down
        principal = price * 0.8
        monthly_principal_interest = int(principal * 0.0065 / 12)
        monthly_tax = int(annual_tax / 12)
        monthly_insurance = int(price * 0.005 / 12)  # 0.5% annual insurance
        
        return monthly_principal_interest + monthly_tax + monthly_insurance + hoa_fees
    
    def generate_comprehensive_sf_database(self, properties_per_neighborhood: int = 25) -> pd.DataFrame:
        """Generate comprehensive SF property database"""
        print(f"Generating comprehensive SF property database...")
        print(f"Target: {properties_per_neighborhood} properties per neighborhood")
        
        all_properties = []
        
        for neighborhood in self.sf_neighborhoods.keys():
            print(f"Generating {properties_per_neighborhood} properties for {neighborhood}...")
            
            # Generate coordinates
            coords = self.generate_realistic_coordinates(neighborhood, properties_per_neighborhood)
            
            # Generate property data for each coordinate
            for coord in coords:
                property_data = self.generate_realistic_property_data(coord)
                all_properties.append(property_data)
        
        # Create DataFrame
        df = pd.DataFrame(all_properties)
        
        print(f"Generated {len(df)} total properties across {len(self.sf_neighborhoods)} neighborhoods")
        return df
    
    def save_database(self, df: pd.DataFrame, output_file: str = "comprehensive_sf_properties.csv") -> str:
        """Save the comprehensive database"""
        df.to_csv(output_file, index=False)
        print(f"Database saved to: {output_file}")
        return output_file
    
    def create_roi_integration_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create data formatted for ROI integration"""
        # Add ROI-specific columns
        roi_df = df.copy()
        
        # Calculate ROI metrics
        roi_df['ROI'] = roi_df.apply(lambda row: self._calculate_roi(row), axis=1)
        roi_df['Current_Value'] = roi_df['Price']
        roi_df['CountyName'] = 'San Francisco'
        roi_df['State'] = 'CA'
        roi_df['City'] = roi_df['Neighborhood']
        
        # Add additional ROI fields
        roi_df['Price_Per_Sqft'] = roi_df['Price'] / roi_df['Square_Footage']
        roi_df['Bed_Bath_Ratio'] = roi_df['Bedrooms'] / roi_df['Bathrooms']
        roi_df['Lot_Size_Sqft'] = roi_df['Lot_Size']
        
        return roi_df
    
    def _calculate_roi(self, row: pd.Series) -> float:
        """Calculate realistic ROI based on property characteristics"""
        base_roi = 5.0  # Base 5% ROI
        
        # Adjust based on neighborhood
        neighborhood_multipliers = {
            'Pacific Heights': 1.3,
            'Nob Hill': 1.3,
            'Marina': 1.2,
            'Russian Hill': 1.2,
            'Financial District': 1.1,
            'Hayes Valley': 1.1,
            'Noe Valley': 1.0,
            'Mission': 0.9,
            'Castro': 0.9,
            'Haight-Ashbury': 0.9,
            'Bernal Heights': 0.8,
            'Potrero Hill': 0.8,
            'Inner Sunset': 0.7,
            'Outer Sunset': 0.6,
            'Dogpatch': 0.8
        }
        
        multiplier = neighborhood_multipliers.get(row['Neighborhood'], 1.0)
        
        # Adjust based on property age (newer = higher ROI)
        age_factor = 1.0 + (2025 - row['Year_Built']) * 0.001
        
        # Adjust based on walk score (higher = higher ROI)
        walk_factor = 1.0 + (row['Walk_Score'] - 70) * 0.005
        
        # Calculate final ROI
        final_roi = base_roi * multiplier * age_factor * walk_factor
        
        # Add some randomness
        final_roi += random.uniform(-1.0, 1.0)
        
        return round(max(2.0, min(12.0, final_roi)), 2)  # Clamp between 2% and 12%

def main():
    """Main function to generate comprehensive SF database"""
    try:
        # Initialize generator
        generator = ComprehensiveSFPropertyGenerator()
        
        # Generate comprehensive database
        df = generator.generate_comprehensive_sf_database(properties_per_neighborhood=25)
        
        # Save main database
        main_file = generator.save_database(df, "comprehensive_sf_properties.csv")
        
        # Create ROI integration data
        roi_df = generator.create_roi_integration_data(df)
        roi_file = generator.save_database(roi_df, "sf_properties_roi_integration.csv")
        
        print(f"\n✅ Comprehensive SF Property Database Generated Successfully!")
        print(f"📊 Total Properties: {len(df):,}")
        print(f"🏘️ Neighborhoods: {len(generator.sf_neighborhoods)}")
        print(f"📁 Main Database: {main_file}")
        print(f"📁 ROI Integration: {roi_file}")
        
        # Show sample data
        print(f"\n📋 Sample Properties:")
        print(df[['Neighborhood', 'Price', 'Square_Footage', 'Bedrooms', 'Bathrooms', 'ROI']].head(10))
        
        # Show neighborhood summary
        print(f"\n🏘️ Neighborhood Summary:")
        neighborhood_summary = df.groupby('Neighborhood').agg({
            'Price': ['count', 'mean', 'min', 'max'],
            'ROI': 'mean'
        }).round(2)
        print(neighborhood_summary)
        
    except Exception as e:
        print(f"❌ Error generating database: {e}")
        raise

if __name__ == "__main__":
    main()
