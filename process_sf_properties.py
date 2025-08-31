#!/usr/bin/env python3
"""
Process San Francisco Properties with Synthetic Data Generation

This script demonstrates how to integrate the synthetic data generator
with your existing ROI analysis system to process real SF properties.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function to process SF properties"""
    print("🏠 San Francisco Properties Synthetic Data Generator")
    print("=" * 60)
    
    try:
        # Import the synthetic data generator
        from synthetic_data_generator import SyntheticDataGenerator
        
        # Initialize the generator
        print("Initializing synthetic data generator...")
        generator = SyntheticDataGenerator()
        
        # Create sample SF properties (you can replace this with real data from your system)
        print("\nCreating sample San Francisco properties...")
        sf_properties = [
            {
                'osm_id': 'sf_mission_1',
                'latitude': 37.7599,
                'longitude': -122.4148,
                'building_type': 'house'
            },
            {
                'osm_id': 'sf_pacific_heights_1',
                'latitude': 37.7925,
                'longitude': -122.4395,
                'building_type': 'house'
            },
            {
                'osm_id': 'sf_marina_1',
                'latitude': 37.8025,
                'longitude': -122.4358,
                'building_type': 'house'
            },
            {
                'osm_id': 'sf_north_beach_1',
                'latitude': 37.8038,
                'longitude': -122.4100,
                'building_type': 'house'
            },
            {
                'osm_id': 'sf_castro_1',
                'latitude': 37.7605,
                'longitude': -122.4352,
                'building_type': 'house'
            }
        ]
        
        print(f"Created {len(sf_properties)} sample properties")
        
        # Process properties with synthetic data generation
        print(f"\nProcessing {len(sf_properties)} properties...")
        print("This will:")
        print("1. Convert coordinates to addresses")
        print("2. Generate synthetic MLS data using GPT-4o-mini")
        print("3. Store results in database")
        print("4. Export to CSV")
        
        # Process the properties
        generator.process_sf_properties(sf_properties, max_properties=len(sf_properties))
        
        # Get processing statistics
        stats = generator.get_processing_stats()
        print(f"\n📊 Processing Statistics:")
        print(f"  Total properties: {stats['total_properties']}")
        print(f"  By neighborhood: {stats['neighborhood_counts']}")
        print(f"  Recent activity: {stats['recent_properties']}")
        
        # Export to CSV
        print(f"\n📁 Exporting data to CSV...")
        df = generator.export_to_csv("sf_synthetic_mls_data.csv")
        print(f"✓ Exported {len(df)} properties to 'sf_synthetic_mls_data.csv'")
        print(f"✓ Data shape: {df.shape}")
        
        # Show sample of generated data
        print(f"\n🔍 Sample of Generated Data:")
        print("=" * 60)
        
        # Display key columns
        display_columns = ['osm_id', 'neighborhood', 'property_type', 'price', 'bedrooms', 'bathrooms']
        sample_df = df[display_columns].head(3)
        
        for _, row in sample_df.iterrows():
            print(f"Property: {row['osm_id']}")
            print(f"  Neighborhood: {row['neighborhood']}")
            print(f"  Type: {row['property_type']}")
            print(f"  Price: ${row['price']:,}")
            print(f"  Beds/Baths: {row['bedrooms']}/{row['bathrooms']}")
            print()
        
        print("🎉 Successfully processed San Francisco properties!")
        print("\nNext steps:")
        print("1. Review the generated data quality")
        print("2. Adjust prompts if needed")
        print("3. Scale up to process more properties")
        print("4. Integrate with your ML model training")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("  pip3 install -r synthetic_requirements.txt")
        return False
        
    except Exception as e:
        print(f"✗ Error processing properties: {e}")
        return False

def show_integration_example():
    """Show how to integrate with existing ROI system"""
    print("\n" + "=" * 60)
    print("🔗 INTEGRATION WITH EXISTING ROI SYSTEM")
    print("=" * 60)
    
    print("""
# Example: Integrate with your existing OpenStreetMap properties

from openstreetmap_properties import OpenStreetMapProperties
from synthetic_data_generator import SyntheticDataGenerator

# Get existing property data
osm_fetcher = OpenStreetMapProperties()
sf_properties = osm_fetcher.get_city_properties("San Francisco", "San Francisco", "CA")

# Filter for properties with coordinates only
properties_with_coords = []
for prop in sf_properties:
    if 'latitude' in prop and 'longitude' in prop:
        # Check if it's in San Francisco area
        lat, lon = prop['latitude'], prop['longitude']
        if 37.7 <= lat <= 37.8 and -122.5 <= lon <= -122.2:
            properties_with_coords.append(prop)

print(f"Found {len(properties_with_coords)} SF properties with coordinates")

# Generate synthetic data
generator = SyntheticDataGenerator()
generator.process_sf_properties(properties_with_coords, max_properties=100)

# Export for ML training
ml_training_data = generator.export_to_csv("ml_training_data_sf.csv")
print(f"Generated {len(ml_training_data)} synthetic property records")
""")

if __name__ == "__main__":
    success = main()
    
    if success:
        show_integration_example()
    else:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
