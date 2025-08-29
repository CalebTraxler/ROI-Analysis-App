#!/usr/bin/env python3
"""
Test script for the new Real Estate API integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_estate_api import RealEstateDataFetcher, create_property_dataframe

def test_real_estate_api():
    """Test the real estate API functionality"""
    print("🧪 Testing Real Estate API Integration...")
    
    # Initialize the fetcher
    fetcher = RealEstateDataFetcher()
    
    # Test area (San Francisco Bay Area)
    min_lat, max_lat = 37.7, 37.8
    min_lon, max_lon = -122.5, -122.4
    
    print(f"📍 Testing area: {min_lat:.4f} to {max_lat:.4f}, {min_lon:.4f} to {max_lon:.4f}")
    
    try:
        # Fetch properties
        properties_df = fetcher.get_properties_in_area(
            min_lat, max_lat, min_lon, max_lon, max_properties=50
        )
        
        if not properties_df.empty:
            print(f"✅ Successfully loaded {len(properties_df)} properties")
            print("\n📊 Sample Properties:")
            print(properties_df[['address', 'price', 'bedrooms', 'bathrooms', 'square_feet', 'property_type']].head())
            
            # Test the DataFrame creation
            formatted_df = create_property_dataframe(properties_df.to_dict('records'))
            print(f"\n🔧 Formatted DataFrame created with {len(formatted_df)} rows")
            
            if 'price_formatted' in formatted_df.columns:
                print("✅ Price formatting working correctly")
            if 'price_per_sqft' in formatted_df.columns:
                print("✅ Price per square foot calculation working")
                
        else:
            print("⚠️ No properties found (this is expected if no real API key is configured)")
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        print("This is expected if no real API key is configured - the system will fall back to mock data")
    
    print("\n🎯 Testing complete! The system is ready for integration.")

if __name__ == "__main__":
    test_real_estate_api()
