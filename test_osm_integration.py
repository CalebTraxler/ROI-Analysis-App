"""
Test script for OpenStreetMap Properties integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openstreetmap_properties import OpenStreetMapProperties
import pandas as pd

def test_osm_integration():
    """Test the OpenStreetMap properties integration"""
    print("Testing OpenStreetMap Properties Integration...")
    
    # Initialize the OSM fetcher
    osm_fetcher = OpenStreetMapProperties()
    
    # Test with a well-known county
    test_county = "Los Angeles"
    test_state = "CA"
    
    print(f"\nTesting with {test_county} County, {test_state}")
    
    try:
        # Test county boundaries
        print("1. Testing county boundaries...")
        bbox = osm_fetcher.get_county_boundaries(test_county, test_state)
        if bbox:
            print(f"   ✓ County boundaries found:")
            print(f"     Lat: {bbox['min_lat']:.4f} to {bbox['max_lat']:.4f}")
            print(f"     Lon: {bbox['min_lon']:.4f} to {bbox['max_lon']:.4f}")
        else:
            print("   ✗ County boundaries not found")
            return False
        
        # Test property fetching (limited to 10 for testing)
        print("\n2. Testing property fetching...")
        properties_df = osm_fetcher.get_county_properties(test_county, test_state, max_properties=10)
        
        if not properties_df.empty:
            print(f"   ✓ Found {len(properties_df)} properties")
            print(f"   ✓ Sample property data:")
            print(f"     Columns: {list(properties_df.columns)}")
            
            # Show first property
            if len(properties_df) > 0:
                first_prop = properties_df.iloc[0]
                print(f"     First property:")
                print(f"       OSM ID: {first_prop.get('osm_id', 'N/A')}")
                print(f"       Type: {first_prop.get('building_type', 'N/A')}")
                print(f"       Coordinates: {first_prop.get('latitude', 'N/A')}, {first_prop.get('longitude', 'N/A')}")
                
                if 'address' in first_prop:
                    addr = first_prop['address']
                    print(f"       Address: {addr.get('street', 'N/A')} {addr.get('housenumber', 'N/A')}")
        else:
            print("   ✗ No properties found")
            return False
        
        # Test property summary
        print("\n3. Testing property summary...")
        summary = osm_fetcher.get_property_summary(properties_df)
        if summary:
            print(f"   ✓ Property summary generated:")
            print(f"     Total properties: {summary.get('total_properties', 0)}")
            print(f"     Avg estimated value: ${summary.get('avg_estimated_value', 0):,.0f}")
            print(f"     Coordinate coverage: {summary.get('coordinate_coverage', 0):.1f}%")
        else:
            print("   ✗ Property summary failed")
            return False
        
        print("\n✅ All tests passed! OpenStreetMap integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_smaller_county():
    """Test with a smaller county that might have fewer properties"""
    print("\n" + "="*50)
    print("Testing with smaller county...")
    
    osm_fetcher = OpenStreetMapProperties()
    
    # Test with a smaller county
    test_county = "Marin"
    test_state = "CA"
    
    print(f"Testing with {test_county} County, {test_state}")
    
    try:
        properties_df = osm_fetcher.get_county_properties(test_county, test_state, max_properties=50)
        
        if not properties_df.empty:
            print(f"✓ Found {len(properties_df)} properties in {test_county} County")
            
            # Show property types
            if 'building_type' in properties_df.columns:
                type_counts = properties_df['building_type'].value_counts()
                print(f"Property types found:")
                for prop_type, count in type_counts.items():
                    print(f"  {prop_type}: {count}")
        else:
            print("✗ No properties found in smaller county")
            
    except Exception as e:
        print(f"✗ Small county test failed: {e}")

if __name__ == "__main__":
    print("OpenStreetMap Properties Integration Test")
    print("=" * 50)
    
    # Run main test
    success = test_osm_integration()
    
    if success:
        # Run additional test with smaller county
        test_smaller_county()
    
    print("\n" + "="*50)
    if success:
        print("🎉 Integration test completed successfully!")
        print("You can now run your Streamlit app with OpenStreetMap properties.")
    else:
        print("💥 Integration test failed. Check the errors above.")
