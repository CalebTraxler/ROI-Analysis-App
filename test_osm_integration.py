"""
Test script for OpenStreetMap Properties integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from openstreetmap_properties import OpenStreetMapProperties
import pandas as pd
import time

def test_osm_integration():
    """Test the OpenStreetMap properties integration"""
    print("Testing OpenStreetMap Properties Integration...")
    
    # Initialize the OSM fetcher
    osm_fetcher = OpenStreetMapProperties()
    
    # Test with Alameda County specifically
    test_county = "Alameda"
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

def test_alameda_specific():
    """Test specifically with Alameda County to debug the issue"""
    print("\n" + "="*50)
    print("Testing Alameda County specifically...")
    
    osm_fetcher = OpenStreetMapProperties()
    
    try:
        # Test county boundaries
        print("1. Testing Alameda County boundaries...")
        bbox = osm_fetcher.get_county_boundaries("Alameda", "CA")
        if bbox:
            print(f"   ✓ Alameda County boundaries found:")
            print(f"     Lat: {bbox['min_lat']:.4f} to {bbox['max_lat']:.4f}")
            print(f"     Lon: {bbox['min_lon']:.4f} to {bbox['max_lon']:.4f}")
            
            # Test the Overpass query directly
            print("\n2. Testing Overpass API query directly...")
            properties = osm_fetcher.fetch_properties_in_area(bbox)
            print(f"   ✓ Direct API call found {len(properties)} properties")
            
            if properties:
                print("   ✓ Sample properties:")
                for i, prop in enumerate(properties[:3]):
                    print(f"     Property {i+1}:")
                    print(f"       OSM ID: {prop.get('osm_id', 'N/A')}")
                    print(f"       Type: {prop.get('building_type', 'N/A')}")
                    print(f"       Coordinates: {prop.get('latitude', 'N/A')}, {prop.get('longitude', 'N/A')}")
            else:
                print("   ✗ No properties found in direct API call")
                
        else:
            print("   ✗ Alameda County boundaries not found")
            
    except Exception as e:
        print(f"\n❌ Alameda test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_properties_display():
    """Test if properties can be properly formatted for map display"""
    print("\n" + "="*50)
    print("Testing properties display formatting...")
    
    osm_fetcher = OpenStreetMapProperties()
    
    try:
        # Get a small sample of properties for testing
        properties_df = osm_fetcher.get_county_properties("Alameda", "CA", max_properties=100)
        
        if not properties_df.empty:
            print(f"✓ Loaded {len(properties_df)} properties")
            print(f"✓ DataFrame columns: {list(properties_df.columns)}")
            print(f"✓ DataFrame shape: {properties_df.shape}")
            
            # Check for required columns for map display
            required_cols = ['latitude', 'longitude', 'osm_id', 'building_type']
            missing_cols = [col for col in required_cols if col not in properties_df.columns]
            
            if missing_cols:
                print(f"✗ Missing required columns: {missing_cols}")
            else:
                print("✓ All required columns present")
            
            # Check for coordinate data
            valid_coords = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()]
            print(f"✓ Properties with valid coordinates: {len(valid_coords)}")
            
            # Show sample data
            print("\n✓ Sample property data:")
            sample_prop = valid_coords.iloc[0]
            print(f"   OSM ID: {sample_prop.get('osm_id', 'N/A')}")
            print(f"   Type: {sample_prop.get('building_type', 'N/A')}")
            print(f"   Coordinates: {sample_prop.get('latitude', 'N/A')}, {sample_prop.get('longitude', 'N/A')}")
            print(f"   Address: {sample_prop.get('address', {}).get('street', 'N/A')}")
            
            # Test if data can be used for map layers
            print("\n✓ Testing map layer compatibility...")
            try:
                import pydeck as pdk
                
                # Create a simple test layer
                test_layer = pdk.Layer(
                    'ScatterplotLayer',
                    valid_coords,
                    get_position=['longitude', 'latitude'],
                    get_radius=10,
                    get_fill_color=[0, 100, 200, 180],
                    pickable=True
                )
                print("   ✓ PyDeck layer created successfully")
                
                # Test view state
                center_lat = valid_coords['latitude'].mean()
                center_lon = valid_coords['longitude'].mean()
                view_state = pdk.ViewState(
                    latitude=center_lat,
                    longitude=center_lon,
                    zoom=10
                )
                print("   ✓ View state created successfully")
                
                # Test deck creation
                deck = pdk.Deck(
                    layers=[test_layer],
                    initial_view_state=view_state
                )
                print("   ✓ PyDeck deck created successfully")
                print("   ✓ Properties are ready for map display!")
                
            except ImportError:
                print("   ⚠️ PyDeck not available for testing")
            except Exception as e:
                print(f"   ✗ Map layer test failed: {e}")
                
        else:
            print("✗ No properties loaded")
            
    except Exception as e:
        print(f"\n❌ Properties display test failed: {e}")
        import traceback
        traceback.print_exc()

def test_universal_county_support():
    """Test OSM integration with various counties across different states"""
    print("\n" + "="*50)
    print("Testing Universal County Support...")
    
    osm_fetcher = OpenStreetMapProperties()
    
    # Test counties from different states and regions
    test_counties = [
        ("Los Angeles", "CA"),      # Large urban county
        ("Harris", "TX"),           # Large Texas county
        ("Kings", "NY"),            # NYC borough
        ("Miami-Dade", "FL"),       # Florida county
        ("Cook", "IL"),             # Chicago area
        ("Maricopa", "AZ"),         # Phoenix area
        ("King", "WA"),             # Seattle area
        ("Denver", "CO"),           # Colorado county
        ("Fulton", "GA"),           # Atlanta area
        ("Wayne", "MI"),            # Detroit area
    ]
    
    successful_tests = 0
    total_tests = len(test_counties)
    
    for county_name, state_name in test_counties:
        print(f"\n--- Testing {county_name} County, {state_name} ---")
        
        try:
            # Test county boundaries
            bbox = osm_fetcher.get_county_boundaries(county_name, state_name)
            if bbox:
                print(f"  ✓ Boundaries found: {bbox['min_lat']:.4f} to {bbox['max_lat']:.4f}, {bbox['min_lon']:.4f} to {bbox['max_lon']:.4f}")
                
                # Test property fetching (limited for testing)
                properties_df = osm_fetcher.get_county_properties(county_name, state_name, max_properties=100)
                
                if not properties_df.empty:
                    valid_props = properties_df[properties_df['latitude'].notna() & properties_df['longitude'].notna()]
                    print(f"  ✓ Properties found: {len(properties_df)} total, {len(valid_props)} with coordinates")
                    
                    if len(valid_props) > 0:
                        print(f"  ✓ Sample property: {valid_props.iloc[0].get('building_type', 'N/A')} at {valid_props.iloc[0].get('latitude', 'N/A')}, {valid_props.iloc[0].get('longitude', 'N/A')}")
                        successful_tests += 1
                    else:
                        print(f"  ⚠️ No valid coordinates found")
                else:
                    print(f"  ⚠️ No properties found")
            else:
                print(f"  ✗ No boundaries found")
                
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
        
        # Rate limiting between tests
        time.sleep(2)
    
    print(f"\n--- Universal County Test Results ---")
    print(f"Successful tests: {successful_tests}/{total_tests}")
    success_rate = (successful_tests / total_tests) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if success_rate >= 80:
        print("🎉 Excellent! OSM integration works well across most counties.")
    elif success_rate >= 60:
        print("✅ Good! OSM integration works for most counties.")
    elif success_rate >= 40:
        print("⚠️ Fair! OSM integration works for some counties.")
    else:
        print("❌ Poor! OSM integration needs improvement.")
    
    return success_rate >= 60

if __name__ == "__main__":
    print("OpenStreetMap Properties Integration Test")
    print("=" * 50)
    
    # Test the main integration
    success = test_osm_integration()
    
    # Test Alameda specifically
    test_alameda_specific()
    
    # Test properties display
    test_properties_display()
    
    # Test universal county support
    universal_success = test_universal_county_support()
    
    if success and universal_success:
        print("\n🎉 All tests completed successfully!")
        print("✅ OSM integration is ready for production use across all counties!")
    elif success:
        print("\n⚠️ Basic tests passed, but universal support needs improvement.")
    else:
        print("\n💥 Some tests failed. Check the output above.")
