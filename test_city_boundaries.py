#!/usr/bin/env python3
"""
Test script for city boundary and house loading functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_data_sources import EnhancedRealEstateDataFetcher
    print("✅ Successfully imported EnhancedRealEstateDataFetcher")
except ImportError as e:
    print(f"❌ Failed to import EnhancedRealEstateDataFetcher: {e}")
    print("Please install the required dependencies first:")
    print("python install_enhanced_dependencies.py")
    sys.exit(1)

def test_city_functionality():
    """Test city boundary and house loading functionality"""
    print("\n🚀 Testing City Boundary and House Loading Functionality")
    print("=" * 60)
    
    # Initialize the fetcher
    fetcher = EnhancedRealEstateDataFetcher()
    
    # Test cities
    test_cities = [
        ("San Francisco", "CA"),
        ("Austin", "TX"),
        ("Miami", "FL"),
        ("Seattle", "WA")
    ]
    
    for city_name, state in test_cities:
        print(f"\n🏙️ Testing {city_name}, {state}")
        print("-" * 40)
        
        try:
            # Test city boundary
            print("📐 Loading city boundary...")
            city_boundary = fetcher.get_city_boundary(city_name, state)
            
            if city_boundary:
                area_sqkm = city_boundary.area
                area_sqmi = area_sqkm * 0.386102
                print(f"✅ City boundary loaded: {area_sqmi:.1f} sq mi ({area_sqkm:.1f} sq km)")
                
                # Test city houses
                print("🏠 Loading city houses...")
                houses = fetcher.get_houses_within_city(city_name, state, max_houses=100)
                
                if houses:
                    print(f"✅ City houses loaded: {len(houses)} properties")
                    
                    # Show sample properties
                    for i, house in enumerate(houses[:3]):
                        print(f"   {i+1}. {house.address} - {house.building_type} ({house.area_sqft:.0f} sq ft)")
                    
                    if len(houses) > 3:
                        print(f"   ... and {len(houses) - 3} more properties")
                else:
                    print("⚠️ No houses found in city")
                
                # Test city amenities
                print("🏪 Loading city amenities...")
                amenities = fetcher.get_city_amenities(city_name, state)
                
                if amenities:
                    total_amenities = sum(len(amenities.get(cat, [])) for cat in amenities.keys())
                    print(f"✅ City amenities loaded: {total_amenities} total amenities")
                    
                    # Show amenity breakdown
                    for category, category_amenities in amenities.items():
                        if category_amenities:
                            print(f"   • {category.title()}: {len(category_amenities)}")
                else:
                    print("⚠️ No amenities found in city")
                
            else:
                print("❌ Failed to load city boundary")
                
        except Exception as e:
            print(f"❌ Error testing {city_name}: {e}")
        
        print()

def test_map_creation():
    """Test city map creation with boundaries"""
    print("\n🗺️ Testing City Map Creation")
    print("=" * 40)
    
    fetcher = EnhancedRealEstateDataFetcher()
    
    try:
        # Test with San Francisco
        city_name, state = "San Francisco", "CA"
        
        print(f"Creating city map for {city_name}, {state}...")
        
        # Get city data
        city_boundary = fetcher.get_city_boundary(city_name, state)
        houses = fetcher.get_houses_within_city(city_name, state, max_houses=50)
        amenities = fetcher.get_city_amenities(city_name, state)
        
        if city_boundary:
            # Create city map
            city_map = fetcher.create_city_map_with_boundaries(
                city_name, state, houses, amenities
            )
            
            if city_map:
                print("✅ City map created successfully!")
                print(f"   • City boundary: {city_boundary.area * 0.386102:.1f} sq mi")
                print(f"   • Properties: {len(houses) if houses else 0}")
                print(f"   • Amenities: {sum(len(amenities.get(cat, [])) for cat in amenities.keys()) if amenities else 0}")
            else:
                print("❌ Failed to create city map")
        else:
            print("❌ City boundary not available for map creation")
            
    except Exception as e:
        print(f"❌ Error testing map creation: {e}")

if __name__ == "__main__":
    print("🧪 Enhanced Data Sources Test Suite")
    print("=" * 50)
    
    # Test city functionality
    test_city_functionality()
    
    # Test map creation
    test_map_creation()
    
    print("\n✅ Test suite completed!")
    print("\n💡 If you see any errors, make sure to:")
    print("   1. Install dependencies: python install_enhanced_dependencies.py")
    print("   2. Check your internet connection")
    print("   3. Verify the enhanced_data_sources.py file is present")
