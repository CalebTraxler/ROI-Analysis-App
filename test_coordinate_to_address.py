#!/usr/bin/env python3
"""
Test Script for Coordinate to Address Translation

This script tests the coordinate-to-address functionality using reverse geocoding
to ensure we can properly translate coordinates to addresses before generating
synthetic MLS data.
"""

import os
import time
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_coordinate_to_address():
    """Test coordinate to address translation with San Francisco coordinates"""
    
    # Initialize geocoder
    geolocator = Nominatim(user_agent="coordinate_test")
    
    # Test coordinates in San Francisco
    test_coordinates = [
        (37.7749, -122.4194),  # Downtown SF
        (37.7849, -122.4094),  # North Beach
        (37.7649, -122.4094),  # Mission District
        (37.7949, -122.4094),  # Russian Hill
        (37.7549, -122.4094),  # South of Market
        (37.8049, -122.4094),  # Pacific Heights
        (37.7449, -122.4094),  # Potrero Hill
        (37.8149, -122.4094),  # Marina District
        (37.7349, -122.4094),  # Bayview
        (37.8249, -122.4094)   # Presidio
    ]
    
    print("Testing coordinate to address translation for San Francisco locations...")
    print("=" * 80)
    
    successful_translations = 0
    total_coordinates = len(test_coordinates)
    
    for i, (lat, lon) in enumerate(test_coordinates, 1):
        try:
            print(f"\n{i}/{total_coordinates}: Testing coordinates ({lat}, {lon})")
            
            # Add delay to respect rate limits
            if i > 1:
                time.sleep(1)
            
            # Attempt reverse geocoding
            location = geolocator.reverse((lat, lon), timeout=10)
            
            if location:
                address = location.address
                print(f"   ✓ SUCCESS: {address}")
                successful_translations += 1
                
                # Extract neighborhood information
                address_parts = address.split(', ')
                if len(address_parts) >= 3:
                    neighborhood = address_parts[0]
                    city = address_parts[1]
                    state = address_parts[2]
                    print(f"      Neighborhood: {neighborhood}")
                    print(f"      City: {city}")
                    print(f"      State: {state}")
                
            else:
                print(f"   ✗ FAILED: No address found")
                
        except GeocoderTimedOut:
            print(f"   ⚠ TIMEOUT: Geocoding request timed out")
        except Exception as e:
            print(f"   ✗ ERROR: {str(e)}")
    
    print("\n" + "=" * 80)
    print(f"Translation Results:")
    print(f"  Total coordinates tested: {total_coordinates}")
    print(f"  Successful translations: {successful_translations}")
    print(f"  Success rate: {(successful_translations/total_coordinates)*100:.1f}%")
    
    if successful_translations > 0:
        print("\n✓ Coordinate-to-address translation is working!")
        print("  Ready to proceed with synthetic data generation.")
    else:
        print("\n✗ Coordinate-to-address translation failed.")
        print("  Check internet connection and geocoding service availability.")
    
    return successful_translations > 0

def test_specific_sf_locations():
    """Test specific known San Francisco locations"""
    
    print("\n" + "=" * 80)
    print("Testing specific San Francisco landmarks...")
    print("=" * 80)
    
    geolocator = Nominatim(user_agent="sf_landmarks_test")
    
    # Famous SF locations
    sf_landmarks = [
        ("Golden Gate Bridge", 37.8199, -122.4783),
        ("Fisherman's Wharf", 37.8080, -122.4177),
        ("Alcatraz Island", 37.8270, -122.4230),
        ("Chinatown", 37.7941, -122.4075),
        ("Castro District", 37.7605, -122.4352),
        ("Haight-Ashbury", 37.7696, -122.4468),
        ("Mission District", 37.7599, -122.4148),
        ("North Beach", 37.8038, -122.4100),
        ("Pacific Heights", 37.7925, -122.4395),
        ("Marina District", 37.8025, -122.4358)
    ]
    
    for landmark, lat, lon in sf_landmarks:
        try:
            print(f"\nTesting: {landmark}")
            print(f"Coordinates: ({lat}, {lon})")
            
            time.sleep(1)  # Rate limiting
            
            location = geolocator.reverse((lat, lon), timeout=10)
            
            if location:
                address = location.address
                print(f"   Address: {address}")
                
                # Check if it's in San Francisco
                if "San Francisco" in address or "CA" in address:
                    print("   ✓ Confirmed: Location is in San Francisco")
                else:
                    print("   ⚠ Note: Location may not be in San Francisco")
            else:
                print("   ✗ No address found")
                
        except Exception as e:
            print(f"   ✗ Error: {str(e)}")

def main():
    """Main test function"""
    print("Coordinate to Address Translation Test")
    print("Testing reverse geocoding for San Francisco properties")
    print("=" * 80)
    
    try:
        # Test basic coordinate translation
        basic_test_passed = test_coordinate_to_address()
        
        # Test specific SF landmarks
        test_specific_sf_locations()
        
        print("\n" + "=" * 80)
        if basic_test_passed:
            print("🎉 All tests completed successfully!")
            print("The coordinate-to-address translation is working properly.")
            print("You can now proceed with the synthetic data generation.")
        else:
            print("⚠ Some tests failed. Please check the issues above.")
            
    except Exception as e:
        print(f"\n✗ Test execution failed: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
