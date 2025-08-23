#!/usr/bin/env python3
"""
Cache Pre-population Script for ROI Analysis

This script pre-populates the coordinate cache for all states and counties
to dramatically improve the first-time loading performance of the ROI analysis app.

Usage:
    python prepopulate_cache.py [--states STATE1,STATE2] [--counties COUNTY1,COUNTY2]
    
Examples:
    python prepopulate_cache.py  # Process all states and counties
    python prepopulate_cache.py --states "California,Texas"  # Process specific states
    python prepopulate_cache.py --counties "Los Angeles,Harris"  # Process specific counties
"""

import pandas as pd
import sqlite3
import time
import argparse
import logging
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cache_prepopulation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_coordinates_cache():
    """Initialize the coordinates cache database"""
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS coordinates
                 (location_key TEXT PRIMARY KEY, latitude REAL, longitude REAL, 
                  state TEXT, county TEXT, timestamp REAL)''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_state_county ON coordinates(state, county)')
    conn.commit()
    conn.close()
    logger.info("Coordinates cache database initialized")

def get_cached_location(location_key):
    """Get a cached location from the database"""
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('SELECT latitude, longitude FROM coordinates WHERE location_key = ?', (location_key,))
    result = c.fetchone()
    conn.close()
    return result

def save_location_to_cache(location_key, lat, lon, state, county):
    """Save a location to the cache database"""
    conn = sqlite3.connect('coordinates_cache.db')
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO coordinates 
                 (location_key, latitude, longitude, state, county, timestamp) 
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (location_key, lat, lon, state, county, time.time()))
    conn.commit()
    conn.close()

def geocode_single_location(loc, state, county, geolocator, max_retries=3):
    """Geocode a single location with retry logic"""
    location_key = f"{loc}_{county}_{state}"
    
    # Check if already cached
    cached_result = get_cached_location(location_key)
    if cached_result:
        return loc, cached_result, True  # True indicates it was cached
    
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(f"{loc}, {county} County, {state}, USA", timeout=15)
            if location:
                save_location_to_cache(location_key, location.latitude, location.longitude, state, county)
                return loc, (location.latitude, location.longitude), False  # False indicates it was newly geocoded
            else:
                logger.warning(f"Could not geocode {loc}, {county}, {state}")
                return loc, (None, None), False
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {loc}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to geocode {loc} after {max_retries} attempts")
                return loc, (None, None), False
    
    return loc, (None, None), False

def process_state_county(state, county, max_workers=3):
    """Process all locations for a specific state-county combination"""
    logger.info(f"Processing {county}, {state}")
    
    # Load data for this state-county combination
    df = pd.read_csv('Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    df = df[(df['State'] == state) & (df['CountyName'] == county)]
    
    if len(df) == 0:
        logger.warning(f"No data found for {county}, {state}")
        return 0, 0
    
    unique_locations = df['RegionName'].unique()
    logger.info(f"Found {len(unique_locations)} unique locations in {county}, {state}")
    
    # Initialize geocoder
    geolocator = Nominatim(user_agent="cache_prepopulation_v1")
    
    # Process locations in parallel with rate limiting
    cached_count = 0
    new_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_loc = {executor.submit(geocode_single_location, loc, state, county, geolocator): loc 
                        for loc in unique_locations}
        
        # Process completed tasks with rate limiting
        completed = 0
        for future in as_completed(future_to_loc):
            loc, coords, was_cached = future.result()
            if was_cached:
                cached_count += 1
            else:
                new_count += 1
            
            completed += 1
            
            # Rate limiting: pause every 10 requests
            if completed % 10 == 0:
                time.sleep(1)
                logger.info(f"Progress: {completed}/{len(unique_locations)} locations processed")
    
    logger.info(f"Completed {county}, {state}: {cached_count} cached, {new_count} newly geocoded")
    return cached_count, new_count

def main():
    parser = argparse.ArgumentParser(description='Pre-populate coordinate cache for ROI analysis')
    parser.add_argument('--states', help='Comma-separated list of states to process')
    parser.add_argument('--counties', help='Comma-separated list of counties to process')
    parser.add_argument('--max-workers', type=int, default=3, help='Maximum parallel workers for geocoding')
    
    args = parser.parse_args()
    
    # Initialize cache
    setup_coordinates_cache()
    
    # Load the main dataset
    logger.info("Loading main dataset...")
    df = pd.read_csv('Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
    
    # Determine which states and counties to process
    if args.states:
        target_states = [s.strip() for s in args.states.split(',')]
        df = df[df['State'].isin(target_states)]
        logger.info(f"Processing specific states: {target_states}")
    else:
        target_states = df['State'].unique()
        logger.info(f"Processing all states: {len(target_states)} states found")
    
    if args.counties:
        target_counties = [c.strip() for c in args.counties.split(',')]
        df = df[df['CountyName'].isin(target_counties)]
        logger.info(f"Processing specific counties: {target_counties}")
    
    # Get unique state-county combinations
    state_county_combinations = df[['State', 'CountyName']].drop_duplicates()
    state_county_combinations = state_county_combinations.sort_values(['State', 'CountyName'])
    
    logger.info(f"Found {len(state_county_combinations)} state-county combinations to process")
    
    # Process each combination
    total_cached = 0
    total_new = 0
    start_time = time.time()
    
    for idx, (_, row) in enumerate(state_county_combinations.iterrows()):
        state = row['State']
        county = row['CountyName']
        
        logger.info(f"Processing combination {idx + 1}/{len(state_county_combinations)}: {county}, {state}")
        
        try:
            cached, new = process_state_county(state, county, args.max_workers)
            total_cached += cached
            total_new += new
            
            # Progress update
            elapsed_time = time.time() - start_time
            avg_time_per_combo = elapsed_time / (idx + 1)
            remaining_combos = len(state_county_combinations) - (idx + 1)
            estimated_remaining_time = remaining_combos * avg_time_per_combo
            
            logger.info(f"Progress: {idx + 1}/{len(state_county_combinations)} "
                       f"({(idx + 1)/len(state_county_combinations)*100:.1f}%) "
                       f"Est. remaining time: {estimated_remaining_time/60:.1f} minutes")
            
        except Exception as e:
            logger.error(f"Error processing {county}, {state}: {str(e)}")
            continue
        
        # Small delay between state-county combinations to be respectful to the geocoding service
        time.sleep(2)
    
    total_time = time.time() - start_time
    
    logger.info("=" * 50)
    logger.info("CACHE PRE-POPULATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Total locations processed: {total_cached + total_new}")
    logger.info(f"Locations from cache: {total_cached}")
    logger.info(f"Newly geocoded locations: {total_new}")
    logger.info(f"Average time per state-county combination: {total_time/len(state_county_combinations):.1f} seconds")
    
    # Save summary to file
    summary = {
        'total_time_minutes': total_time / 60,
        'total_combinations': len(state_county_combinations),
        'total_cached': total_cached,
        'total_new': total_new,
        'timestamp': time.time()
    }
    
    with open('cache_prepopulation_summary.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    logger.info("Summary saved to cache_prepopulation_summary.json")

if __name__ == "__main__":
    main()
