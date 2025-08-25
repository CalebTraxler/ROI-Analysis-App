#!/usr/bin/env python3
"""
Bulk Load All Regions Script for ROI Analysis

This script loads and caches ALL regions at once with a beautiful progress bar,
giving you maximum performance immediately without waiting for natural cache building.

Usage:
    python bulk_load_all_regions.py
"""

import pandas as pd
import sqlite3
import time
import logging
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import sys
from tqdm import tqdm
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bulk_load_all_regions.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BulkRegionLoader:
    def __init__(self):
        self.total_locations = 0
        self.processed_locations = 0
        self.cached_locations = 0
        self.new_locations = 0
        self.start_time = None
        self.lock = threading.Lock()
        
    def setup_coordinates_cache(self):
        """Initialize the coordinates cache database"""
        conn = sqlite3.connect('coordinates_cache.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS coordinates
                     (location_key TEXT PRIMARY KEY, latitude, longitude REAL, 
                      state TEXT, county TEXT, timestamp REAL)''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_state_county ON coordinates(state, county)')
        conn.commit()
        conn.close()
        logger.info("Coordinates cache database initialized")
    
    def get_cached_location(self, location_key):
        """Get a cached location from the database"""
        conn = sqlite3.connect('coordinates_cache.db')
        c = conn.cursor()
        c.execute('SELECT latitude, longitude FROM coordinates WHERE location_key = ?', (location_key,))
        result = c.fetchone()
        conn.close()
        return result
    
    def save_location_to_cache(self, location_key, lat, lon, state, county):
        """Save a location to the cache database"""
        conn = sqlite3.connect('coordinates_cache.db')
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO coordinates 
                     (location_key, latitude, longitude, state, county, timestamp) 
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (location_key, lat, lon, state, county, time.time()))
        conn.commit()
        conn.close()
    
    def geocode_single_location(self, loc, state, county, geolocator, max_retries=3):
        """Geocode a single location with retry logic"""
        location_key = f"{loc}_{county}_{state}"
        
        # Check if already cached
        cached_result = self.get_cached_location(location_key)
        if cached_result:
            with self.lock:
                self.cached_locations += 1
            return loc, cached_result, True  # True indicates it was cached
        
        for attempt in range(max_retries):
            try:
                location = geolocator.geocode(f"{loc}, {county} County, {state}, USA", timeout=15)
                if location:
                    self.save_location_to_cache(location_key, location.latitude, location.longitude, state, county)
                    with self.lock:
                        self.new_locations += 1
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
    
    def process_state_county_batch(self, state_county_batch, max_workers=5):
        """Process a batch of state-county combinations"""
        state, county = state_county_batch
        
        # Load data for this state-county combination
        df = pd.read_csv('Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
        df = df[(df['State'] == state) & (df['CountyName'] == county)]
        
        if len(df) == 0:
            return 0, 0
        
        unique_locations = df['RegionName'].unique()
        
        # Initialize geocoder
        geolocator = Nominatim(user_agent="bulk_loader_v1")
        
        # Process locations in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_loc = {executor.submit(self.geocode_single_location, loc, state, county, geolocator): loc 
                            for loc in unique_locations}
            
            # Process completed tasks
            for future in as_completed(future_to_loc):
                loc, coords, was_cached = future.result()
                with self.lock:
                    self.processed_locations += 1
                
                # Rate limiting: pause every 20 requests
                if self.processed_locations % 20 == 0:
                    time.sleep(1)
        
        return len(unique_locations), 0
    
    def load_all_regions(self, max_workers=5):
        """Load and cache ALL regions with progress bar"""
        logger.info("🚀 Starting bulk load of ALL regions...")
        
        # Initialize cache
        self.setup_coordinates_cache()
        
        # Load the main dataset
        logger.info("📊 Loading main dataset...")
        df = pd.read_csv('Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
        
        # Get all unique state-county combinations
        state_county_combinations = df[['State', 'CountyName']].drop_duplicates()
        state_county_combinations = state_county_combinations.sort_values(['State', 'CountyName'])
        
        # Calculate total locations
        total_locations = 0
        for _, row in state_county_combinations.iterrows():
            state_data = df[(df['State'] == row['State']) & (df['CountyName'] == row['CountyName'])]
            total_locations += len(state_data['RegionName'].unique())
        
        self.total_locations = total_locations
        logger.info(f"📈 Found {len(state_county_combinations)} state-county combinations with {total_locations} total locations")
        
        # Start timing
        self.start_time = time.time()
        
        # Process with progress bar
        with tqdm(total=len(state_county_combinations), 
                 desc="🌍 Loading All Regions", 
                 unit="county",
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            
            for idx, (_, row) in enumerate(state_county_combinations.iterrows()):
                state = row['State']
                county = row['CountyName']
                
                # Update progress bar description
                pbar.set_description(f"🌍 Loading {county}, {state}")
                
                try:
                    # Process this state-county combination
                    locations_count, _ = self.process_state_county_batch((state, county), max_workers)
                    
                    # Update progress
                    pbar.set_postfix({
                        'Processed': f"{self.processed_locations}/{total_locations}",
                        'Cached': self.cached_locations,
                        'New': self.new_locations,
                        'ETA': self.calculate_eta(idx, len(state_county_combinations))
                    })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Error processing {county}, {state}: {str(e)}")
                    pbar.update(1)
                    continue
                
                # Small delay between state-county combinations
                time.sleep(1)
        
        # Final summary
        total_time = time.time() - self.start_time
        self.print_final_summary(total_time, len(state_county_combinations))
        
        return {
            'total_time_minutes': total_time / 60,
            'total_combinations': len(state_county_combinations),
            'total_locations': total_locations,
            'cached_locations': self.cached_locations,
            'new_locations': self.new_locations,
            'timestamp': time.time()
        }
    
    def calculate_eta(self, current_idx, total_count):
        """Calculate estimated time remaining"""
        if current_idx == 0:
            return "Calculating..."
        
        elapsed_time = time.time() - self.start_time
        avg_time_per_combo = elapsed_time / (current_idx + 1)
        remaining_combos = total_count - (current_idx + 1)
        estimated_remaining_time = remaining_combos * avg_time_per_combo
        
        if estimated_remaining_time < 60:
            return f"{estimated_remaining_time:.0f}s"
        elif estimated_remaining_time < 3600:
            return f"{estimated_remaining_time/60:.1f}m"
        else:
            return f"{estimated_remaining_time/3600:.1f}h"
    
    def print_final_summary(self, total_time, total_combinations):
        """Print final summary with beautiful formatting"""
        print("\n" + "="*80)
        print("🎉 BULK LOAD COMPLETE! 🎉")
        print("="*80)
        print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"🌍 Total Counties: {total_combinations}")
        print(f"📍 Total Locations: {self.total_locations}")
        print(f"💾 Locations from Cache: {self.cached_locations}")
        print(f"🆕 Newly Geocoded: {self.new_locations}")
        print(f"⚡ Average Time per County: {total_time/total_combinations:.1f} seconds")
        print(f"🚀 Performance: All regions now load INSTANTLY!")
        print("="*80)
        
        # Save summary to file
        summary = {
            'total_time_minutes': total_time / 60,
            'total_combinations': total_combinations,
            'total_locations': self.total_locations,
            'cached_locations': self.cached_locations,
            'new_locations': self.new_locations,
            'timestamp': time.time()
        }
        
        with open('bulk_load_summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)
        
        logger.info("📄 Summary saved to bulk_load_summary.json")

def main():
    """Main function to run bulk loading"""
    print("🚀 Starting Bulk Load of ALL Regions...")
    print("This will load and cache every single region for maximum performance!")
    print("Estimated time: 1-3 hours depending on your internet speed")
    print("\n" + "="*80)
    
    # Check if tqdm is installed
    try:
        import tqdm
    except ImportError:
        print("❌ Error: tqdm not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
        print("✅ tqdm installed successfully!")
    
    loader = BulkRegionLoader()
    
    try:
        # Run bulk loading
        results = loader.load_all_regions(max_workers=5)
        
        print(f"\n✅ Bulk loading completed successfully!")
        print(f"📄 Detailed summary saved to: bulk_load_summary.json")
        print(f"🎯 Your ROI app will now be LIGHTNING FAST for ALL regions!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Bulk loading interrupted by user")
        print("💡 Partial cache has been saved. You can resume later.")
    except Exception as e:
        logger.error(f"Bulk loading failed: {e}")
        print(f"❌ Bulk loading failed: {e}")

if __name__ == "__main__":
    main()

