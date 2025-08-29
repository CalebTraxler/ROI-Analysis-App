#!/usr/bin/env python3
"""
Startup script for the Enhanced ROI Analysis App
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'pydeck',
        'numpy',
        'geopy',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'Neighborhood_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
        'openstreetmap_properties.py'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        print("Please ensure all data files are in the current directory")
        return False
    
    print("✅ All required data files are present")
    return True

def start_streamlit_app():
    """Start the Streamlit app"""
    try:
        print("🚀 Starting Enhanced ROI Analysis App...")
        print("📱 The app will open in your default web browser")
        print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
        print("\n" + "="*50)
        
        # Start Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ROI_enhanced_clickable.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting app: {e}")
        print("Please check the error message above and try again")

def main():
    """Main function"""
    print("🏠 Enhanced Real Estate ROI Analysis App")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start app due to missing dependencies")
        return
    
    # Check data files
    if not check_data_files():
        print("\n❌ Cannot start app due to missing data files")
        return
    
    print("\n✅ All checks passed! Starting app...")
    
    # Start the app
    start_streamlit_app()

if __name__ == "__main__":
    main()
