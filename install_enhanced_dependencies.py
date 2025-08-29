#!/usr/bin/env python3
"""
Enhanced Dependencies Installation Script
Installs all required packages for the enhanced ROI analysis platform
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a single package with error handling"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 Enhanced ROI Analysis Platform - Dependency Installer")
    print("=" * 60)
    
    # Core enhanced dependencies
    enhanced_packages = [
        "osmnx>=1.6.0",
        "geopandas>=0.14.0", 
        "shapely>=2.0.0",
        "folium>=0.15.0",
        "contextily>=1.4.0",
        "cenpy>=1.0.0",
        "overpy>=0.7.0",
        "geocoder>=1.38.1",
        "rtree>=1.1.0",
        "fiona>=1.9.0",
        "pyproj>=3.6.0"
    ]
    
    # Original dependencies (in case they're missing)
    original_packages = [
        "streamlit>=1.28.0",
        "pandas>=2.0.0",
        "pydeck>=0.8.0",
        "numpy>=1.24.0",
        "geopy>=2.3.0",
        "pathlib2>=2.3.7",
        "requests>=2.31.0"
    ]
    
    print("\n📦 Installing enhanced dependencies...")
    print("-" * 40)
    
    success_count = 0
    total_packages = len(enhanced_packages) + len(original_packages)
    
    # Install original packages first
    for package in original_packages:
        if install_package(package):
            success_count += 1
    
    # Install enhanced packages
    for package in enhanced_packages:
        if install_package(package):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Installation Summary:")
    print(f"   Successfully installed: {success_count}/{total_packages} packages")
    
    if success_count == total_packages:
        print("🎉 All dependencies installed successfully!")
        print("\n🚀 You can now run the enhanced ROI analysis platform:")
        print("   streamlit run ROI_optimized.py")
    else:
        print("⚠️ Some packages failed to install. Please check the errors above.")
        print("\n💡 Try installing failed packages manually:")
        print("   pip install <package_name>")
    
    print("\n📚 For more information, see ENHANCED_FEATURES_README.md")

if __name__ == "__main__":
    main()
