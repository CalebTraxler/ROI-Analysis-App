#!/usr/bin/env python3
"""
Coordinate API Server for ROI Analysis

This Flask app serves cached coordinates from the SQLite database
to the React frontend, dramatically improving performance.

Usage:
    python coordinate_api.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sqlite3
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Database path
DB_PATH = 'coordinates_cache.db'

def get_db_connection():
    """Create a database connection"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "database": Path(DB_PATH).exists(),
        "message": "Coordinate API is running"
    })

@app.route('/api/coordinates/<state>/<county>')
def get_coordinates(state, county):
    """Get cached coordinates for a state-county combination"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        
        # Query for coordinates in the specified state and county
        cursor.execute('''
            SELECT location_key, latitude, longitude, timestamp
            FROM coordinates 
            WHERE state = ? AND county = ?
            ORDER BY location_key
        ''', (state, county))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return jsonify({
                "message": f"No cached coordinates found for {county}, {state}",
                "coordinates": {},
                "count": 0
            }), 404
        
        # Parse location_key to extract neighborhood name
        coordinates = {}
        for row in results:
            location_key = row['location_key']
            # Extract neighborhood name from location_key (format: "neighborhood_county_state")
            parts = location_key.split('_')
            if len(parts) >= 3:
                neighborhood = parts[0]  # First part is neighborhood name
                coordinates[neighborhood] = {
                    "latitude": row['latitude'],
                    "longitude": row['longitude'],
                    "timestamp": row['timestamp']
                }
        
        logger.info(f"Found {len(coordinates)} cached coordinates for {county}, {state}")
        
        return jsonify({
            "state": state,
            "county": county,
            "coordinates": coordinates,
            "count": len(coordinates),
            "message": "Coordinates retrieved successfully"
        })
        
    except Exception as e:
        logger.error(f"Error retrieving coordinates for {county}, {state}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/coordinates/states')
def get_states():
    """Get list of all states with cached coordinates"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT state FROM coordinates ORDER BY state')
        states = [row['state'] for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            "states": states,
            "count": len(states)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving states: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/coordinates/counties/<state>')
def get_counties(state):
    """Get list of counties for a specific state"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT county FROM coordinates WHERE state = ? ORDER BY county', (state,))
        counties = [row['county'] for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            "state": state,
            "counties": counties,
            "count": len(counties)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving counties for {state}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/coordinates/stats')
def get_stats():
    """Get overall statistics about cached coordinates"""
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        
        # Total coordinates
        cursor.execute('SELECT COUNT(*) as total FROM coordinates')
        total_coords = cursor.fetchone()['total']
        
        # States count
        cursor.execute('SELECT COUNT(DISTINCT state) as states FROM coordinates')
        states_count = cursor.fetchone()['states']
        
        # Counties count
        cursor.execute('SELECT COUNT(DISTINCT county) as counties FROM coordinates')
        counties_count = cursor.fetchone()['counties']
        
        # Recent activity (last 24 hours)
        cursor.execute('''
            SELECT COUNT(*) as recent 
            FROM coordinates 
            WHERE timestamp > strftime('%s', 'now', '-1 day')
        ''')
        recent_count = cursor.fetchone()['recent']
        
        conn.close()
        
        return jsonify({
            "total_coordinates": total_coords,
            "states": states_count,
            "counties": counties_count,
            "recent_24h": recent_count,
            "database_size_mb": round(Path(DB_PATH).stat().st_size / (1024 * 1024), 2)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/coordinates/search')
def search_coordinates():
    """Search coordinates by neighborhood name"""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400
    
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor()
        
        # Search for neighborhoods containing the query
        cursor.execute('''
            SELECT location_key, latitude, longitude, state, county, timestamp
            FROM coordinates 
            WHERE location_key LIKE ? 
            ORDER BY location_key
            LIMIT 50
        ''', (f'%{query}%',))
        
        results = cursor.fetchall()
        conn.close()
        
        # Parse results
        coordinates = []
        for row in results:
            location_key = row['location_key']
            parts = location_key.split('_')
            if len(parts) >= 3:
                coordinates.append({
                    "neighborhood": parts[0],
                    "county": row['county'],
                    "state": row['state'],
                    "latitude": row['latitude'],
                    "longitude": row['longitude'],
                    "timestamp": row['timestamp']
                })
        
        return jsonify({
            "query": query,
            "results": coordinates,
            "count": len(coordinates)
        })
        
    except Exception as e:
        logger.error(f"Error searching coordinates: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Coordinate API Server...")
    logger.info(f"Database path: {DB_PATH}")
    
    # Check if database exists
    if not Path(DB_PATH).exists():
        logger.error(f"Database file not found: {DB_PATH}")
        logger.error("Please run bulk_load_all_regions.py first to populate the database")
        exit(1)
    
    # Test database connection
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM coordinates')
        total = cursor.fetchone()[0]
        conn.close()
        logger.info(f"Database connected successfully. Total coordinates: {total}")
    else:
        logger.error("Failed to connect to database")
        exit(1)
    
    logger.info("Starting Flask server on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
