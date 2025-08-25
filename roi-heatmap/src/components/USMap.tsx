import React, { useEffect, useState, useMemo } from 'react';
import { MapContainer, TileLayer, GeoJSON, CircleMarker, Popup } from 'react-leaflet';
import { ROIData } from '../types/ROITypes';
import { CoordinateService } from '../services/CoordinateService';
import { useLeafletFix } from '../hooks/useLeafletFix';
import 'leaflet/dist/leaflet.css';
import './USMap.css';

interface USMapProps {
  roiData: ROIData[];
  loading: boolean;
}

interface StateData {
  [key: string]: {
    totalROI: number;
    count: number;
    avgROI: number;
    totalValue: number;
    neighborhoods: string[];
  };
}

interface ProcessedROIData extends ROIData {
  coordinates: [number, number] | null;
  validCoords: boolean;
}

export const USMap: React.FC<USMapProps> = ({ roiData, loading }) => {
  // Fix Leaflet icon issues
  useLeafletFix();
  
  const [usStates, setUsStates] = useState<any>(null);
  const [stateData, setStateData] = useState<StateData>({});
  const [processedROIData, setProcessedROIData] = useState<ProcessedROIData[]>([]);
  const [coordinateService] = useState(() => CoordinateService.getInstance());
  const [coordinatesLoaded, setCoordinatesLoaded] = useState(false);

  // Initialize CoordinateService when component mounts
  useEffect(() => {
    const initCoordinateService = async () => {
      try {
        await coordinateService.loadCoordinates();
        console.log('✅ CoordinateService initialized successfully');
        setCoordinatesLoaded(true);
      } catch (error) {
        console.error('❌ Failed to initialize CoordinateService:', error);
        setCoordinatesLoaded(true); // Still set to true to proceed
      }
    };
    
    initCoordinateService();
  }, [coordinateService]);

  // Load US states GeoJSON data
  useEffect(() => {
    const loadUSStates = async () => {
      console.log('Starting to load US states data...');
      try {
        // Try multiple US states data sources
        const sources = [
          'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json',
          'https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json',
          'https://gist.githubusercontent.com/meiqimichelle/7727723/raw/0109432d22f18fd4afb5c1c4acd3c4930496a5f4/us-states.json'
        ];
        
        let data = null;
        let successfulSource = null;
        
        for (const source of sources) {
          try {
            console.log(`Trying to load from: ${source}`);
            const response = await fetch(source);
            if (response.ok) {
              data = await response.json();
              // Validate the GeoJSON structure
              if (data && (data.type === 'FeatureCollection' || data.type === 'Topology') && 
                  (Array.isArray(data.features) || data.objects)) {
                console.log('Successfully loaded US states from:', source);
                successfulSource = source;
                break;
              } else {
                console.warn(`Invalid data structure from ${source}:`, data);
              }
            } else {
              console.warn(`HTTP error from ${source}:`, response.status);
            }
          } catch (e) {
            console.warn(`Failed to load from ${source}:`, e);
            continue;
          }
        }
        
        // Handle TopoJSON format (like from us-atlas)
        if (data && data.type === 'Topology' && data.objects && data.objects.states) {
          // Convert TopoJSON to GeoJSON (simplified conversion)
          // Note: For production, you'd want to use the topojson library
          console.warn('TopoJSON detected but not fully supported. Using fallback.');
          data = null;
        }
        
        if (data && data.type === 'FeatureCollection') {
          console.log('US States data loaded successfully:', {
            source: successfulSource,
            type: data.type,
            featureCount: data.features?.length,
            sampleFeature: data.features?.[0]?.properties
          });
          setUsStates(data);
        } else {
          console.error('Invalid US states data structure:', data);
          // Create a minimal fallback
          console.log('Creating minimal fallback US outline');
          setUsStates({
            type: "FeatureCollection",
            features: [
              {
                type: "Feature",
                properties: { name: "United States", NAME: "United States" },
                geometry: {
                  type: "Polygon",
                  coordinates: [[[-125, 25], [-65, 25], [-65, 50], [-125, 50], [-125, 25]]]
                }
              }
            ]
          });
        }
      } catch (error) {
        console.error('Failed to load US states data:', error);
        // Create minimal fallback
        setUsStates({
          type: "FeatureCollection",
          features: []
        });
      }
    };

    loadUSStates();
  }, []);

  // Get cached coordinates from the Flask API
  const getCachedCoordinates = async (state: string, county: string): Promise<Map<string, [number, number]>> => {
    try {
      console.log(`🔍 Checking coordinate cache for ${county}, ${state}`);
      const response = await fetch(`http://localhost:5000/api/coordinates/${state}/${county}`);
      
      if (response.ok) {
        const data = await response.json();
        const coordMap = new Map();
        
        Object.entries(data.coordinates).forEach(([neighborhood, coordData]: [string, any]) => {
          coordMap.set(neighborhood, [coordData.latitude, coordData.longitude]);
        });
        
        console.log(`✅ Found ${coordMap.size} cached coordinates for ${county}, ${state}`);
        return coordMap;
      } else if (response.status === 404) {
        console.log(`⚠️ No cached coordinates found for ${county}, ${state}`);
        return new Map();
      } else {
        console.warn(`Cache API error: ${response.status}`);
        return new Map();
      }
    } catch (error) {
      console.warn('Cache API unavailable, falling back to geocoding:', error);
      return new Map();
    }
  };

  // Fallback geocoding function using Nominatim
  const geocodeLocation = async (neighborhood: string, county: string, state: string): Promise<[number, number] | null> => {
    const queries = [
      `${neighborhood}, ${county} County, ${state}, USA`,
      `${neighborhood}, ${county}, ${state}, USA`,
      `${neighborhood}, ${state}, USA`,
      `${county} County, ${state}, USA`, // Fallback to county center
      `${county}, ${state}, USA`
    ];

    for (const query of queries) {
      try {
        console.log(`🌐 Geocoding query: ${query}`);
        
        // Using Nominatim (OpenStreetMap) geocoding service
        const encodedQuery = encodeURIComponent(query);
        const response = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&q=${encodedQuery}&limit=1&countrycodes=us`
        );
        
        if (response.ok) {
          const results = await response.json();
          if (results && results.length > 0) {
            const lat = parseFloat(results[0].lat);
            const lng = parseFloat(results[0].lon);
            
            if (!isNaN(lat) && !isNaN(lng)) {
              console.log(`✅ Geocoded "${query}" to: [${lat}, ${lng}]`);
              return [lat, lng];
            }
          }
        }
        
        // Add delay to respect rate limits
        await new Promise(resolve => setTimeout(resolve, 1000));
        
      } catch (error) {
        console.warn(`Failed to geocode "${query}":`, error);
        continue;
      }
    }
    
    return null;
  };

  // Process ROI data with coordinates when both data and coordinate service are ready
  useEffect(() => {
    if (roiData.length === 0) {
      setProcessedROIData([]);
      return;
    }

    const processData = async () => {
      console.log('=== PROCESSING ROI DATA WITH COORDINATES ===');
      console.log('Total ROI items:', roiData.length);
      
      const processed: ProcessedROIData[] = [];
      
      for (let i = 0; i < roiData.length; i++) {
        const item = roiData[i];
        console.log(`Processing item ${i + 1}/${roiData.length}: ${item.neighborhood} in ${item.county}, ${item.state}`);
        
        let coordinates: [number, number] | null = null;
        let validCoords = false;
        
        try {
          // First, try to use the original coordinates if they're valid
          if (item.latitude && item.longitude) {
            const lat = Number(item.latitude);
            const lng = Number(item.longitude);
            
            if (!isNaN(lat) && !isNaN(lng) && 
                lat >= -90 && lat <= 90 && 
                lng >= -180 && lng <= 180 &&
                lat !== 0 && lng !== 0) { // Avoid 0,0 coordinates
              coordinates = [lat, lng];
              validCoords = true;
              console.log(`✅ Using original coordinates for ${item.neighborhood}: [${lat}, ${lng}]`);
            }
          }
          
          // If coordinates are invalid, try cached coordinates first (FASTEST)
          if (!validCoords) {
            console.log(`🔍 Checking coordinate cache for ${item.neighborhood}`);
            
            try {
              const cachedCoords = await getCachedCoordinates(item.state, item.county);
              if (cachedCoords.has(item.neighborhood)) {
                const [lat, lng] = cachedCoords.get(item.neighborhood)!;
                coordinates = [lat, lng];
                validCoords = true;
                console.log(`✅ Found cached coordinates for ${item.neighborhood}: [${lat}, ${lng}]`);
              }
            } catch (error) {
              console.warn(`Cache lookup failed for ${item.neighborhood}:`, error);
            }
          }
          
          // If still no coordinates, try CoordinateService
          if (!validCoords && coordinatesLoaded) {
            console.log(`🔍 Trying CoordinateService for ${item.neighborhood}`);
            
            try {
              const coordMap = await coordinateService.getCoordinatesForNeighborhoods(
                [item.neighborhood], 
                item.state, 
                item.county
              );
              
              if (coordMap.has(item.neighborhood)) {
                const [lat, lng] = coordMap.get(item.neighborhood)!;
                coordinates = [lat, lng];
                validCoords = true;
                console.log(`✅ CoordinateService found ${item.neighborhood}: [${lat}, ${lng}]`);
              }
            } catch (error) {
              console.warn(`CoordinateService failed for ${item.neighborhood}:`, error);
            }
          }
          
          // If still no coordinates, use direct geocoding (SLOWEST - fallback)
          if (!validCoords) {
            console.log(`🌐 Direct geocoding for ${item.neighborhood}`);
            coordinates = await geocodeLocation(item.neighborhood, item.county, item.state);
            validCoords = coordinates !== null;
            
            if (validCoords && coordinates) {
              console.log(`✅ Direct geocoded ${item.neighborhood}: [${coordinates[0]}, ${coordinates[1]}]`);
            }
          }
        } catch (error) {
          console.error(`❌ Error processing coordinates for ${item.neighborhood}:`, error);
        }
        
        if (!validCoords) {
          console.warn(`⚠️ No coordinates found for ${item.neighborhood} in ${item.county}, ${item.state}`);
        }
        
        processed.push({
          ...item,
          coordinates,
          validCoords
        });
        
        // Add a small delay to avoid overwhelming geocoding services
        if (i < roiData.length - 1) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
      
      const validCount = processed.filter(item => item.validCoords).length;
      console.log(`=== COORDINATE PROCESSING COMPLETE ===`);
      console.log(`Total items: ${processed.length}`);
      console.log(`Valid coordinates: ${validCount}`);
      console.log(`Invalid coordinates: ${processed.length - validCount}`);
      
      if (validCount === 0) {
        console.error('❌ NO VALID COORDINATES FOUND! Check your data or geocoding service.');
      }
      
      setProcessedROIData(processed);
    };
    
    processData();
  }, [roiData, coordinatesLoaded, coordinateService]);

  // Process state data for choropleth visualization
  useEffect(() => {
    if (processedROIData.length === 0) {
      setStateData({});
      return;
    }

    console.log('=== PROCESSING STATE DATA FOR CHOROPLETH ===');
    
    const stateAggregation: StateData = {};
    
    processedROIData.forEach((item) => {
      const state = item.state || 'Unknown';
      
      if (!stateAggregation[state]) {
        stateAggregation[state] = {
          totalROI: 0,
          count: 0,
          avgROI: 0,
          totalValue: 0,
          neighborhoods: []
        };
      }
      
      stateAggregation[state].totalROI += item.roi;
      stateAggregation[state].count += 1;
      stateAggregation[state].totalValue += item.propertyValue;
      stateAggregation[state].neighborhoods.push(item.neighborhood);
    });

    // Calculate averages
    Object.keys(stateAggregation).forEach(state => {
      stateAggregation[state].avgROI = stateAggregation[state].totalROI / stateAggregation[state].count;
    });

    console.log('=== STATE DATA PROCESSED ===');
    console.log('States with data:', Object.keys(stateAggregation));
    Object.entries(stateAggregation).forEach(([state, data]) => {
      console.log(`  ${state}: ${data.count} neighborhoods, avg ROI: ${data.avgROI.toFixed(2)}%`);
    });
    
    setStateData(stateAggregation);
  }, [processedROIData]);

  // Color function for ROI visualization (matching Python implementation)
  const getColorByROI = (roi: number): string => {
    if (roi >= 15) return '#BD0026'; // Deep Red (High ROI)
    if (roi >= 10) return '#E31A1C'; // Red (Good ROI)
    if (roi >= 5) return '#FC4E2A';  // Red-Orange (Moderate ROI)
    if (roi >= 0) return '#FD8D3C';  // Orange (Low ROI)
    return '#FEB24C';                // Light Orange (Negative ROI)
  };

  // Style function for GeoJSON features (states)
  const stateStyle = useMemo(() => {
    return (feature: any) => {
      const stateName = feature.properties.name || feature.properties.NAME || 'Unknown';
      
      // Find matching state data (case-insensitive)
      const matchingState = Object.keys(stateData).find(
        state => state.toLowerCase() === stateName.toLowerCase()
      );
      
      const data = stateData[matchingState || ''];
      
      if (!data) {
        return {
          fillColor: '#f0f0f0',
          weight: 1,
          opacity: 1,
          color: '#666',
          fillOpacity: 0.2
        };
      }

      return {
        fillColor: getColorByROI(data.avgROI),
        weight: 2,
        opacity: 1,
        color: 'white',
        fillOpacity: 0.6
      };
    };
  }, [stateData]);

  // Event handlers for GeoJSON features
  const onEachFeature = (feature: any, layer: any) => {
    const stateName = feature.properties.name || feature.properties.NAME || 'Unknown';
    const data = stateData[stateName] || stateData[Object.keys(stateData).find(
      state => state.toLowerCase() === stateName.toLowerCase()
    ) || ''];
    
    if (data) {
      layer.bindPopup(`
        <div style="font-family: Arial, sans-serif; max-width: 250px;">
          <h3 style="margin: 0 0 8px 0; color: #333;">${stateName}</h3>
          <p style="margin: 4px 0;"><strong>Average ROI:</strong> ${data.avgROI.toFixed(2)}%</p>
          <p style="margin: 4px 0;"><strong>Neighborhoods:</strong> ${data.count}</p>
          <p style="margin: 4px 0;"><strong>Total Value:</strong> $${(data.totalValue / 1000000).toFixed(1)}M</p>
          <p style="margin: 8px 0 4px 0;"><strong>Top Neighborhoods:</strong></p>
          <ul style="margin: 4px 0; padding-left: 16px;">
            ${data.neighborhoods.slice(0, 5).map(n => `<li style="margin: 2px 0;">${n}</li>`).join('')}
            ${data.neighborhoods.length > 5 ? `<li style="margin: 2px 0; font-style: italic;">... and ${data.neighborhoods.length - 5} more</li>` : ''}
          </ul>
        </div>
      `);
    }
  };

  // Generate neighborhood markers (synchronously from processed data)
  const neighborhoodMarkers = useMemo(() => {
    return processedROIData
      .filter(item => item.validCoords && item.coordinates)
      .map((item, index) => {
        const [lat, lng] = item.coordinates!;
        const radius = Math.max(4, Math.min(15, Math.abs(item.roi) / 3 + 5));
        
        return (
          <CircleMarker
            key={`${item.neighborhood}-${index}`}
            center={[lat, lng]}
            radius={radius}
            fillColor={getColorByROI(item.roi)}
            color="white"
            weight={2}
            opacity={0.9}
            fillOpacity={0.8}
          >
            <Popup>
              <div style={{ fontFamily: 'Arial, sans-serif', minWidth: '200px' }}>
                <h4 style={{ margin: '0 0 8px 0', color: '#333' }}>{item.neighborhood}</h4>
                <p style={{ margin: '4px 0' }}><strong>ROI:</strong> {item.roi.toFixed(2)}%</p>
                <p style={{ margin: '4px 0' }}><strong>Property Value:</strong> ${item.propertyValue.toLocaleString()}</p>
                <p style={{ margin: '4px 0' }}><strong>Location:</strong> {item.county}, {item.state}</p>
                <p style={{ margin: '4px 0', fontSize: '12px', color: '#666' }}>
                  Coordinates: {lat.toFixed(4)}, {lng.toFixed(4)}
                </p>
              </div>
            </Popup>
          </CircleMarker>
        );
      });
  }, [processedROIData]);

  // Calculate map center and zoom based on data
  const mapCenter = useMemo((): [number, number] => {
    if (processedROIData.length === 0) {
      return [39.8283, -98.5795]; // Default US center
    }
    
    const validItems = processedROIData.filter(item => item.validCoords && item.coordinates);
    if (validItems.length === 0) {
      return [39.8283, -98.5795];
    }
    
    const avgLat = validItems.reduce((sum, item) => sum + item.coordinates![0], 0) / validItems.length;
    const avgLng = validItems.reduce((sum, item) => sum + item.coordinates![1], 0) / validItems.length;
    
    return [avgLat, avgLng];
  }, [processedROIData]);

  const mapZoom = useMemo(() => {
    if (processedROIData.length === 0) return 4;
    
    const validItems = processedROIData.filter(item => item.validCoords && item.coordinates);
    if (validItems.length <= 1) return 10;
    
    const lats = validItems.map(item => item.coordinates![0]);
    const lngs = validItems.map(item => item.coordinates![1]);
    
    const latRange = Math.max(...lats) - Math.min(...lats);
    const lngRange = Math.max(...lngs) - Math.min(...lngs);
    
    const maxRange = Math.max(latRange, lngRange);
    
    if (maxRange > 20) return 4;  // Country level
    if (maxRange > 5) return 6;   // Multi-state level
    if (maxRange > 1) return 8;   // State level
    if (maxRange > 0.3) return 10; // County level
    return 12; // City level
  }, [processedROIData]);

  if (loading) {
    return (
      <div className="us-map-container">
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">Loading US map data...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="us-map-container">
      <div className="map-header">
        <h3>US Real Estate ROI Investment Map</h3>
        <p>Return on Investment by State and Neighborhood</p>
      </div>
      
      <div className="map-content">
        <MapContainer
          center={mapCenter}
          zoom={mapZoom}
          style={{ height: '600px', width: '100%' }}
          zoomControl={true}
          scrollWheelZoom={true}
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
          
          {/* Render US States with ROI color coding */}
          {usStates && usStates.features && usStates.features.length > 0 && (
            <GeoJSON
              key={`us-states-${Object.keys(stateData).length}`}
              data={usStates}
              style={stateStyle}
              onEachFeature={onEachFeature}
            />
          )}
          
          {/* Render neighborhood markers */}
          {neighborhoodMarkers}
        </MapContainer>
      </div>
      
      <div className="map-legend">
        <div className="legend-title">ROI Performance Scale</div>
        <div className="legend-items">
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#BD0026' }}></span>
            <span>Excellent ROI (≥15%)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#E31A1C' }}></span>
            <span>Good ROI (10-15%)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#FC4E2A' }}></span>
            <span>Moderate ROI (5-10%)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#FD8D3C' }}></span>
            <span>Low ROI (0-5%)</span>
          </div>
          <div className="legend-item">
            <span className="legend-color" style={{ backgroundColor: '#FEB24C' }}></span>
            <span>Negative ROI (&lt;0%)</span>
          </div>
        </div>
      </div>
      
      <div className="map-summary">
        <div className="summary-stats">
          <div className="stat-item">
            <span className="stat-label">States with Data:</span>
            <span className="stat-value">{Object.keys(stateData).length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Total Neighborhoods:</span>
            <span className="stat-value">{processedROIData.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Valid Coordinates:</span>
            <span className="stat-value">{processedROIData.filter(item => item.validCoords).length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Average ROI:</span>
            <span className="stat-value">
              {processedROIData.length > 0 ? 
                (processedROIData.reduce((sum, item) => sum + item.roi, 0) / processedROIData.length).toFixed(1) : '0'}%
            </span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Geographic Spread:</span>
            <span className="stat-value">
              {(() => {
                const validItems = processedROIData.filter(item => item.validCoords && item.coordinates);
                if (validItems.length === 0) return 'No data';
                
                const lats = validItems.map(item => item.coordinates![0]);
                const lngs = validItems.map(item => item.coordinates![1]);
                
                const latRange = Math.max(...lats) - Math.min(...lats);
                const lngRange = Math.max(...lngs) - Math.min(...lngs);
                
                return `${Math.max(latRange, lngRange).toFixed(2)}°`;
              })()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
