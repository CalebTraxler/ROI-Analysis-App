import React, { useEffect, useState, useMemo, useRef } from 'react';
import { DeckGL } from '@deck.gl/react';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import { ScatterplotLayer } from '@deck.gl/layers';
import { ROIData } from '../types/ROITypes';
import './ROIMap3D.css';

interface ROIMap3DProps {
  roiData: ROIData[];
  loading: boolean;
}

export const ROIMap3D: React.FC<ROIMap3DProps> = ({ roiData, loading }) => {
  const [viewState, setViewState] = useState({
    longitude: -98.5795,
    latitude: 39.8283,
    zoom: 4,
    pitch: 45,
    bearing: 0
  });

  // Color function matching Python implementation exactly
  const getColorByROI = (roi: number, minROI: number, maxROI: number): [number, number, number, number] => {
    if (maxROI === minROI) {
      return [255, 70, 0, 180]; // Default orange if all ROI values are the same
    }
    const normalized = (roi - minROI) / (maxROI - minROI);
    return [
      255, // R
      Math.max(0, Math.min(255, Math.round(140 * (1 - normalized)))),
      0,   // B
      180  // Alpha (transparency)
    ];
  };

  // Calculate view state based on data (matching Python logic)
  useEffect(() => {
    if (roiData.length > 0) {
      const avgLat = roiData.reduce((sum, item) => sum + item.latitude, 0) / roiData.length;
      const avgLon = roiData.reduce((sum, item) => sum + item.longitude, 0) / roiData.length;
      
      const latSpread = Math.max(...roiData.map(item => item.latitude)) - Math.min(...roiData.map(item => item.latitude));
      const lonSpread = Math.max(...roiData.map(item => item.longitude)) - Math.min(...roiData.map(item => item.longitude));
      const maxSpread = Math.max(latSpread, lonSpread);
      const calculatedZoom = Math.min(20 / maxSpread, 12);
      
      setViewState(prev => ({
        ...prev,
        longitude: avgLon,
        latitude: avgLat,
        zoom: calculatedZoom,
        pitch: 45 // Matching Python pitch
      }));
    }
  }, [roiData]);

  // Prepare data for deck.gl layers (matching Python implementation)
  const layers = useMemo(() => {
    if (roiData.length === 0) return [];

    const minROI = Math.min(...roiData.map(item => item.roi));
    const maxROI = Math.max(...roiData.map(item => item.roi));

    const mapData = roiData.map(item => ({
      position: [item.longitude, item.latitude],
      roi: item.roi,
      propertyValue: item.propertyValue,
      neighborhood: item.neighborhood,
      weightedROI: Math.exp(item.roi / 50) - 1, // Exponential scaling like Python
      color: getColorByROI(item.roi, minROI, maxROI),
      tooltipText: `${item.neighborhood}<br/>$${item.propertyValue.toLocaleString()}<br/>ROI: ${item.roi.toFixed(2)}%`
    }));

    return [
      // Heatmap layer showing ROI density (matching Python exactly)
      new HeatmapLayer({
        id: 'roi-heatmap',
        data: mapData,
        getPosition: (d: any) => d.position,
        getWeight: (d: any) => d.weightedROI,
        radiusPixels: 60, // Matching Python radiusPixels
        intensity: 2,     // Matching Python intensity
        threshold: 0.02,  // Matching Python threshold
        colorRange: [
          [255, 255, 178, 100],  // Light yellow (matching Python)
          [254, 204, 92, 150],   // Yellow (matching Python)
          [253, 141, 60, 200],   // Orange (matching Python)
          [240, 59, 32, 250],    // Red-Orange (matching Python)
          [189, 0, 38, 255]      // Deep Red (matching Python)
        ],
        pickable: false
      }),

      // Scatterplot layer for individual properties with tooltips (matching Python)
      new ScatterplotLayer({
        id: 'roi-scatterplot',
        data: mapData,
        getPosition: (d: any) => d.position,
        getRadius: 30, // Matching Python get_radius
        getFillColor: (d: any) => d.color,
        getLineColor: [255, 255, 255],
        getLineWidth: 1,
        pickable: true,
        opacity: 0.8, // Matching Python opacity
        stroked: true,
        filled: true
      })
    ];
  }, [roiData]);

  if (loading) {
    return (
      <div className="map-3d-container">
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <div className="loading-text">Analyzing ROI data...</div>
        </div>
      </div>
    );
  }

  if (roiData.length === 0) {
    return (
      <div className="map-3d-container">
        <div className="no-data-overlay">
          <div className="no-data-icon">🗺️</div>
          <div className="no-data-title">3D ROI Investment Heatmap</div>
          <div className="no-data-subtitle">
            Select a location and load data to view the 3D ROI heatmap
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="map-3d-container">
      <div className="map-3d-header">
        <div className="map-3d-title-section">
          <h3 className="map-3d-title">3D ROI Investment Heatmap</h3>
          <p className="map-3d-subtitle">Real Zillow Data Visualization (PyDeck Style)</p>
        </div>
        
        <div className="legend-container">
          <div className="legend-title">ROI Performance Heatmap</div>
          <div className="legend-grid">
            <div className="legend-item">
              <span className="legend-color high-roi"></span>
              <span className="legend-label">High ROI (≥15%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color good-roi"></span>
              <span className="legend-label">Good ROI (10-15%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color moderate-roi"></span>
              <span className="legend-label">Moderate ROI (5-10%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color low-roi"></span>
              <span className="legend-label">Low ROI (0-5%)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color negative-roi"></span>
              <span className="legend-label">Negative ROI (&lt;0%)</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="map-3d-content">
        {/* ROI Data Overlay */}
        <div style={{ position: 'relative', zIndex: 1, width: '100%', height: '100%' }}>
          <DeckGL
            initialViewState={viewState}
            controller={true}
            layers={layers}
            style={{ width: '100%', height: '100%' }}
            getTooltip={({ object }) => {
              if (object) {
                return {
                  html: `
                    <div style="background: steelblue; color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
                      <b>${object.tooltipText}</b>
                    </div>
                  `,
                  style: {
                    backgroundColor: 'transparent',
                    border: 'none'
                  }
                };
              }
              return null;
            }}
          />
        </div>
      </div>
      
      <div className="map-3d-footer">
        <div className="data-summary">
          <div className="summary-item">
            <span className="summary-label">Neighborhoods:</span>
            <span className="summary-value">{roiData.length}</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Avg ROI:</span>
            <span className="summary-value">{(roiData.reduce((sum, item) => sum + item.roi, 0) / roiData.length).toFixed(1)}%</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Top ROI:</span>
            <span className="summary-value">{Math.max(...roiData.map(item => item.roi)).toFixed(1)}%</span>
          </div>
          <div className="summary-item">
            <span className="summary-label">Total Value:</span>
            <span className="summary-value">${(roiData.reduce((sum, item) => sum + item.propertyValue, 0) / 1000000).toFixed(1)}M</span>
          </div>
        </div>
      </div>
    </div>
  );
};
