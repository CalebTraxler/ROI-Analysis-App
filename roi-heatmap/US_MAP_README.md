# US Map ROI Visualization

## Overview
This application now includes a comprehensive US map visualization for real estate ROI (Return on Investment) analysis. The US map provides a state-level view of investment performance with interactive features and detailed popups.

## Features

### 🗺️ US Map Component
- **State-level visualization**: Color-coded states based on average ROI performance
- **Interactive popups**: Click on states to see detailed statistics
- **Neighborhood markers**: Individual property locations with ROI data
- **Responsive design**: Works on desktop and mobile devices

### 🎨 Color Coding System
- **High ROI (≥15%)**: Red (#d73027) - Excellent investment opportunities
- **Good ROI (10-15%)**: Orange (#fc8d59) - Good investment potential
- **Moderate ROI (5-10%)**: Yellow (#fee08b) - Moderate returns
- **Low ROI (0-5%)**: Light Green (#d9ef8b) - Lower returns
- **Negative ROI (<0%)**: Green (#91cf60) - Declining values

### 🔧 Technical Implementation
- **Leaflet.js**: Open-source mapping library for interactive maps
- **React-Leaflet**: React wrapper for Leaflet integration
- **GeoJSON**: US states boundary data from US Atlas
- **OpenStreetMap**: Free tile layer for map backgrounds

## Map Types

### 1. US Map (Default)
- Shows the entire United States with state boundaries
- Color-coded by average ROI performance
- Interactive state popups with detailed statistics
- Individual neighborhood markers for specific locations

### 2. 3D Heatmap
- 3D visualization using deck.gl
- Heatmap density based on ROI values
- Interactive 3D controls and zoom
- Detailed tooltips for each data point

## Usage

### Switching Between Maps
1. Use the toggle buttons in the controls section
2. Choose between "US Map" and "3D Heatmap"
3. The selected map type will be highlighted

### Interacting with the US Map
1. **Zoom**: Use mouse wheel or zoom controls
2. **Pan**: Click and drag to move around the map
3. **State Information**: Click on any state to see:
   - Average ROI percentage
   - Number of neighborhoods
   - Total property value
   - Top neighborhood names
4. **Neighborhood Details**: Click on individual markers for:
   - Neighborhood name
   - ROI percentage
   - Property value
   - State information

## Data Requirements

The US map expects ROI data with the following structure:
```typescript
interface ROIData {
  id: number;
  neighborhood: string;
  county: string;
  state: string;
  latitude: number;
  longitude: number;
  roi: number;
  propertyValue: number;
  appreciation: number;
  rentalYield: number;
  marketTrend: 'rising' | 'stable' | 'declining';
}
```

## Performance Features

- **Lazy loading**: US states data loads only when needed
- **Caching**: State data is processed and cached efficiently
- **Responsive rendering**: Map adapts to different screen sizes
- **Error handling**: Graceful fallbacks for data loading issues

## Browser Compatibility

- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Dependencies

```json
{
  "leaflet": "^1.9.4",
  "react-leaflet": "^4.2.1",
  "@types/leaflet": "^1.9.8"
}
```

## Troubleshooting

### Map Not Loading
1. Check internet connection (requires external US states data)
2. Verify Leaflet CSS is imported
3. Check browser console for errors

### Icons Not Displaying
1. Ensure Leaflet images are accessible
2. Check the useLeafletFix hook is working
3. Verify package installation

### Performance Issues
1. Reduce the number of neighborhood markers
2. Use data filtering to limit displayed points
3. Check for memory leaks in data processing

## Future Enhancements

- [ ] County-level boundaries and data
- [ ] Time-series animation for ROI trends
- [ ] Custom map themes and styles
- [ ] Export functionality for map data
- [ ] Advanced filtering and search
- [ ] Integration with real-time market data

## Contributing

When adding new features to the US map:
1. Maintain the existing color scheme
2. Follow the established component structure
3. Add proper TypeScript types
4. Include responsive design considerations
5. Test with various data scenarios
