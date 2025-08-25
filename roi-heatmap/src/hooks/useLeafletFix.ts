import { useEffect } from 'react';
import L from 'leaflet';

// Fix for Leaflet default icon issues in React
export const useLeafletFix = () => {
  useEffect(() => {
    // Fix Leaflet default icon issues
    delete (L.Icon.Default.prototype as any)._getIconUrl;
    
    // Use CDN URLs for Leaflet icons to avoid build issues
    L.Icon.Default.mergeOptions({
      iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
      iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
      shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
    });
  }, []);
};
