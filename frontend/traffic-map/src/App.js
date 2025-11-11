import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './App.css';
import { 
  FaTrafficLight, 
  FaLocationArrow, 
  FaSearch, 
  FaBolt, 
  FaCrosshairs, 
  FaCar, 
  FaTruck, 
  FaBus,
  FaChevronLeft,
  FaChevronRight,
  FaMapMarkerAlt,
  FaTachometerAlt
} from 'react-icons/fa';

import { MdSpeed } from 'react-icons/md';
import { BiTargetLock } from 'react-icons/bi';
import { AiOutlineLoading3Quarters } from 'react-icons/ai';
import SplashScreen from './components/SplashScreen';

const API_BASE_URL = 'http://127.0.0.1:8000';

// Fix default marker icon issue with webpack
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Get traffic color helper function
const getTrafficColor = (level) => {
  switch(level) {
    case 'high': return '#EF4444'; // Red
    case 'medium': return '#F59E0B'; // Amber
    case 'low': return '#10B981'; // Green
    default: return '#6B7280'; // Gray
  }
};

// Create custom marker icon based on traffic level
const createCustomIcon = (trafficLevel) => {
  const color = getTrafficColor(trafficLevel);
  // Convert hex to rgba for shadow
  const hexToRgba = (hex, alpha) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };
  
  const iconHtml = `
    <div style="
      width: 40px;
      height: 40px;
      background: ${color};
      border: 3px solid white;
      border-radius: 50%;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), 0 0 0 4px ${hexToRgba(color, 0.4)};
      display: flex;
      align-items: center;
      justify-content: center;
      animation: pulse 2s ease-in-out infinite;
    ">
      <div style="
        width: 12px;
        height: 12px;
        background: white;
        border-radius: 50%;
      "></div>
    </div>
  `;
  
  return L.divIcon({
    html: iconHtml,
    className: 'custom-traffic-marker',
    iconSize: [40, 40],
    iconAnchor: [20, 20],
    popupAnchor: [0, -20]
  });
};

// Component to recenter map
function ChangeView({ center, zoom }) {
  const map = useMap();
  map.setView(center, zoom);
  return null;
}

function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [lat, setLat] = useState('24.4539');
  const [lon, setLon] = useState('54.3773');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mapCenter, setMapCenter] = useState([24.4539, 54.3773]);
  const [panelCollapsed, setPanelCollapsed] = useState(false);

  const handleStart = () => {
    setShowSplash(false);
  };

  const fetchPrediction = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict/current`, {
        lat: parseFloat(lat),
        lon: parseFloat(lon)
      });

      setPrediction({
        ...response.data,
        lat: parseFloat(lat),
        lon: parseFloat(lon)
      });
      setMapCenter([parseFloat(lat), parseFloat(lon)]);
      setLoading(false);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch traffic prediction');
      setLoading(false);
      setPrediction(null);
    }
  };


  const getTrafficIcon = (level) => {
    switch(level) {
      case 'high': return <div style={{ display: 'flex', gap: '4px' }}><FaCar /><FaTruck /><FaBus /></div>;
      case 'medium': return <div style={{ display: 'flex', gap: '4px' }}><FaCar /><FaTruck /></div>;
      case 'low': return <FaCar />;
      default: return <FaCar />;
    }
  };

  const useCurrentLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setLat(position.coords.latitude.toFixed(4));
          setLon(position.coords.longitude.toFixed(4));
        },
        () => {
          setError('Unable to get your location');
        }
      );
    } else {
      setError('Geolocation is not supported by your browser');
    }
  };

  if (showSplash) {
    return <SplashScreen onStart={handleStart} />;
  }

  return (
    <div className="App">
      <div className={`floating-panel ${panelCollapsed ? 'collapsed' : ''}`}>
        <button 
          className="panel-toggle"
          onClick={() => setPanelCollapsed(!panelCollapsed)}
          aria-label={panelCollapsed ? 'Expand panel' : 'Collapse panel'}
        >
          {panelCollapsed ? <FaChevronRight /> : <FaChevronLeft />}
        </button>

        {!panelCollapsed && (
          <>
            <div className="panel-header">
              <div className="logo">
                <FaTrafficLight className="logo-icon" />
                <div>
                  <h1>Urban Geo</h1>
                  <p className="tagline">Traffic Intelligence</p>
                </div>
              </div>
            </div>

            <div className="panel-content">
              <form onSubmit={fetchPrediction} className="input-form">
                <div className="form-group">
                  <label htmlFor="lat">
                    <FaMapMarkerAlt className="label-icon" />
                    <span>Latitude</span>
                  </label>
                  <input
                    id="lat"
                    type="number"
                    step="0.0001"
                    value={lat}
                    onChange={(e) => setLat(e.target.value)}
                    placeholder="24.4539"
                    required
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="lon">
                    <FaMapMarkerAlt className="label-icon" />
                    <span>Longitude</span>
                  </label>
                  <input
                    id="lon"
                    type="number"
                    step="0.0001"
                    value={lon}
                    onChange={(e) => setLon(e.target.value)}
                    placeholder="54.3773"
                    required
                  />
                </div>

                <div className="button-group">
                  <button
                    type="button"
                    onClick={useCurrentLocation}
                    className="location-btn"
                  >
                    <FaLocationArrow /> Use My Location
                  </button>

                  <button type="submit" className="predict-btn" disabled={loading}>
                    {loading ? (
                      <>
                        <AiOutlineLoading3Quarters className="spinner" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <FaSearch />
                        Get Prediction
                      </>
                    )}
                  </button>
                </div>
              </form>

              {error && (
                <div className="error-card">
                  <FaBolt className="error-icon" />
                  <p>{error}</p>
                </div>
              )}

              {prediction && (
                <div className={`traffic-report-card traffic-${prediction.traffic_level}`}>
                  <div className="report-header">
                    <div>
                      <h2>Traffic Report</h2>
                      <p className="report-subtitle">Real-time Analysis</p>
                    </div>
                    <div className="traffic-icon-wrapper">
                      {getTrafficIcon(prediction.traffic_level)}
                    </div>
                  </div>

                  <div className="report-metrics">
                    <div className="metric-item">
                      <div className="metric-label">
                        <FaTachometerAlt className="metric-icon" />
                        <span>Traffic Level</span>
                      </div>
                      <div className="metric-value traffic-badge">
                        {prediction.traffic_level.toUpperCase()}
                      </div>
                    </div>

                    <div className="metric-item">
                      <div className="metric-label">
                        <MdSpeed className="metric-icon" />
                        <span>Average Speed</span>
                      </div>
                      <div className="metric-value large">
                        {prediction.speed_kmh.toFixed(1)} <span className="unit">km/h</span>
                      </div>
                    </div>

                    <div className="metric-item">
                      <div className="metric-label">
                        <BiTargetLock className="metric-icon" />
                        <span>Confidence</span>
                      </div>
                      <div className="confidence-container">
                        <div className="confidence-bar">
                          <div
                            className="confidence-fill"
                            style={{ width: `${prediction.confidence * 100}%` }}
                          ></div>
                        </div>
                        <span className="confidence-text">
                          {(prediction.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>

                    <div className="metric-item">
                      <div className="metric-label">
                        <FaCrosshairs className="metric-icon" />
                        <span>Location</span>
                      </div>
                      <div className="location-coords">
                        {prediction.lat.toFixed(4)}, {prediction.lon.toFixed(4)}
                      </div>
                    </div>
                  </div>

                  <div className="insight">
                    {prediction.traffic_level === 'high' && (
                      <p>
                        <FaBolt className="insight-icon" />
                        Heavy traffic detected. Consider alternative routes or delay travel.
                      </p>
                    )}
                    {prediction.traffic_level === 'medium' && (
                      <p>
                        <FaTrafficLight className="insight-icon" />
                        Moderate traffic. Travel time may be slightly longer than usual.
                      </p>
                    )}
                    {prediction.traffic_level === 'low' && (
                      <p>
                        <FaCar className="insight-icon" />
                        Light traffic. Good time to travel!
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      <div className="map-container">
        <MapContainer
          center={mapCenter}
          zoom={13}
          style={{ height: '100%', width: '100%' }}
          zoomControl={true}
        >
          <ChangeView center={mapCenter} zoom={13} />
          <TileLayer
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
            opacity={0.9}
          />

          {prediction && (
            <>
              <Circle
                center={[prediction.lat, prediction.lon]}
                radius={800}
                pathOptions={{
                  color: getTrafficColor(prediction.traffic_level),
                  fillColor: getTrafficColor(prediction.traffic_level),
                  fillOpacity: 0.15,
                  weight: 0
                }}
              />
              <Circle
                center={[prediction.lat, prediction.lon]}
                radius={500}
                pathOptions={{
                  color: getTrafficColor(prediction.traffic_level),
                  fillColor: getTrafficColor(prediction.traffic_level),
                  fillOpacity: 0.25,
                  weight: 0
                }}
              />
              <Circle
                center={[prediction.lat, prediction.lon]}
                radius={300}
                pathOptions={{
                  color: getTrafficColor(prediction.traffic_level),
                  fillColor: getTrafficColor(prediction.traffic_level),
                  fillOpacity: 0.35,
                  weight: 0
                }}
              />
              <Circle
                center={[prediction.lat, prediction.lon]}
                radius={500}
                pathOptions={{
                  color: getTrafficColor(prediction.traffic_level),
                  fillColor: 'transparent',
                  fillOpacity: 0,
                  weight: 3,
                  opacity: 0.6,
                  dashArray: '10, 10'
                }}
              />
              <Marker 
                position={[prediction.lat, prediction.lon]}
                icon={createCustomIcon(prediction.traffic_level)}
              >
                <Popup className="custom-popup">
                  <div className="popup-content">
                    <div className="popup-header">
                      <h3>Traffic Details</h3>
                      <div 
                        className="popup-badge"
                        style={{ backgroundColor: getTrafficColor(prediction.traffic_level) }}
                      >
                        {prediction.traffic_level.toUpperCase()}
                      </div>
                    </div>
                    <div className="popup-stats">
                      <div className="popup-stat">
                        <MdSpeed className="popup-stat-icon" />
                        <div>
                          <div className="popup-stat-label">Speed</div>
                          <div className="popup-stat-value">{prediction.speed_kmh.toFixed(1)} km/h</div>
                        </div>
                      </div>
                      <div className="popup-stat">
                        <BiTargetLock className="popup-stat-icon" />
                        <div>
                          <div className="popup-stat-label">Confidence</div>
                          <div className="popup-stat-value">{(prediction.confidence * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                    </div>
                  </div>
                </Popup>
              </Marker>
            </>
          )}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;
