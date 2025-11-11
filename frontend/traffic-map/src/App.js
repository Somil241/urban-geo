import React, { useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import L from 'leaflet';
import './App.css';
import { FaTrafficLight, FaMapMarkerAlt, FaLocationArrow, FaSearch, FaBolt, FaCrosshairs, FaCar, FaTruck, FaBus } from 'react-icons/fa';
import { MdSpeed } from 'react-icons/md';
import { BiTargetLock } from 'react-icons/bi';
import { AiOutlineLoading3Quarters } from 'react-icons/ai';

const API_BASE_URL = 'http://127.0.0.1:8000';

// Fix default marker icon issue with webpack
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
  iconUrl: require('leaflet/dist/images/marker-icon.png'),
  shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Component to recenter map
function ChangeView({ center, zoom }) {
  const map = useMap();
  map.setView(center, zoom);
  return null;
}

function App() {
  const [lat, setLat] = useState('24.4539');
  const [lon, setLon] = useState('54.3773');
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mapCenter, setMapCenter] = useState([24.4539, 54.3773]);

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

  const getTrafficColor = (level) => {
    switch(level) {
      case 'high': return '#EF4444'; // Red
      case 'medium': return '#F59E0B'; // Amber
      case 'low': return '#10B981'; // Green
      default: return '#6B7280'; // Gray
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

  return (
    <div className="App">
      <div className="sidebar">
        <div className="logo-section">
          <div className="logo">
            <FaTrafficLight className="logo-icon" />
            <h1>Urban Geo</h1>
          </div>
          <p className="tagline">Real-time Traffic Intelligence</p>
        </div>

        <form onSubmit={fetchPrediction} className="input-form">
          <div className="form-group">
            <label htmlFor="lat">
              <FaMapMarkerAlt className="label-icon" />
              Latitude
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
              Longitude
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
                Get Traffic Prediction
              </>
            )}
          </button>
        </form>

        {error && (
          <div className="error-card">
            <FaTrafficLight className="error-icon" style={{ color: '#991B1B' }} />
            <p>{error}</p>
          </div>
        )}

        {prediction && (
          <div className="result-card">
            <div className="result-header">
              <h2>Traffic Report</h2>
              <span className="traffic-emoji">{getTrafficIcon(prediction.traffic_level)}</span>
            </div>

            <div className="result-item">
              <div className="result-label">Traffic Level</div>
              <div
                className="traffic-badge"
                style={{ backgroundColor: getTrafficColor(prediction.traffic_level) }}
              >
                {prediction.traffic_level.toUpperCase()}
              </div>
            </div>

            <div className="result-item">
              <div className="result-label">
                <MdSpeed className="icon" />
                Average Speed
              </div>
              <div className="result-value">
                {prediction.speed_kmh.toFixed(1)} <span className="unit">km/h</span>
              </div>
            </div>

            <div className="result-item">
              <div className="result-label">
                <BiTargetLock className="icon" />
                Confidence
              </div>
              <div className="result-value">
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

            <div className="result-item">
              <div className="result-label">
                <FaCrosshairs className="icon" />
                Location
              </div>
              <div className="result-value location-coords">
                {prediction.lat.toFixed(4)}, {prediction.lon.toFixed(4)}
              </div>
            </div>

            <div className="insight">
              {prediction.traffic_level === 'high' && (
                <p><FaBolt style={{ color: '#EF4444' }} /> Heavy traffic detected. Consider alternative routes or delay travel.</p>
              )}
              {prediction.traffic_level === 'medium' && (
                <p><FaTrafficLight style={{ color: '#F59E0B' }} /> Moderate traffic. Travel time may be slightly longer than usual.</p>
              )}
              {prediction.traffic_level === 'low' && (
                <p><FaCar style={{ color: '#10B981' }} /> Light traffic. Good time to travel!</p>
              )}
            </div>
          </div>
        )}

        <div className="info-section">
          <p className="info-text">
            Enter coordinates or use your current location to get real-time traffic predictions powered by machine learning.
          </p>
        </div>
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
            attribution='&copy; <a href="https://www.openstreetmap.org/blackcopyright">OpenStreetMap</a> contributors  &copy; <a href="https://carto.com/attributions">CARTO</a>'
          />

          {prediction && (
            <>
              <Marker position={[prediction.lat, prediction.lon]}>
                <Popup>
                  <div className="popup-content">
                    <h3>Traffic Details</h3>
                    <p><strong>Level:</strong> {prediction.traffic_level}</p>
                    <p><strong>Speed:</strong> {prediction.speed_kmh.toFixed(1)} km/h</p>
                    <p><strong>Confidence:</strong> {(prediction.confidence * 100).toFixed(1)}%</p>
                  </div>
                </Popup>
              </Marker>
              <Circle
                center={[prediction.lat, prediction.lon]}
                radius={500}
                pathOptions={{
                  color: getTrafficColor(prediction.traffic_level),
                  fillColor: getTrafficColor(prediction.traffic_level),
                  fillOpacity: 0.3,
                  weight: 2
                }}
              />
            </>
          )}
        </MapContainer>
      </div>
    </div>
  );
}

export default App;
