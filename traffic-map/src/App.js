import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';
import './App.css';

const API_BASE_URL = 'http://127.0.0.1:8000';

// Abu Dhabi coordinates
const UAE_CENTER = [24.4539, 54.3773];

// Generate grid of points covering Abu Dhabi area
const generateGridPoints = () => {
  const points = [];
  const latMin = 24.2;
  const latMax = 24.7;
  const lonMin = 54.2;
  const lonMax = 55.0;
  const step = 0.05; // Grid resolution

  for (let lat = latMin; lat <= latMax; lat += step) {
    for (let lon = lonMin; lon <= lonMax; lon += step) {
      points.push({ lat, lon });
    }
  }
  return points;
};

function App() {
  const [trafficData, setTrafficData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchTrafficData();
  }, []);

  const fetchTrafficData = async () => {
    setLoading(true);
    setError(null);

    try {
      const gridPoints = generateGridPoints();
      const predictions = [];

      // Fetch predictions for each grid point
      for (const point of gridPoints) {
        try {
          const response = await axios.post(`${API_BASE_URL}/predict/current`, {
            lat: point.lat,
            lon: point.lon
          });

          predictions.push({
            lat: point.lat,
            lon: point.lon,
            speed: response.data.speed_kmh,
            trafficLevel: response.data.traffic_level,
            confidence: response.data.confidence
          });
        } catch (err) {
          console.error(`Error fetching data for ${point.lat}, ${point.lon}:`, err);
        }
      }

      setTrafficData(predictions);
      setLoading(false);
    } catch (err) {
      setError('Failed to fetch traffic data');
      setLoading(false);
    }
  };

  const getColorByTrafficLevel = (trafficLevel, speed) => {
    // Red for high traffic (slow speeds)
    // Yellow for medium traffic
    // Green for low traffic (fast speeds)
    if (trafficLevel === 'high' || speed < 30) {
      return '#ff0000'; // Red
    } else if (trafficLevel === 'medium' || speed < 50) {
      return '#ffaa00'; // Orange/Yellow
    } else {
      return '#00ff00'; // Green
    }
  };

  const getRadiusBySpeed = (speed) => {
    // Larger circles for slower traffic
    if (speed < 30) return 400;
    if (speed < 50) return 300;
    return 200;
  };

  return (
    <div className="App">
      <div className="header">
        <h1>Urban-Geo Traffic Map - Abu Dhabi</h1>
        <div className="controls">
          <button onClick={fetchTrafficData} disabled={loading}>
            {loading ? 'Loading...' : 'Refresh Data'}
          </button>
          <div className="legend">
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: '#00ff00' }}></span>
              Low Traffic
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: '#ffaa00' }}></span>
              Medium Traffic
            </div>
            <div className="legend-item">
              <span className="legend-color" style={{ backgroundColor: '#ff0000' }}></span>
              High Traffic
            </div>
          </div>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      <MapContainer
        center={UAE_CENTER}
        zoom={11}
        style={{ height: 'calc(100vh - 150px)', width: '100%' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        {trafficData.map((point, index) => (
          <CircleMarker
            key={index}
            center={[point.lat, point.lon]}
            radius={getRadiusBySpeed(point.speed) / 50}
            fillColor={getColorByTrafficLevel(point.trafficLevel, point.speed)}
            color={getColorByTrafficLevel(point.trafficLevel, point.speed)}
            weight={1}
            opacity={0.6}
            fillOpacity={0.4}
          >
            <Popup>
              <div>
                <strong>Traffic Level:</strong> {point.trafficLevel}<br />
                <strong>Speed:</strong> {point.speed.toFixed(1)} km/h<br />
                <strong>Confidence:</strong> {(point.confidence * 100).toFixed(1)}%<br />
                <strong>Location:</strong> {point.lat.toFixed(3)}, {point.lon.toFixed(3)}
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>

      <div className="stats">
        <p>Total Points: {trafficData.length}</p>
        <p>
          High Traffic: {trafficData.filter(p => p.trafficLevel === 'high').length} |
          Medium: {trafficData.filter(p => p.trafficLevel === 'medium').length} |
          Low: {trafficData.filter(p => p.trafficLevel === 'low').length}
        </p>
      </div>
    </div>
  );
}

export default App;
