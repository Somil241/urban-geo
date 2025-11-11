import React, { useState, useEffect } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import axios from "axios";

const API_BASE_URL = 'http://127.0.0.1:8000'; // Backend API URL

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
  const [mapCenter, setMapCenter] = useState([24.4539, 54.3773]); // Default center
  const [trafficData, setTrafficData] = useState([]); // Traffic data from API

  // Fetch traffic data from the backend
  useEffect(() => {
    const fetchTrafficData = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/traffic-data`); // Replace with your endpoint
        setTrafficData(response.data);
      } catch (error) {
        console.error("Error fetching traffic data:", error);
      }
    };

    fetchTrafficData();
  }, []);

  // Function to determine marker color based on traffic congestion
  const getTrafficIcon = (congestionLevel) => {
    let iconUrl;
    if (congestionLevel === "high") {
      iconUrl = "https://upload.wikimedia.org/wikipedia/commons/8/88/Map_marker-red.svg"; // Red marker
    } else if (congestionLevel === "medium") {
      iconUrl = "https://upload.wikimedia.org/wikipedia/commons/1/1c/Map_marker-yellow.svg"; // Yellow marker
    } else {
      iconUrl = "https://upload.wikimedia.org/wikipedia/commons/7/72/Map_marker-green.svg"; // Green marker
    }

    return new L.Icon({
      iconUrl,
      iconSize: [25, 41],
      iconAnchor: [12, 41],
      popupAnchor: [1, -34],
      shadowUrl: require("leaflet/dist/images/marker-shadow.png"),
      shadowSize: [41, 41],
    });
  };

  return (
    <MapContainer center={mapCenter} zoom={13} style={{ height: "100vh", width: "100%" }}>
      <ChangeView center={mapCenter} zoom={13} />
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {trafficData.map((data, index) => (
        <Marker
          key={index}
          position={[data.lat, data.lon]}
          icon={getTrafficIcon(data.congestionLevel)}
        >
          <Popup>
            <strong>Location:</strong> {data.city}, {data.country} <br />
            <strong>Traffic Level:</strong> {data.congestionLevel} <br />
            <strong>Traffic Index:</strong> {data.trafficIndex}
          </Popup>
        </Marker>
      ))}
    </MapContainer>
  );
}

export default App;