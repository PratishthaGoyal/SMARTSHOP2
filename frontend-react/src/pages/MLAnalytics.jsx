import React, { useState } from "react";
import axios from "axios";

const API_URL = "http://localhost:8000/predict";

const MLAnalytics = () => {
  const [loading, setLoading] = useState(false);
  const [forecast, setForecast] = useState([]);
  const [classification, setClassification] = useState(null);
  const [error, setError] = useState("");

  const runPrediction = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await axios.get(API_URL);
      const data = response.data;

      setForecast(data.forecast || []);
      setClassification(data.classification || null);
    } catch (err) {
      setError("Failed to fetch data from backend");
    }
    setLoading(false);
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h3>🤖 AI-Driven Sales Forecast & Performance</h3>
      <button onClick={runPrediction} style={styles.btn}>
        {loading ? "Running..." : "Run ML Prediction"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {forecast.length > 0 && (
        <div>
          <h4>📈 Sales Forecast</h4>
          <table border="1" cellPadding="8" style={{ marginTop: "1rem" }}>
            <thead>
              <tr>
                {Object.keys(forecast[0]).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {forecast.map((row, i) => (
                <tr key={i}>
                  {Object.values(row).map((val, j) => (
                    <td key={j}>{val}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {classification && (
        <div style={{ marginTop: "2rem" }}>
          <h4>🎯 Classification Results</h4>
          <p>
            <strong>Accuracy:</strong>{" "}
            {(classification.accuracy * 100).toFixed(2)}%
          </p>
          <pre>{JSON.stringify(classification.feature_importance, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

const styles = {
  btn: {
    backgroundColor: "#28a745",
    color: "white",
    padding: "0.5rem 1rem",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    marginBottom: "1rem",
  },
};

export default MLAnalytics;
