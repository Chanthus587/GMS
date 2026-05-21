# GMS Mission Control: Microclimate Anomaly Detection and Climate Sensor Network Dashboard

GMS Mission Control is a Python and Flask application for detecting microclimate instability, thermal anomalies, and sensor-network events in real time. It combines a Gradient-Momentum Score model with spatial-temporal analysis, NASA POWER climate data support, interactive dashboards, anomaly alerts, geospatial visualizations, and optimization tools for climate monitoring research.

## Topics And Keywords

`microclimate` `anomaly-detection` `climate-tech` `climate-data` `sensor-network` `iot-sensors` `environmental-monitoring` `thermal-instability` `spatial-temporal-analysis` `machine-learning` `flask` `python` `data-visualization` `nasa-power-api` `dashboard` `optimization`

## Overview

The **Gradient-Momentum Score (GMS)** model detects microclimate instability by combining four signals from a distributed temperature and humidity sensor network:

1. **Spatial Gradient (Delta T)** - temperature differences between neighboring nodes.
2. **Temporal Momentum (M)** - rate of change in temperature gradients over time.
3. **Duration/Persistence (D)** - how long instability persists at a location.
4. **Neighbor Influence Score (NIS)** - anomaly propagation through the sensor network.

The composite GMS score classifies each sensor location into three stability states:

- **Stable** - normal microclimate conditions.
- **Moderately Unstable** - emerging temperature anomalies.
- **Highly Unstable** - critical instability events.

## Key Features

- **40-node sensor network** for distributed temperature and humidity monitoring.
- **Real-time anomaly detection** with dynamic instability classification.
- **Flask web dashboard** with Mission Control views at `http://localhost:5000`.
- **Interactive climate visualizations** including maps, heatmaps, timelines, alerts, and analysis charts.
- **NASA POWER API integration** for real-world climate and weather data.
- **Spatial-temporal GMS algorithm** for thermal anomaly and instability detection.
- **Hyperparameter optimizer** for weights, thresholds, and duration-window tuning.
- **Evaluation metrics** including accuracy, precision, recall, F1-score, and false alarm rate.
- **CLI and desktop GUI entry points** for research workflows and local experimentation.

## Project Structure

```text
gms_project/
|-- app.py                    # Flask web interface
|-- main.py                   # CLI entry point
|-- gui_mission_control.py    # Desktop GUI launcher
|-- requirements.txt          # Python dependencies
|-- README.md
|-- PROJECT_STRUCTURE.md
|
|-- backend/                  # Web backend: engine, routes, optimizer
|-- frontend/                 # Templates, CSS, and JavaScript assets
|-- config/                   # Application settings
|-- core/                     # Core GMS algorithm
|-- data/                     # Data loading and cached climate data
|-- evaluation/               # Metric calculations
|-- tests/                    # Unit tests
|-- utils/                    # Shared helpers
|-- visualization/            # Plotting utilities
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/Chanthus587/GMS.git
cd GMS
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run The Web Dashboard

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

The dashboard includes:

- Real-time sensor network visualization.
- Interactive event timeline.
- Temperature heatmaps and anomaly alerts.
- Configurable noise, baseline, and GMS weight parameters.
- Optimization controls for model tuning.

### Run The CLI Pipeline

```bash
python main.py
```

Use the CLI to load simulated or NASA climate data, run GMS analysis, calculate metrics, and export results.

### Run The Desktop GUI

```bash
python gui_mission_control.py
```

## Configuration

Edit `config/settings.py` for the web app and 40-node Mission Control defaults. The `config` package also exposes legacy names used by the CLI and core pipeline.

```python
# Network topology
n_nodes = 40
grid_size = 10.0
neighbor_radius = 2.8

# Algorithm weights; must sum to 1.0
w1 = 0.35  # Spatial Gradient
w2 = 0.25  # Temporal Momentum
w3 = 0.20  # Neighbor Influence
w4 = 0.20  # Duration/Persistence

# Thresholds
alpha = 0.30  # Stable to Moderately Unstable
beta = 0.60   # Moderately Unstable to Highly Unstable
```

## Algorithm Details

### Spatial Gradient

```text
DeltaT_ij(t) = T_i(t) - T_j(t)
```

Computes mean temperature difference between a node and its neighbors to detect spatial anomalies.

### Temporal Momentum

```text
M_ij(t) = DeltaT(t) - DeltaT(t-1)
```

Captures rapid changes in spatial gradients.

### Duration/Persistence

```text
D_i(t) = fraction of time steps in a sliding window where |DeltaT| > theta
```

Distinguishes transient noise from sustained microclimate instability.

### Neighbor Influence Score

```text
NIS_i(t) = weighted sum of anomalies in neighboring nodes
```

Models how anomaly signals propagate through the sensor network.

### Composite GMS Score

```text
GMS_i(t) = w1*G2 + w2*M + w3*NIS + w4*D
```

The weighted score is normalized to the `[0, 1]` range and mapped to stability states.

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `scipy` | Scientific computing |
| `scikit-learn` | Machine learning utilities |
| `matplotlib` | Plotting and visualization |
| `networkx` | Sensor network analysis |
| `requests` | NASA POWER API calls |
| `flask` | Web dashboard and API |

## Testing

```bash
python -m pytest
```

## Data Sources

### Simulated Data

- Synthetic temperature and humidity time series.
- Embedded thermal events for controlled testing.
- Configurable noise and instability patterns.

### NASA POWER API

- Satellite-derived 2-meter air temperature (`T2M`).
- Relative humidity at 2 meters (`RH2M`).
- Daily climate measurements by latitude and longitude.

## Hyperparameter Optimizer

The optimizer tunes component weights, thresholds, and the duration window from the web UI or Flask API.

```bash
curl -X POST http://localhost:5000/api/optimize \
  -H "Content-Type: application/json" \
  -d "{\"iterations\": 50, \"seed\": 42, \"target_recall\": 0.70, \"target_fp_rate\": 0.05}"
```

The optimizer evaluates accuracy, precision, recall, F1-score, and false alarm rate to reduce unnecessary alerts while preserving useful anomaly detection.

## Who This Project Is For

This project is useful for students, researchers, and developers working on:

- Climate monitoring and environmental sensing.
- IoT sensor networks.
- Thermal anomaly detection.
- Spatial-temporal machine learning.
- Real-time Flask dashboards.
- Data visualization for weather and climate systems.

## Contributing

Contributions are welcome. Good areas for improvement include additional anomaly detection models, real-time streaming integrations, map enhancements, performance tuning, and broader climate data support.

## License

Add a license before using this project in production or publishing derivative work.

## Contact

For questions or suggestions, open an issue on GitHub.

---

**Version:** 2.0 | **Last Updated:** April 2026
