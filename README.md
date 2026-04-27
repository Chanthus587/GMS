# GMS: Gradient-Momentum Score Microclimate Instability Detection System

A sophisticated machine learning system for detecting microclimate thermal instabilities using a network of distributed sensor nodes. The system leverages spatial-temporal analysis to identify anomalous temperature patterns in real-time.

## 🎯 Overview

The **Gradient-Momentum Score (GMS)** model combines four key components to detect microclimate instabilities:

1. **Spatial Gradient (ΔT)** - Temperature differences between neighboring nodes
2. **Temporal Momentum (M)** - Rate of change in temperature gradients over time
3. **Duration/Persistence (D)** - How long instability persists at a location
4. **Neighbor Influence Score (NIS)** - Anomaly propagation through the sensor network

The composite GMS score classifies each sensor location into three stability states:
- **Stable** - Normal microclimate conditions
- **Moderately Unstable** - Emerging temperature anomalies
- **Highly Unstable** - Critical instability events

## ✨ Key Features

- **40-Node Sensor Network** - Distributed temperature and humidity monitoring
- **Real-time Anomaly Detection** - Dynamic classification of instability events
- **Web-based Dashboard** - Interactive Mission Control interface at `http://localhost:5000`
- **Multi-baseline Support** - Configurable absolute temperature thresholds
- **Noise Tolerance** - Toggle-able noise injection for robustness testing
- **NASA POWER API Integration** - Real-world climate data from satellite measurements
- **Comprehensive Visualization** - Spatial plots, time-series analysis, and network graphs
- **Metric Evaluation** - Precision, recall, and F1-score calculations

## 📊 Project Structure

```
gms_project/
├── app.py                    # Flask web interface (Mission Control)
├── main.py                   # CLI entry point
├── config.py                 # Central configuration hub
├── requirements.txt          # Python dependencies
│
├── core/
│   ├── gms_model.py         # Core GMS algorithm implementation
│   └── __init__.py
│
├── data/
│   ├── loader.py            # Data loading (simulated & NASA)
│   ├── nasa_cache.csv       # Cached NASA POWER API data
│   └── __init__.py
│
├── visualization/
│   ├── plots.py             # Matplotlib visualizations
│   └── __init__.py
│
├── evaluation/
│   ├── metrics.py           # Performance metrics (precision, recall, F1)
│   └── __init__.py
│
├── tests/
│   ├── test_gms.py          # Unit tests
│   └── __init__.py
│
├── outputs/
│   ├── gms_clean.csv        # Clean temperature data
│   ├── gms_noise.csv        # Noisy temperature data
│   └── plots/               # Generated visualizations
│
└── gui_mission_control.py   # Interactive GUI launcher
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd gms_project

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Web Dashboard (Recommended)
```bash
python app.py
```
Open your browser and navigate to **http://localhost:5000**
- Real-time sensor network visualization
- Interactive event timeline
- Temperature heatmap and anomaly alerts
- Configurable noise and weight parameters

#### 2. Command-line Interface
```bash
python main.py
```
- Load and analyze simulated or real NASA data
- Generate evaluation metrics
- Export results to CSV

#### 3. GUI Mission Control
```bash
python gui_mission_control.py
```
- Standalone desktop interface for advanced analysis

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Network topology
n_nodes = 40              # Number of sensor nodes
grid_size = 10.0          # Coverage area (km²)
neighbor_radius = 2.8     # Neighborhood detection radius

# Algorithm weights (must sum to 1.0)
w1 = 0.35  # Spatial Gradient weight
w2 = 0.25  # Temporal Momentum weight
w3 = 0.20  # Neighbor Influence weight
w4 = 0.20  # Duration/Persistence weight

# Thresholds
alpha = 0.30  # Stable → Moderately Unstable
beta = 0.60   # Moderately Unstable → Highly Unstable

# Simulated events (for testing)
EVENTS = [
    dict(nodes=[0,1,2,3,4], t_start=20, t_end=55, delta_T=8.0, label="Event A"),
    # ... more events
]
```

## 📈 Algorithm Details

### Spatial Gradient (Component 1)
```
ΔT_ij(t) = T_i(t) − T_j(t)
```
Computes mean temperature difference between a node and its neighbors, detecting spatial anomalies.

### Temporal Momentum (Component 2)
```
M_ij(t) = ΔT(t) − ΔT(t−1)
```
Captures rate of change in spatial gradients, identifying rapid temperature shifts.

### Duration/Persistence (Component 3)
```
D_i(t) = fraction of time steps in sliding window where |ΔT| > θ
```
Measures how long instability persists, distinguishing transient from sustained anomalies.

### Neighbor Influence Score (Component 4)
```
NIS_i(t) = weighted sum of anomalies in neighboring nodes
```
Propagates anomaly signals through the network topology.

### Composite GMS Score
```
GMS_i(t) = w1·G2 + w2·M + w3·NIS + w4·D
```
Weighted combination normalized to [0, 1] range.

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥1.24.0 | Numerical computations |
| `pandas` | ≥2.0.0 | Data manipulation |
| `scipy` | ≥1.10.0 | Scientific computing |
| `scikit-learn` | ≥1.3.0 | Machine learning utilities |
| `matplotlib` | ≥3.7.0 | Visualization |
| `networkx` | ≥3.1 | Network analysis |
| `requests` | ≥2.31.0 | HTTP API calls (NASA POWER) |

## 🧪 Testing

```bash
# Run test suite
python -m pytest tests/test_gms.py -v

# Generate test report with coverage
pytest tests/test_gms.py --cov=core --cov=data
```

## 📊 Data Sources

### Simulated Data (Default)
- Synthetic temperature and humidity time series
- Embedded thermal events (A, B, C, D)
- Configurable noise levels

### NASA POWER API (Real Data)
- Satellite-derived 2-meter air temperature (T2M)
- Relative humidity at 2 meters (RH2M)
- Daily aggregated values
- Geographic coordinates (latitude/longitude) configurable

## 📈 Output Files

- **gms_clean.csv** - Clean temperature readings without noise
- **gms_noise.csv** - Temperature data with injected noise
- **plots/** - Generated visualizations (heatmaps, time series, network graphs)

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Additional anomaly detection algorithms
- Real-time data streaming support
- Advanced visualization features
- Performance optimizations for large networks

## 📝 License

[Add your license here - MIT, Apache 2.0, etc.]

## 📧 Contact

For questions or suggestions, open an issue or contact the development team.

---

**Last Updated:** April 2026 | **Version:** 2.0
