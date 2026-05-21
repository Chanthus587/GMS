"""
config/settings.py
─────────────────────────────────────────────────────────
Application configuration for the GMS Microclimate Instability
Detection System.  Change values here; every module reads
from this file so nothing is hard-coded elsewhere.
"""

# ── Flask Application Settings ─────────────────────────────────────────
DEBUG = False
HOST = '0.0.0.0'
PORT = 5000
THREADED = True

# ── GMS Model weights  (must sum to 1.0) ─────────────────────────────────
DEFAULT_WEIGHTS = {
    'w1': 0.35,   # Spatial Gradient
    'w2': 0.25,   # Temporal Momentum
    'w3': 0.20,   # Neighbor Influence Score
    'w4': 0.20,   # Duration / Persistence
}

# ── Algorithm hyper-parameters ───────────────────────────────────────────
DEFAULT_THRESHOLDS = {
    'theta': 1.2,   # °C — minimum gradient to count as instability
    'alpha': 0.25,  # GMS threshold: Stable → Mod-Unstable
    'beta': 0.60,   # GMS threshold: Mod-Unstable → High-Unstable
}

DEFAULT_WINDOW = 8

# ── Network parameters ─────────────────────────────────────────────────
NUM_NODES = 40
TIME_STEPS = 120
GRID_SIZE = 10.0
NEIGHBOR_RADIUS = 2.8

# ── Playback ───────────────────────────────────────────────────────────
DEFAULT_SPEED = 0.25  # seconds per frame

# ── Baseline Thresholds ───────────────────────────────────────────────
ABSOLUTE_THRESHOLD = 26.5   # °C — fires alarm when T_i > this
Z_SCORE_THRESHOLD = 1.2     # Z-score anomaly threshold

# ── Events Injection ───────────────────────────────────────────────────
EVENTS = [
    dict(nodes=[0,1,2,3,4],   t_start=20,t_end=55,  dT=8.0,label="Event A",color="#388BFD"),
    dict(nodes=[15,16,17,18], t_start=35,t_end=70,  dT=6.0,label="Event B",color="#3FB950"),
    dict(nodes=[8,9,10,11],   t_start=50,t_end=90,  dT=7.0,label="Event C",color="#D29922"),
    dict(nodes=[25,26,27,28], t_start=65,t_end=100, dT=5.5,label="Event D",color="#BC8CFF"),
]

# ── Output paths ───────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
DATA_CACHE = "data/nasa_cache.csv"
