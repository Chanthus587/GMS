"""
config.py
─────────────────────────────────────────────────────────
Central configuration for the GMS Microclimate Instability
Detection System.  Change values here; every module reads
from this file so nothing is hard-coded elsewhere.
"""

# ── Network ───────────────────────────────────────────────────────────────────
NETWORK = dict(
    n_nodes       = 12,          # number of sensor nodes
    grid_size     = 10.0,        # km, spatial extent of the grid
    neighbor_radius = 3.8,       # km, max distance to be a neighbor
    time_steps    = 120,         # number of time steps to simulate
    random_seed   = 2024,
)

# ── GMS Model weights  (must sum to 1.0) ─────────────────────────────────────
WEIGHTS = dict(
    w1 = 0.35,   # Spatial Gradient
    w2 = 0.25,   # Temporal Momentum
    w3 = 0.20,   # Neighbor Influence Score
    w4 = 0.20,   # Duration / Persistence
)

# ── Algorithm hyper-parameters ───────────────────────────────────────────────
ALGO = dict(
    theta          = 1.2,   # °C — minimum gradient to count as instability
    window         = 8,     # time steps — sliding window for persistence
    alpha          = 0.30,  # GMS threshold: Stable → Mod-Unstable
    beta           = 0.60,  # GMS threshold: Mod-Unstable → High-Unstable
)

# ── Instability injection (simulation only) ───────────────────────────────────
EVENTS = [
    dict(nodes=[0,1,2],   t_start=25, t_end=55, delta_T=7.0,  label="Event A"),
    dict(nodes=[8,9],     t_start=38, t_end=70, delta_T=4.5,  label="Event B"),
    dict(nodes=[3,4,5],   t_start=55, t_end=90, delta_T=5.5,  label="Event C"),
]

# ── Absolute-threshold baseline ───────────────────────────────────────────────
BASELINE = dict(
    abs_threshold  = 26.5,   # °C — fires alarm when T_i > this
)

# ── NASA POWER API (optional real data) ──────────────────────────────────────
NASA = dict(
    base_url   = "https://power.larc.nasa.gov/api/temporal/daily/point",
    parameters = "T2M,RH2M",      # temperature 2 m, relative humidity 2 m
    community  = "RE",
    # Lat/lon pairs treated as sensor node positions
    # (lon, lat) tuples — add / remove pairs to add more nodes
    locations  = [
        (75.85, 30.90),   # node 0 — Ludhiana area
        (75.90, 30.95),   # node 1
        (75.95, 31.00),   # node 2
        (76.00, 31.05),   # node 3
        (76.05, 31.10),   # node 4
        (76.10, 31.15),   # node 5
        (76.15, 31.20),   # node 6
        (76.20, 31.25),   # node 7
        (76.25, 31.30),   # node 8
        (76.30, 31.35),   # node 9
        (76.35, 31.40),   # node 10
        (76.40, 31.45),   # node 11
    ],
    start_date = "20240101",
    end_date   = "20240501",
)

# ── Output paths ─────────────────────────────────────────────────────────────
PATHS = dict(
    output_dir = "outputs",
    fig_prefix = "gms",
    data_cache = "data/nasa_cache.csv",
)
