"""Compatibility configuration package.

The project now uses ``config.settings`` for the Flask web app, while the
research/core pipeline still imports top-level names from ``config``. Keeping
those names here prevents the ``config/`` package from shadowing the legacy
``config.py`` module with an empty namespace.
"""

from .settings import (
    DATA_CACHE,
    DEBUG,
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    DEFAULT_WINDOW,
    HOST,
    OUTPUT_DIR,
    PORT,
    THREADED,
)

NETWORK = dict(
    n_nodes=12,
    grid_size=10.0,
    neighbor_radius=3.8,
    time_steps=120,
    random_seed=2024,
)

WEIGHTS = dict(
    w1=0.35,
    w2=0.25,
    w3=0.20,
    w4=0.20,
)

ALGO = dict(
    theta=1.2,
    window=8,
    alpha=0.30,
    beta=0.60,
)

EVENTS = [
    dict(nodes=[0, 1, 2], t_start=25, t_end=55, delta_T=7.0, label="Event A"),
    dict(nodes=[8, 9], t_start=38, t_end=70, delta_T=4.5, label="Event B"),
    dict(nodes=[3, 4, 5], t_start=55, t_end=90, delta_T=5.5, label="Event C"),
]

BASELINE = dict(
    abs_threshold=26.5,
)

NASA = dict(
    base_url="https://power.larc.nasa.gov/api/temporal/daily/point",
    parameters="T2M,RH2M",
    community="RE",
    locations=[
        (75.85, 30.90),
        (75.90, 30.95),
        (75.95, 31.00),
        (76.00, 31.05),
        (76.05, 31.10),
        (76.10, 31.15),
        (76.15, 31.20),
        (76.20, 31.25),
        (76.25, 31.30),
        (76.30, 31.35),
        (76.35, 31.40),
        (76.40, 31.45),
    ],
    start_date="20240101",
    end_date="20240501",
)

PATHS = dict(
    output_dir=OUTPUT_DIR,
    fig_prefix="gms",
    data_cache=DATA_CACHE,
)
