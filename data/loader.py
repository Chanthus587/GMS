"""
data/loader.py
─────────────────────────────────────────────────────────
Two data sources:

  SimulatedData   – synthetic NASA POWER-style data with
                    three controlled instability events.
                    Always works offline.

  NASAPowerData   – fetches real meteorological data from
                    the NASA POWER REST API and writes a
                    local CSV cache for reproducibility.
                    Requires internet access.

Usage
─────
from data.loader import SimulatedData, NASAPowerData

env = SimulatedData()          # instant, no internet
# OR
env = NASAPowerData()          # real data, needs internet
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# Allow running this file directly from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ══════════════════════════════════════════════════════════════════════════════
#  BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SensorEnvironment:
    """
    Base container shared by both data sources.

    Attributes
    ──────────
    N          : int              — number of sensor nodes
    T          : int              — number of time steps
    pos        : ndarray (N,2)   — (x,y) spatial positions in km
    adj        : dict{int:list}  — adjacency list
    dist       : ndarray (N,N)   — pairwise Euclidean distances
    Temp       : ndarray (N,T)   — temperature °C
    Humid      : ndarray (N,T)   — relative humidity %
    """

    def __init__(self, pos, Temp, Humid, radius):
        self.pos   = np.asarray(pos,   dtype=float)
        self.Temp  = np.asarray(Temp,  dtype=float)
        self.Humid = np.asarray(Humid, dtype=float)
        self.N     = self.pos.shape[0]
        self.T     = self.Temp.shape[1]
        self.G     = float(config.NETWORK['grid_size'])
        self.dist  = cdist(self.pos, self.pos)
        self.adj   = {
            i: [j for j in range(self.N)
                if j != i and self.dist[i, j] <= radius]
            for i in range(self.N)
        }

    def summary(self):
        edges = sum(len(v) for v in self.adj.values()) // 2
        print(f"  Nodes        : {self.N}")
        print(f"  Time steps   : {self.T}")
        print(f"  Edges        : {edges}")
        print(f"  T range      : {self.Temp.min():.1f} – {self.Temp.max():.1f} °C")
        print(f"  H range      : {self.Humid.min():.1f} – {self.Humid.max():.1f} %")


# ══════════════════════════════════════════════════════════════════════════════
#  SIMULATED DATA
# ══════════════════════════════════════════════════════════════════════════════

class SimulatedData(SensorEnvironment):
    """
    Generates realistic synthetic sensor data:
      • Diurnal temperature cycle (background)
      • Spatially correlated Gaussian noise
      • Three injected instability events (configurable in config.py)

    Node positions are semi-random with a fixed seed; the first eight
    nodes are pinned so they match the cluster layout used in the paper.
    """

    def __init__(self):
        cfg  = config.NETWORK
        np.random.seed(cfg['random_seed'])

        pos   = self._place_nodes(cfg['n_nodes'], cfg['grid_size'])
        Temp, Humid = self._simulate(pos, cfg['n_nodes'],
                                     cfg['time_steps'], cfg['grid_size'])

        super().__init__(pos, Temp, Humid, cfg['neighbor_radius'])
        print("[SimulatedData] Environment ready.")
        self.summary()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _place_nodes(self, N, G):
        pos = np.random.uniform(0.5, G - 0.5, (N, 2))
        # Pin first 8 to match paper topology
        pins = [(1.5,1.5),(2.2,1.8),(1.8,2.5),
                (5.0,5.0),(5.8,4.8),(5.3,5.9),
                (8.0,7.5),(8.6,8.0)]
        for i, (x, y) in enumerate(pins[:N]):
            pos[i] = [x, y]
        return pos

    def _simulate(self, pos, N, T, G):
        t_phase = np.linspace(0, 2 * np.pi, T)

        Temp  = np.zeros((N, T))
        Humid = np.zeros((N, T))

        for i in range(N):
            elev  = (pos[i, 0] + pos[i, 1]) / (2 * G)
            Temp[i]  = (22 + 6 * np.sin(t_phase - 0.3) + elev * 2.5
                        + np.random.normal(0, 0.25, T))
            Humid[i] = (65 - 8 * np.sin(t_phase) - elev * 3
                        + np.random.normal(0, 0.5, T))

        # Inject controlled instability events
        for ev in config.EVENTS:
            ns, ts, te, dT = ev['nodes'], ev['t_start'], ev['t_end'], ev['delta_T']
            dur = te - ts
            for i in ns:
                if i >= N:
                    continue
                ramp = np.zeros(T)
                ramp[ts:te] = np.linspace(0, dT, dur)
                ramp[te:]   = dT * np.exp(-np.arange(T - te) / 12.0)
                Temp[i]  += ramp
                Humid[i] -= ramp * 0.8   # humidity drops with temperature rise

        return Temp, Humid


# ══════════════════════════════════════════════════════════════════════════════
#  NASA POWER DATA
# ══════════════════════════════════════════════════════════════════════════════

class NASAPowerData(SensorEnvironment):
    """
    Fetches daily temperature (T2M) and relative humidity (RH2M)
    from the NASA POWER REST API for each lat/lon in config.NASA['locations'].

    Data is cached to  data/nasa_cache.csv  so subsequent runs are instant.
    If the API is unreachable, falls back to SimulatedData automatically.

    Each (lon, lat) location becomes one sensor node.
    Spatial positions are projected to kilometres using a simple flat-earth
    approximation centred on the first location.
    """

    def __init__(self):
        cfg = config.NASA
        cache = config.PATHS['data_cache']

        if os.path.exists(cache):
            print(f"[NASAPowerData] Loading cached data from {cache}")
            Temp, Humid, pos = self._load_cache(cache)
        else:
            print("[NASAPowerData] Fetching from NASA POWER API …")
            try:
                Temp, Humid, pos = self._fetch(cfg, cache)
            except Exception as exc:
                print(f"[NASAPowerData] Fetch failed: {exc}")
                print("[NASAPowerData] Falling back to SimulatedData.")
                sim = SimulatedData()
                super().__init__(sim.pos, sim.Temp, sim.Humid,
                                 config.NETWORK['neighbor_radius'])
                return

        super().__init__(pos, Temp, Humid, config.NETWORK['neighbor_radius'])
        print("[NASAPowerData] Environment ready.")
        self.summary()

    # ── fetch ─────────────────────────────────────────────────────────────────

    def _fetch(self, cfg, cache_path):
        import urllib.request

        locs  = cfg['locations']
        N     = len(locs)
        Temp_list  = []
        Humid_list = []

        for lon, lat in locs:
            url = (
                f"{cfg['base_url']}?"
                f"parameters={cfg['parameters']}"
                f"&community={cfg['community']}"
                f"&longitude={lon}&latitude={lat}"
                f"&start={cfg['start_date']}&end={cfg['end_date']}"
                f"&format=JSON"
            )
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            props = data['properties']['parameter']
            dates = sorted(props['T2M'].keys())
            Temp_list.append([props['T2M'][d]  for d in dates])
            Humid_list.append([props['RH2M'][d] for d in dates])

        Temp  = np.array(Temp_list,  dtype=float)
        Humid = np.array(Humid_list, dtype=float)

        # Replace fill values
        Temp[Temp   < -900] = np.nan
        Humid[Humid < -900] = np.nan

        # Forward-fill NaN
        for i in range(N):
            df = pd.DataFrame({'T': Temp[i], 'H': Humid[i]})
            df.ffill(inplace=True); df.bfill(inplace=True)
            Temp[i]  = df['T'].values
            Humid[i] = df['H'].values

        # Project lat/lon → km (flat earth)
        lat0, lon0 = locs[0][1], locs[0][0]
        pos = np.array([
            [(lon - lon0) * 111.32 * np.cos(np.radians(lat0)),
             (lat - lat0) * 110.574]
            for lon, lat in locs
        ])

        # Cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        rows = []
        T = Temp.shape[1]
        for i in range(N):
            for t in range(T):
                rows.append({'node': i, 't': t,
                             'T': Temp[i,t], 'H': Humid[i,t],
                             'x': pos[i,0], 'y': pos[i,1]})
        pd.DataFrame(rows).to_csv(cache_path, index=False)
        print(f"[NASAPowerData] Cached to {cache_path}")

        return Temp, Humid, pos

    def _load_cache(self, cache_path):
        df   = pd.read_csv(cache_path)
        N    = df['node'].nunique()
        T    = df['t'].nunique()
        Temp  = df.pivot(index='node', columns='t', values='T').values
        Humid = df.pivot(index='node', columns='t', values='H').values
        pos_df = df.groupby('node')[['x','y']].first()
        pos  = pos_df.values
        return Temp, Humid, pos
