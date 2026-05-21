"""
backend/gms_engine.py
─────────────────────────────────────────────────────────
GMS Engine — 40 nodes, 4 events, 3 baselines, noise toggle
Real-time microclimate instability detection with streaming updates
"""

import json, time, threading
import numpy as np
from scipy.spatial.distance import cdist
from config.settings import (DEFAULT_WEIGHTS, DEFAULT_THRESHOLDS, DEFAULT_WINDOW,
                              NUM_NODES, TIME_STEPS, GRID_SIZE, NEIGHBOR_RADIUS,
                              ABSOLUTE_THRESHOLD, Z_SCORE_THRESHOLD, EVENTS)


class GMSEngine:
    """
    Core GMS computation engine for 40-node sensor network.
    Supports real-time playback, noise injection, parameter tuning, SSE streaming.
    """

    def __init__(self):
        np.random.seed(2024)

        # ── Network geometry
        self.N = NUM_NODES
        self.T = TIME_STEPS
        self.G = GRID_SIZE
        self.RAD = NEIGHBOR_RADIUS

        # ── GMS parameters (from config, adjustable via API)
        self.w1, self.w2, self.w3, self.w4 = (DEFAULT_WEIGHTS['w1'], DEFAULT_WEIGHTS['w2'],
                                               DEFAULT_WEIGHTS['w3'], DEFAULT_WEIGHTS['w4'])
        self.theta = DEFAULT_THRESHOLDS['theta']
        self.window = DEFAULT_WINDOW
        self.alpha = DEFAULT_THRESHOLDS['alpha']
        self.beta = DEFAULT_THRESHOLDS['beta']

        # ── Baseline thresholds
        self.ABS_THRESH = ABSOLUTE_THRESHOLD
        self.Z_THRESH = Z_SCORE_THRESHOLD

        # ── Events
        self.EVENTS = EVENTS

        # ── Playback state
        self.t = 0
        self.playing = False
        self.speed = 0.25
        self.noise_on = False

        # ── Threading
        self._lock = threading.Lock()
        self._thread = None
        self._subs = []
        self._sub_lock = threading.Lock()

        # ── History
        self.alert_history = []
        self.alert_seq = 0
        self.alert_cooldown = 6
        self.hysteresis_margin = 0.05
        self.alert_persistence_frames = 2
        self.alert_runtime_label = np.zeros(self.N, dtype=int)
        self._high_candidate_count = np.zeros(self.N, dtype=int)
        self.hysteresis_suppressed = 0
        self._alert_last_seen = {}
        self.logs = []
        self.last_logged_t = None

        # ── Initialize data
        self._build()
        self._simulate()
        self._gms()

    # ── Network geometry ──────────────────────────────────────────────
    def _build(self):
        """Build spatial network: nodes + adjacency graph"""
        pos = np.random.uniform(0.5, self.G - 0.5, (self.N, 2))
        self.pos = pos

        d = cdist(pos, pos)
        self.adj = {i: [j for j in range(self.N) if j != i and d[i, j] <= self.RAD]
                    for i in range(self.N)}

    # ── Simulation: clean temperature ─────────────────────────────────
    def _simulate(self):
        """Generate base temperature + humidity with event injections"""
        t = np.linspace(0, 2*np.pi, self.T)
        T = np.zeros((self.N, self.T))
        H = np.zeros((self.N, self.T))

        # ── Base oscillation + spatial variation
        for i in range(self.N):
            e = (self.pos[i, 0] + self.pos[i, 1]) / (2 * self.G)
            T[i] = 22 + 6*np.sin(t - 0.3) + e*2.5 + np.random.normal(0, 0.25, self.T)
            H[i] = 65 - 8*np.sin(t) - e*3 + np.random.normal(0, 0.50, self.T)

        # ── Inject events
        for ev in self.EVENTS:
            dur = ev['t_end'] - ev['t_start']
            for i in ev['nodes']:
                r = np.zeros(self.T)
                r[ev['t_start']:ev['t_end']] = np.linspace(0, ev['dT'], dur)
                r[ev['t_end']:] = ev['dT'] * np.exp(-np.arange(self.T - ev['t_end']) / 12.)
                T[i] += r
                H[i] -= r * 0.8

        self._Temp_clean = T
        self.Humid = H

    # ── GMS pipeline ──────────────────────────────────────────────────
    def _n(self, x):
        """Min-max normalization [0,1]"""
        a, b = x.min(), x.max()
        return (x - a) / (b - a + 1e-12)

    def _gms(self):
        """Compute all GMS components and classifications"""
        N, T = self.N, self.T

        # ── Apply noise if enabled
        np.random.seed(42)
        if self.noise_on:
            self.Temp = self._Temp_clean + np.random.uniform(-0.5, 0.5, (N, T))
        else:
            self.Temp = self._Temp_clean.copy()

        # ── Component 1: Spatial Gradient ΔT_ij = T_i - T_j
        G2 = np.zeros((N, T))
        for i in range(N):
            nb = self.adj[i]
            if nb:
                G2[i] = np.array([self.Temp[i] - self.Temp[j] for j in nb]).mean(0)

        # ── Component 2: Temporal Momentum M = ΔT(t) - ΔT(t-1)
        M = np.zeros((N, T))
        M[:, 1:] = G2[:, 1:] - G2[:, :-1]

        # ── Component 3: Duration / Persistence
        D = np.zeros((N, T))
        for i in range(N):
            for t_idx in range(T):
                ws = max(0, t_idx - self.window + 1)
                D[i, t_idx] = np.mean(np.abs(G2[i, ws:t_idx+1]) > self.theta)

        # ── Component 4: Neighbor Influence Score
        NIS = G2.copy()
        a, b = NIS.min(), NIS.max()
        if b > a:
            NIS = (NIS - a) / (b - a)

        # ── GMS Composite Score S = w1|ΔT| + w2|M| + w3·NIS + w4·D
        raw = (self.w1 * self._n(np.abs(G2)) + self.w2 * self._n(np.abs(M)) +
               self.w3 * NIS + self.w4 * D)
        gms = np.clip(self._n(raw), 0, 1)

        lbl = np.zeros((N, T), dtype=int)
        lbl[gms >= self.alpha] = 1
        lbl[gms >= self.beta] = 2

        self.grad = G2
        self.mom = M
        self.dur = D
        self.nis = NIS
        self.gms = gms
        self.label = lbl

        # ── Baselines
        # 1. Absolute threshold
        self.baseline_abs = (self.Temp > self.ABS_THRESH).astype(int)

        # 2. Z-score per-node
        mu = self.Temp.mean(axis=1, keepdims=True)
        sig = self.Temp.std(axis=1, keepdims=True) + 1e-9
        self.z_scores = np.abs((self.Temp - mu) / sig)
        self.baseline_z = (self.z_scores > self.Z_THRESH).astype(int)

        # ── Onset times
        self.onset = np.full(N, np.inf)
        for i in range(N):
            ab = np.where(gms[i] > self.alpha)[0]
            if len(ab):
                self.onset[i] = ab[0]

    # ── Ground truth ──────────────────────────────────────────────────
    def _gt(self):
        """Build ground truth from EVENTS"""
        gt = np.zeros((self.N, self.T), dtype=int)
        for ev in self.EVENTS:
            for i in ev['nodes']:
                gt[i, ev['t_start']:ev['t_end']] = 1
        return gt

    def time_label(self, t):
        """Map a simulation timestep onto a 24-hour clock label."""
        t = max(0, min(int(t), self.T - 1))
        minutes = int(round((t / max(1, self.T)) * 24 * 60)) % (24 * 60)
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def time_axis(self):
        return [self.time_label(t) for t in range(self.T)]

    def _event_payload(self):
        events = []
        for ev in self.EVENTS:
            item = dict(ev)
            item["start_time"] = self.time_label(ev["t_start"])
            item["end_time"] = self.time_label(ev["t_end"])
            item["time_range"] = f"{item['start_time']}-{item['end_time']}"
            events.append(item)
        return events

    # ── Metrics ───────────────────────────────────────────────────────
    def _mets(self, p, gt):
        """Compute accuracy, precision, recall, FAR, F1"""
        TP = int(((p == 1) & (gt == 1)).sum())
        TN = int(((p == 0) & (gt == 0)).sum())
        FP = int(((p == 1) & (gt == 0)).sum())
        FN = int(((p == 0) & (gt == 1)).sum())

        tot = TP + TN + FP + FN
        acc = (TP + TN) / tot if tot else 0
        pr = TP / (TP + FP) if TP + FP else 0
        re = TP / (TP + FN) if TP + FN else 0
        fa = FP / (FP + TN) if FP + TN else 0
        f1 = 2*pr*re / (pr + re) if pr + re else 0

        return dict(acc=round(acc*100, 1), prec=round(pr*100, 1),
                    rec=round(re*100, 1), far=round(fa*100, 1), f1=round(f1*100, 1))

    def _perf(self):
        """Compare all 3 baselines + GMS"""
        gt = self._gt()
        pg = self._mets((self.gms >= self.alpha).astype(int), gt)
        pb = self._mets(self.baseline_abs, gt)
        pz = self._mets(self.baseline_z, gt)
        return pg, pb, pz

    def log_step(self, t):
        """Log all node data for this timestep (once)"""
        if hasattr(self, "last_logged_t") and self.last_logged_t == t:
            return
        self.last_logged_t = t

        gt = self._gt()
        for i in range(self.N):
            self.logs.append({
                "time": int(t),
                "time_label": self.time_label(t),
                "node": int(i),
                "temp": float(self.Temp[i, t]),
                "gradient": float(self.grad[i, t]),
                "momentum": float(self.mom[i, t]),
                "duration": float(self.dur[i, t]),
                "nis": float(self.nis[i, t]),
                "gms": float(self.gms[i, t]),
                "zscore": float(self.z_scores[i, t]),
                "pred": int(self.gms[i, t] >= self.alpha),
                "truth": int(gt[i, t])
            })

    # ── Noise toggle ──────────────────────────────────────────────────
    def toggle_noise(self, on):
        """Toggle noise injection and recompute GMS"""
        with self._lock:
            self.noise_on = on
            self._gms()
            self.alert_runtime_label = np.zeros(self.N, dtype=int)
            self._high_candidate_count = np.zeros(self.N, dtype=int)
            self.hysteresis_suppressed = 0
        self._bcast()
        self._alert(f"Noise {'ENABLED (+0.5°C random)' if on else 'DISABLED (clean data)'}", "info")

    # ── Rerun with new params ─────────────────────────────────────────
    def rerun(self, p):
        """Update parameters and recompute GMS"""
        with self._lock:
            for k in ['w1', 'w2', 'w3', 'w4', 'theta', 'alpha', 'beta', 'window']:
                if k in p:
                    try:
                        value = float(p[k])
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(value):
                        continue
                    setattr(self, k, value if k != 'window' else int(value))
            if 'hysteresis_margin' in p:
                try:
                    margin = float(p['hysteresis_margin'])
                    if np.isfinite(margin):
                        self.hysteresis_margin = max(0.0, min(margin, max(0.0, self.beta - self.alpha)))
                except (TypeError, ValueError):
                    pass
            if 'alert_persistence_frames' in p:
                try:
                    self.alert_persistence_frames = max(1, min(int(p['alert_persistence_frames']), 8))
                except (TypeError, ValueError):
                    pass
            self._gms()
            self.alert_runtime_label = np.zeros(self.N, dtype=int)
            self._high_candidate_count = np.zeros(self.N, dtype=int)
            self.hysteresis_suppressed = 0
        self._bcast()
        return {
            "alpha": round(float(self.alpha), 4),
            "beta": round(float(self.beta), 4),
            "hysteresis_margin": round(float(self.hysteresis_margin), 4),
            "alert_persistence_frames": int(self.alert_persistence_frames),
        }

    def set_hysteresis(self, margin):
        """Update the release margin used to suppress threshold flicker."""
        with self._lock:
            self.hysteresis_margin = max(0.0, min(float(margin), max(0.0, self.beta - self.alpha)))
            self.alert_runtime_label = np.zeros(self.N, dtype=int)
            self._high_candidate_count = np.zeros(self.N, dtype=int)
            self.hysteresis_suppressed = 0
        self._bcast()
        return {
            "margin": round(float(self.hysteresis_margin), 3),
            "high_enter": round(float(self.beta), 3),
            "high_release": round(float(max(0, self.beta - self.hysteresis_margin)), 3),
        }

    def set_alert_persistence(self, frames):
        with self._lock:
            self.alert_persistence_frames = max(1, min(int(frames), 8))
            self.alert_runtime_label = np.zeros(self.N, dtype=int)
            self._high_candidate_count = np.zeros(self.N, dtype=int)
            self.hysteresis_suppressed = 0
        self._bcast()
        return {"persistence_frames": int(self.alert_persistence_frames)}

    # ── Frame data payload ────────────────────────────────────────────
    def frame_data(self, t=None):
        """Serializable frame data for current timestep"""
        if t is None:
            t = self.t
        t = max(0, min(t, self.T - 1))

        ae = [ev['label'] for ev in self.EVENTS if ev['t_start'] <= t < ev['t_end']]

        pe = []
        for i in range(self.N):
            for j in self.adj[i]:
                if j > i:
                    oi, oj = self.onset[i], self.onset[j]
                    if (np.isfinite(oi) and np.isfinite(oj) and
                            t >= min(oi, oj) and abs(oj - oi) <= 20):
                        src = i if oi < oj else j
                        dst = j if oi < oj else i
                        pe.append({'src': int(src), 'dst': int(dst),
                                   'strength': float(np.exp(-abs(oj - oi) / 10))})

        pg, pb, pz = self._perf()

        return {
            't': int(t), 'T': self.T, 'time_label': self.time_label(t),
            'time_axis': self.time_axis(),
            'playing': self.playing, 'noise_on': self.noise_on,
            'high_count': int((self.label[:, t] == 2).sum()),
            'mod_count': int((self.label[:, t] == 1).sum()),
            'N': self.N,
            'nodes': [{'id': i, 'x': float(self.pos[i, 0]), 'y': float(self.pos[i, 1]),
                       'gms': round(float(self.gms[i, t]), 4),
                       'label': int(self.label[i, t]),
                       'grad': round(float(self.grad[i, t]), 4),
                       'mom': round(float(self.mom[i, t]), 4),
                       'dur': round(float(self.dur[i, t]), 4),
                       'nis': round(float(self.nis[i, t]), 4),
                       'temp': round(float(self.Temp[i, t]), 2),
                       'zscore': round(float(self.z_scores[i, t]), 3),
                       'onset': int(self.onset[i]) if np.isfinite(self.onset[i]) else None}
                      for i in range(self.N)],
            'adj': {str(i): self.adj[i] for i in range(self.N)},
            'prop_edges': pe, 'active_events': ae,
            'gms_full': self.gms.tolist(),
            'temp_full': self.Temp.tolist(),
            'grad_full': self.grad.tolist(),
            'mom_full': self.mom.tolist(),
            'dur_full': self.dur.tolist(),
            'nis_full': self.nis.tolist(),
            'zscore_full': self.z_scores.tolist(),
            'events': self._event_payload(),
            'alpha': self.alpha, 'beta': self.beta,
            'alert_policy': {
                'high_enter': round(float(self.beta), 3),
                'high_release': round(float(max(0, self.beta - self.hysteresis_margin)), 3),
                'moderate_enter': round(float(self.alpha), 3),
                'moderate_release': round(float(max(0, self.alpha - self.hysteresis_margin)), 3),
                'margin': round(float(self.hysteresis_margin), 3),
                'persistence_frames': int(self.alert_persistence_frames),
                'suppressed_flickers': int(self.hysteresis_suppressed),
            },
            'weights': {'w1': self.w1, 'w2': self.w2, 'w3': self.w3, 'w4': self.w4},
            'perf_gms': pg, 'perf_base': pb, 'perf_z': pz,
        }

    # ── SSE pub/sub ───────────────────────────────────────────────────
    def subscribe(self):
        """Add subscription queue for SSE streaming"""
        q = []
        with self._sub_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        """Remove subscription queue"""
        with self._sub_lock:
            if q in self._subs:
                self._subs.remove(q)

    def _bcast(self):
        """Broadcast frame to all subscribers"""
        d = json.dumps({'type': 'frame', 'data': self.frame_data()})
        with self._sub_lock:
            for q in self._subs:
                q.append(d)

    def _alert_action(self, level, category, details=None):
        details = details or {}
        nodes = details.get("nodes") or []
        if category == "group-risk" and len(nodes) >= 4:
            return "Possible spreading instability - monitor nearby nodes."
        if level == "danger":
            return "Escalate if this persists for 3 frames."
        if level == "warn":
            return "Monitor nearby nodes and confirm persistence."
        if level == "ok":
            return "Confirm recovery and close if conditions stay stable."
        if category == "system":
            return "System state changed."
        return "Check sensor calibration if the pattern looks isolated."

    def _resolve_node_alerts(self, nodes, t):
        if nodes is None:
            return
        if isinstance(nodes, (int, np.integer)):
            node_set = {int(nodes)}
        else:
            node_set = {int(node) for node in nodes}
        for alert in self.alert_history:
            alert_nodes = alert.get("nodes") or []
            if alert.get("node") is not None:
                alert_nodes = list(set(alert_nodes + [alert.get("node")]))
            if node_set.intersection({int(node) for node in alert_nodes}) and alert.get("status") in ("active", "acknowledged"):
                alert["status"] = "resolved"
                alert["resolved_t"] = int(t)

    def _alert(self, msg, level='info', node=None, category='system',
               details=None, nodes=None, suppress_key=None, cooldown=None):
        """Generate and broadcast alert."""
        t_now = int(self.t)
        if suppress_key:
            cooldown = self.alert_cooldown if cooldown is None else cooldown
            last_t = self._alert_last_seen.get(suppress_key)
            if last_t is not None and t_now - last_t < cooldown:
                return None
            self._alert_last_seen[suppress_key] = t_now

        if nodes is None:
            nodes = [] if node is None else [int(node)]
        else:
            nodes = [int(n) for n in nodes]

        if level == "ok":
            self._resolve_node_alerts(nodes, t_now)

        self.alert_seq += 1
        status = "active" if level in ("danger", "warn") else "resolved"
        details = details or {}
        if nodes and "nodes" not in details:
            details["nodes"] = nodes
        e = {
            'id': f"A{self.alert_seq:04d}",
            'msg': msg,
            'level': level,
            't': t_now,
            'time_label': self.time_label(t_now),
            'status': status,
            'node': None if node is None else int(node),
            'nodes': nodes,
            'category': category,
            'details': details or {},
            'action': self._alert_action(level, category, details),
            'created_at': time.time(),
            'ack_t': None,
            'resolved_t': t_now if status == "resolved" else None,
        }
        self.alert_history.insert(0, e)
        if len(self.alert_history) > 300:
            self.alert_history.pop()
        d = json.dumps({'type': 'alert', **e})
        with self._sub_lock:
            for q in self._subs:
                q.append(d)
        return e

    def set_alert_status(self, alert_id, status):
        """Acknowledge or resolve an alert by id."""
        if status not in {"active", "acknowledged", "resolved"}:
            return None
        t_now = int(self.t)
        for alert in self.alert_history:
            if alert.get("id") == alert_id:
                alert["status"] = status
                if status == "acknowledged":
                    alert["ack_t"] = t_now
                if status == "resolved":
                    alert["resolved_t"] = t_now
                return alert
        return None

    def clear_alerts(self):
        self.alert_history = []
        self._alert_last_seen = {}
        self.alert_runtime_label = np.zeros(self.N, dtype=int)
        self._high_candidate_count = np.zeros(self.N, dtype=int)
        self.hysteresis_suppressed = 0

    def _node_region(self, node, t):
        for ev in self.EVENTS:
            if node in ev["nodes"] and ev["t_start"] <= t < ev["t_end"]:
                return ev["label"]
        x, y = self.pos[node]
        east_west = "East" if x >= self.G / 2 else "West"
        north_south = "North" if y >= self.G / 2 else "South"
        return f"{north_south}-{east_west} region"

    def _group_nodes_by_region(self, nodes, t):
        groups = {}
        for node in nodes:
            groups.setdefault(self._node_region(int(node), t), []).append(int(node))
        return groups

    def _node_snapshot(self, nodes, t):
        nodes = [int(node) for node in nodes]
        if not nodes:
            return {}
        scores = [float(self.gms[node, t]) for node in nodes]
        temps = [float(self.Temp[node, t]) for node in nodes]
        return {
            "nodes": nodes,
            "gms": round(float(np.mean(scores)), 3),
            "max_gms": round(float(np.max(scores)), 3),
            "temp": round(float(np.mean(temps)), 2),
            "max_temp": round(float(np.max(temps)), 2),
            "gradient": round(float(np.mean([self.grad[node, t] for node in nodes])), 3),
            "momentum": round(float(np.mean([self.mom[node, t] for node in nodes])), 3),
            "duration": round(float(np.mean([self.dur[node, t] for node in nodes])), 3),
            "nis": round(float(np.mean([self.nis[node, t] for node in nodes])), 3),
        }

    def _next_alert_state(self, node, t, previous):
        score = float(self.gms[node, t])
        if previous == 2:
            if score >= self.beta - self.hysteresis_margin:
                if score < self.beta:
                    self.hysteresis_suppressed += 1
                self._high_candidate_count[node] = self.alert_persistence_frames
                return 2
            self._high_candidate_count[node] = 0
            return 1 if score >= self.alpha else 0
        if score >= self.beta:
            self._high_candidate_count[node] += 1
            if self._high_candidate_count[node] >= self.alert_persistence_frames:
                return 2
            return 1 if score >= self.alpha else previous
        self._high_candidate_count[node] = 0
        if previous == 1:
            if score < self.alpha - self.hysteresis_margin:
                return 0
            return 1
        if score >= self.alpha:
            return 1
        return 0

    def _emit_grouped_node_alerts(self, t, previous, current):
        high_nodes = [i for i in range(self.N) if previous[i] < 2 and current[i] == 2]
        mod_nodes = [i for i in range(self.N) if previous[i] == 0 and current[i] == 1]
        eased_nodes = [i for i in range(self.N) if previous[i] == 2 and current[i] == 1]
        stable_nodes = [i for i in range(self.N) if previous[i] > 0 and current[i] == 0]

        for region, nodes in self._group_nodes_by_region(high_nodes, t).items():
            details = self._node_snapshot(nodes, t)
            details["region"] = region
            node_list = ", ".join(f"N{node}" for node in nodes)
            self._alert(
                f"{len(nodes)} high-risk node{'s' if len(nodes) != 1 else ''} detected in {region}: {node_list}",
                "danger",
                node=nodes[0] if len(nodes) == 1 else None,
                nodes=nodes,
                category="group-risk",
                details=details,
                suppress_key=f"group:high:{region}",
            )

        for region, nodes in self._group_nodes_by_region(mod_nodes, t).items():
            details = self._node_snapshot(nodes, t)
            details["region"] = region
            node_list = ", ".join(f"N{node}" for node in nodes)
            self._alert(
                f"{len(nodes)} moderate-risk node{'s' if len(nodes) != 1 else ''} detected in {region}: {node_list}",
                "warn",
                node=nodes[0] if len(nodes) == 1 else None,
                nodes=nodes,
                category="group-risk",
                details=details,
                suppress_key=f"group:moderate:{region}",
            )

        if eased_nodes:
            self._resolve_node_alerts(eased_nodes, t)
            for region, nodes in self._group_nodes_by_region(eased_nodes, t).items():
                node_list = ", ".join(f"N{node}" for node in nodes)
                self._alert(
                    f"{len(nodes)} high-risk node{'s' if len(nodes) != 1 else ''} eased to moderate in {region}: {node_list}",
                    "warn",
                    nodes=nodes,
                    category="recovery",
                    details={**self._node_snapshot(nodes, t), "region": region},
                    suppress_key=f"group:eased:{region}",
                )

        for region, nodes in self._group_nodes_by_region(stable_nodes, t).items():
            node_list = ", ".join(f"N{node}" for node in nodes)
            self._alert(
                f"{len(nodes)} node{'s' if len(nodes) != 1 else ''} recovered in {region}: {node_list}",
                "ok",
                node=nodes[0] if len(nodes) == 1 else None,
                nodes=nodes,
                category="recovery",
                details={**self._node_snapshot(nodes, t), "region": region},
                suppress_key=f"group:stable:{region}",
            )

    def alert_analytics(self):
        total = len(self.alert_history)
        critical = len([a for a in self.alert_history if a.get("level") == "danger"])
        active = len([a for a in self.alert_history if a.get("status") != "resolved"])
        acknowledged = len([a for a in self.alert_history if a.get("status") == "acknowledged"])
        resolved = [a for a in self.alert_history if a.get("resolved_t") is not None]
        avg_response = 0
        if resolved:
            avg_response = sum(max(0, int(a["resolved_t"]) - int(a["t"])) for a in resolved) / len(resolved)
        t = int(self.t)
        peak_node = int(np.argmax(self.gms[:, t]))
        risk_alerts = [a for a in self.alert_history if a.get("level") in ("danger", "warn")]
        gt = self._gt()
        false_count = 0
        for alert in risk_alerts:
            nodes = alert.get("nodes") or ([] if alert.get("node") is None else [alert.get("node")])
            at = max(0, min(int(alert.get("t", 0)), self.T - 1))
            if nodes and not any(gt[int(node), at] for node in nodes):
                false_count += 1
        false_est = (false_count / len(risk_alerts) * 100) if risk_alerts else 0
        trend = []
        for idx in range(self.T):
            count = len([a for a in self.alert_history if int(a.get("t", -1)) == idx])
            if count:
                trend.append({"t": idx, "count": count})
        for item in trend:
            item["time_label"] = self.time_label(item["t"])
        return {
            "alerts_today": total,
            "critical_count": critical,
            "active_count": active,
            "acknowledged_count": acknowledged,
            "most_unstable_node": peak_node,
            "most_unstable_gms": round(float(self.gms[peak_node, t]), 3),
            "avg_response_frames": round(avg_response, 1),
            "false_alert_estimate": round(false_est, 1),
            "hysteresis_margin": round(float(self.hysteresis_margin), 3),
            "high_enter": round(float(self.beta), 3),
            "high_release": round(float(max(0, self.beta - self.hysteresis_margin)), 3),
            "persistence_frames": int(self.alert_persistence_frames),
            "suppressed_flickers": int(self.hysteresis_suppressed),
            "trend": trend[-24:],
        }

    def _simulate_alert_policy(self, alpha, beta, margin, persistence):
        gt = self._gt()
        pred = np.zeros((self.N, self.T), dtype=int)
        labels = np.zeros(self.N, dtype=int)
        high_count = np.zeros(self.N, dtype=int)
        suppressed = 0
        transitions = 0
        for t in range(self.T):
            for node in range(self.N):
                score = float(self.gms[node, t])
                prev = int(labels[node])
                if prev == 2:
                    if score >= beta - margin:
                        if score < beta:
                            suppressed += 1
                        nxt = 2
                        high_count[node] = persistence
                    else:
                        high_count[node] = 0
                        nxt = 1 if score >= alpha else 0
                else:
                    if score >= beta:
                        high_count[node] += 1
                        nxt = 2 if high_count[node] >= persistence else (1 if score >= alpha else prev)
                    else:
                        high_count[node] = 0
                        if prev == 1 and score >= alpha - margin:
                            nxt = 1
                        elif score >= alpha:
                            nxt = 1
                        else:
                            nxt = 0
                if nxt != prev:
                    transitions += 1
                labels[node] = nxt
                pred[node, t] = int(nxt > 0)

        TP = int(((pred == 1) & (gt == 1)).sum())
        TN = int(((pred == 0) & (gt == 0)).sum())
        FP = int(((pred == 1) & (gt == 0)).sum())
        FN = int(((pred == 0) & (gt == 1)).sum())
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total else 0
        precision = TP / (TP + FP) if TP + FP else 0
        recall = TP / (TP + FN) if TP + FN else 0
        far = FP / (FP + TN) if FP + TN else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "far": far,
            "f1": f1,
            "tp": TP,
            "fp": FP,
            "fn": FN,
            "tn": TN,
            "transitions": transitions,
            "suppressed": suppressed,
        }

    def tune_alert_policy(self, iterations=350, seed=42, target_far=0.02, min_recall=0.25, apply=False):
        rng = np.random.default_rng(int(seed))
        best = None
        history = []
        iterations = max(20, min(int(iterations), 2000))
        target_far = max(0.0, min(float(target_far), 0.5))
        min_recall = max(0.0, min(float(min_recall), 1.0))

        candidates = [
            (self.alpha, self.beta, self.hysteresis_margin, self.alert_persistence_frames),
            (0.25, 0.60, 0.05, 2),
            (0.30, 0.65, 0.06, 2),
            (0.35, 0.70, 0.08, 3),
        ]
        for _ in range(iterations):
            alpha = float(rng.uniform(0.10, 0.50))
            beta = float(rng.uniform(max(0.45, alpha + 0.12), 0.92))
            margin = float(rng.uniform(0.0, min(0.18, beta - alpha - 0.02)))
            persistence = int(rng.integers(1, 5))
            candidates.append((alpha, beta, margin, persistence))

        for idx, (alpha, beta, margin, persistence) in enumerate(candidates):
            metrics = self._simulate_alert_policy(alpha, beta, margin, persistence)
            quiet = 1 - metrics["far"]
            transition_penalty = min(0.20, metrics["transitions"] / max(1, self.N * self.T) * 2)
            far_penalty = max(0, metrics["far"] - target_far) * 2.5
            recall_penalty = max(0, min_recall - metrics["recall"]) * 1.4
            score = (
                metrics["precision"] * 0.34 +
                metrics["f1"] * 0.28 +
                metrics["accuracy"] * 0.20 +
                quiet * 0.18 -
                far_penalty -
                recall_penalty -
                transition_penalty
            )
            item = {
                "eval": idx + 1,
                "score": float(score),
                "params": {
                    "alpha": round(float(alpha), 4),
                    "beta": round(float(beta), 4),
                    "hysteresis_margin": round(float(margin), 4),
                    "alert_persistence_frames": int(persistence),
                },
                "metrics": metrics,
            }
            history.append(item)
            if best is None or score > best["score"]:
                best = item

        if apply and best:
            self.rerun(best["params"])

        compact = []
        stride = max(1, len(history) // 120)
        best_seen = None
        for idx, item in enumerate(history):
            if idx % stride and idx != len(history) - 1:
                continue
            best_seen = item["score"] if best_seen is None else max(best_seen, item["score"])
            compact.append({
                "eval": item["eval"],
                "score": item["score"],
                "best_score": float(best_seen),
                "precision": item["metrics"]["precision"],
                "recall": item["metrics"]["recall"],
                "far": item["metrics"]["far"],
                "f1": item["metrics"]["f1"],
            })

        return {
            "best": best,
            "history": compact,
            "evaluations": len(history),
            "target_far": target_far,
            "min_recall": min_recall,
        }

    # ── Playback control ──────────────────────────────────────────────
    def play(self):
        """Start playback"""
        with self._lock:
            self.playing = True
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def pause(self):
        """Pause playback"""
        with self._lock:
            self.playing = False

    def reset(self):
        """Reset to t=0"""
        with self._lock:
            self.playing = False
            self.t = 0
            self.alert_runtime_label = np.zeros(self.N, dtype=int)
            self._high_candidate_count = np.zeros(self.N, dtype=int)
        self._bcast()

    def jump(self, t):
        """Jump to time t"""
        with self._lock:
            self.t = max(0, min(t, self.T - 1))
        self._bcast()

    def step(self, d=1):
        """Step forward/backward by d timesteps"""
        with self._lock:
            self.t = max(0, min(self.t + d, self.T - 1))
        self._bcast()

    def trigger(self, idx):
        """Trigger event by index"""
        ev = self.EVENTS[idx]
        self.jump(ev['t_start'])
        self.play()
        self._alert(
            f"Triggered {ev['label']} - nodes {ev['nodes']}",
            "warn",
            category="event",
            details={"event": ev["label"], "nodes": ev["nodes"]},
            suppress_key=f"event:{ev['label']}",
            cooldown=2,
        )

    def _loop(self):
        """Main playback loop"""
        while True:
            with self._lock:
                if not self.playing:
                    break
                self.t = (self.t + 1) % self.T
                nt = self.t

            previous = self.alert_runtime_label.copy()
            current = np.array([
                self._next_alert_state(i, nt, int(previous[i]))
                for i in range(self.N)
            ], dtype=int)
            self.alert_runtime_label = current
            self._emit_grouped_node_alerts(nt, previous, current)

            self.log_step(nt)
            self._bcast()
            time.sleep(self.speed)
