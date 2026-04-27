"""
core/gms_model.py
─────────────────────────────────────────────────────────
Gradient-Momentum Score (GMS) Model
────────────────────────────────────
Implements every component from Section III of the paper:

  Component 1 — Spatial Gradient          ΔT_ij(t) = T_i(t) − T_j(t)
  Component 2 — Temporal Momentum         M_ij(t)  = ΔT(t) − ΔT(t−1)
  Component 3 — Duration / Persistence    D_i(t)   (Eq. 5 in paper)
  Component 4 — Neighbor Influence Score  NIS_i(t) (Eq. 6 in paper)
  Composite   — GMS Score                 S_i(t)   (Eq. 7 in paper)
  Classifier  — Stable / Mod / High       (Section III-H)

All arrays use shape (N, T) convention: axis-0 = nodes, axis-1 = time.

Complexity: O(N·k) per time step  →  O(N) for fixed neighbour degree.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from data.loader import SensorEnvironment


class GMSModel:
    """
    Parameters
    ──────────
    env    : SensorEnvironment  — data container (SimulatedData or NASAPower)
    w1–w4  : float              — component weights  (default from config.py)
    theta  : float              — gradient threshold for Duration count
    window : int                — sliding window length for persistence

    Key outputs (all shape  N × T)
    ──────────────────────────────
    grad    — mean signed gradient ΔT at each node
    mom     — temporal momentum M
    dur     — duration / persistence score  [0,1]
    nis     — Neighbor Influence Score      [0,1]
    gms     — composite GMS score           [0,1]
    label   — integer classification  0=Stable 1=Mod 2=High
    """

    STABLE       = 0
    MOD_UNSTABLE = 1
    HIGH_UNSTABLE= 2

    def __init__(self, env: SensorEnvironment,
                 w1=None, w2=None, w3=None, w4=None,
                 theta=None, window=None):

        self.env = env

        # ── Load defaults from config if not overridden ───────────────────
        W = config.WEIGHTS
        A = config.ALGO
        self.w1     = w1     if w1     is not None else W['w1']
        self.w2     = w2     if w2     is not None else W['w2']
        self.w3     = w3     if w3     is not None else W['w3']
        self.w4     = w4     if w4     is not None else W['w4']
        self.theta  = theta  if theta  is not None else A['theta']
        self.win    = window if window is not None else A['window']
        self.alpha  = A['alpha']
        self.beta   = A['beta']

        # ── Run the pipeline ─────────────────────────────────────────────
        print("[GMS] Computing Spatial Gradient …")
        self.grad = self._compute_gradient()

        print("[GMS] Computing Temporal Momentum …")
        self.mom  = self._compute_momentum()

        print("[GMS] Computing Duration / Persistence …")
        self.dur  = self._compute_duration()

        print("[GMS] Computing Neighbor Influence Score …")
        self.nis  = self._compute_nis()

        print("[GMS] Computing Composite GMS Score …")
        self.gms  = self._compute_gms()

        print("[GMS] Classifying nodes …")
        self.label = self._classify()

        self._print_summary()

    # ══════════════════════════════════════════════════════════════════════
    #  COMPONENT 1 — Spatial Gradient
    # ══════════════════════════════════════════════════════════════════════

    def _compute_gradient(self):
        """
        For each node i at time t:
            ΔT_i(t) = mean_j [ T_i(t) - T_j(t) ]   for j ∈ N(i)

        This is the per-node mean gradient, averaged over all
        immediate neighbours.  Shape: (N, T).
        """
        N, T = self.env.N, self.env.T
        G = np.zeros((N, T))
        for i in range(N):
            nb = self.env.adj[i]
            if not nb:
                continue
            # vectorised over time
            diffs = np.array([self.env.Temp[i] - self.env.Temp[j]
                              for j in nb])          # (k, T)
            G[i] = diffs.mean(axis=0)
        return G

    # ══════════════════════════════════════════════════════════════════════
    #  COMPONENT 2 — Temporal Momentum
    # ══════════════════════════════════════════════════════════════════════

    def _compute_momentum(self):
        """
        M_i(t) = ΔT_i(t) - ΔT_i(t-1)

        First-order finite difference of the gradient time series.
        M[:,0] = 0 (no previous step).  Shape: (N, T).
        """
        M = np.zeros_like(self.grad)
        M[:, 1:] = self.grad[:, 1:] - self.grad[:, :-1]
        return M

    # ══════════════════════════════════════════════════════════════════════
    #  COMPONENT 3 — Duration / Persistence  (Eq. 5 in paper)
    # ══════════════════════════════════════════════════════════════════════

    def _compute_duration(self):
        """
        D_i(t) = fraction of the last `window` steps where |ΔT_i| > theta.

        Paper equation 5 uses a cumulative counter; we expose a
        normalised [0,1] version (fraction of window) so it is directly
        comparable with the other components in the weighted sum.
        Shape: (N, T).
        """
        N, T = self.env.N, self.env.T
        D = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                ws   = max(0, t - self.win + 1)
                slab = np.abs(self.grad[i, ws: t + 1])
                D[i, t] = np.mean(slab > self.theta)
        return D

    # ══════════════════════════════════════════════════════════════════════
    #  COMPONENT 4 — Neighbor Influence Score  (Eq. 6 in paper)
    # ══════════════════════════════════════════════════════════════════════

    def _compute_nis(self):
        """
        NIS_i(t) = (1/|N(i)|) · Σ_{j ∈ N(i)} ΔT_ij(t)

        We use the raw gradient values (positive if T_i > T_j on average).
        Then normalise to [0,1] globally so the weight w3 is meaningful.
        Shape: (N, T).
        """
        N, T = self.env.N, self.env.T
        NIS = np.zeros((N, T))

        for i in range(N):
            nb = self.env.adj[i]
            if not nb:
                continue
            diffs = np.array([self.env.Temp[i] - self.env.Temp[j]
                              for j in nb])          # (k, T)
            NIS[i] = diffs.mean(axis=0)

        # Normalise to [0, 1]
        mn, mx = NIS.min(), NIS.max()
        if mx > mn:
            NIS = (NIS - mn) / (mx - mn)
        return NIS

    # ══════════════════════════════════════════════════════════════════════
    #  COMPOSITE GMS SCORE  (Eq. 7 in paper)
    # ══════════════════════════════════════════════════════════════════════

    def _compute_gms(self):
        """
        S_i(t) = w1·|ΔT_i| + w2·|M_i| + w3·NIS_i + w4·D_i

        Each term is normalised to [0,1] before weighting so the
        magnitude of different physical units does not bias the score.
        Final score also clipped to [0,1].
        """
        def norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-12)

        raw = (self.w1 * norm(np.abs(self.grad)) +
               self.w2 * norm(np.abs(self.mom))  +
               self.w3 * self.nis                 +
               self.w4 * self.dur)

        return np.clip(norm(raw), 0.0, 1.0)

    # ══════════════════════════════════════════════════════════════════════
    #  CLASSIFICATION  (Section III-H)
    # ══════════════════════════════════════════════════════════════════════

    def _classify(self):
        """
        0 → Stable          (GMS < alpha)
        1 → Mod. Unstable   (alpha ≤ GMS < beta)
        2 → High Unstable   (GMS ≥ beta)
        """
        lbl = np.zeros_like(self.gms, dtype=int)
        lbl[self.gms >= self.alpha] = self.MOD_UNSTABLE
        lbl[self.gms >= self.beta]  = self.HIGH_UNSTABLE
        return lbl

    # ══════════════════════════════════════════════════════════════════════
    #  UTILITY
    # ══════════════════════════════════════════════════════════════════════

    def _print_summary(self):
        n_high = int((self.label == self.HIGH_UNSTABLE).sum())
        n_mod  = int((self.label == self.MOD_UNSTABLE).sum())
        peak_t = int(np.argmax(self.gms.mean(axis=0)))
        print(f"[GMS] Peak instability @ t = {peak_t}")
        print(f"[GMS] Max GMS score        = {self.gms.max():.4f}")
        print(f"[GMS] High-Unstable cells  = {n_high}")
        print(f"[GMS] Mod-Unstable  cells  = {n_mod}")

    def onset_times(self, threshold=None):
        """
        Return array (N,) of first time step where GMS exceeds threshold.
        np.inf if node never exceeds threshold.
        """
        thr = threshold if threshold is not None else self.alpha
        onset = np.full(self.env.N, np.inf)
        for i in range(self.env.N):
            above = np.where(self.gms[i] > thr)[0]
            if len(above):
                onset[i] = above[0]
        return onset

    def node_report(self, node_id, t):
        """
        Human-readable breakdown of GMS score for node `node_id` at time `t`.
        """
        i = node_id
        print(f"\n── Node N{i}  @  t = {t} ──────────────────────────────")
        print(f"  Gradient  |ΔT|  : {abs(self.grad[i, t]):.4f}")
        print(f"  Momentum  |M|   : {abs(self.mom[i,  t]):.4f}")
        print(f"  Duration  D     : {self.dur[i,   t]:.4f}")
        print(f"  NIS             : {self.nis[i,   t]:.4f}")
        print(f"  GMS Score       : {self.gms[i,   t]:.4f}")
        cls = ['Stable', 'Moderately Unstable', 'Highly Unstable']
        print(f"  Classification  : {cls[self.label[i, t]]}")
