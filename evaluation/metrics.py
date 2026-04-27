"""
evaluation/metrics.py
─────────────────────────────────────────────────────────
Evaluation metrics and baseline comparison for the GMS model.

Classes
───────
  BaselineDetector   — conventional absolute-threshold detector
  Evaluator          — computes Accuracy, Precision, Recall, FAR
                       and prints the paper's Table I
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from data.loader import SensorEnvironment
from core.gms_model import GMSModel


# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE: Absolute Threshold Detector
# ══════════════════════════════════════════════════════════════════════════════

class BaselineDetector:
    """
    Traditional approach: alarm fires when T_i(t) > abs_threshold.
    No spatial or temporal relationships are considered.

    label : ndarray (N, T)  — 0 = no alarm, 1 = alarm
    """

    def __init__(self, env: SensorEnvironment, threshold=None):
        thr = threshold if threshold is not None else config.BASELINE['abs_threshold']
        self.label = (env.Temp > thr).astype(int)
        self.threshold = thr
        print(f"[Baseline] Absolute threshold = {thr} °C")
        print(f"[Baseline] Alarm cells        = {self.label.sum()}")


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD GROUND TRUTH FROM INJECTION EVENTS
# ══════════════════════════════════════════════════════════════════════════════

def build_ground_truth(N, T):
    """
    Construct a binary ground-truth matrix from config.EVENTS.
    gt[i, t] = 1 if node i is inside an injected instability window.
    """
    gt = np.zeros((N, T), dtype=int)
    for ev in config.EVENTS:
        for i in ev['nodes']:
            if i < N:
                gt[i, ev['t_start']:ev['t_end']] = 1
    return gt


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class Evaluator:
    """
    Compares two binary detection arrays against ground truth.

    Parameters
    ──────────
    gt          : ndarray (N, T)  — ground truth  (1 = truly unstable)
    pred_gms    : ndarray (N, T)  — GMS predictions (1 if GMS ≥ alpha)
    pred_base   : ndarray (N, T)  — Baseline predictions

    All metrics are computed globally (flattened over N×T).
    """

    def __init__(self, gt, pred_gms, pred_base):
        self.gt        = gt.ravel().astype(int)
        self.pred_gms  = pred_gms.ravel().astype(int)
        self.pred_base = pred_base.ravel().astype(int)

        self.metrics_gms  = self._compute(self.pred_gms)
        self.metrics_base = self._compute(self.pred_base)

    def _compute(self, pred):
        gt = self.gt
        TP = int(((pred == 1) & (gt == 1)).sum())
        TN = int(((pred == 0) & (gt == 0)).sum())
        FP = int(((pred == 1) & (gt == 0)).sum())
        FN = int(((pred == 0) & (gt == 1)).sum())

        total      = TP + TN + FP + FN
        accuracy   = (TP + TN) / total   if total  > 0 else 0.0
        precision  = TP / (TP + FP)      if TP+FP  > 0 else 0.0
        recall     = TP / (TP + FN)      if TP+FN  > 0 else 0.0
        far        = FP / (FP + TN)      if FP+TN  > 0 else 0.0
        f1         = (2 * precision * recall / (precision + recall)
                      if precision + recall > 0 else 0.0)

        return dict(TP=TP, TN=TN, FP=FP, FN=FN,
                    accuracy=accuracy, precision=precision,
                    recall=recall, far=far, f1=f1)

    def early_detection_lead(self, env, gms_model: GMSModel,
                             baseline: BaselineDetector):
        """
        For each node that appears in an event, compute how many time steps
        earlier GMS fires compared to the baseline.
        Returns array of lead times (positive = GMS fires first).
        """
        leads = []
        for ev in config.EVENTS:
            for i in ev['nodes']:
                if i >= env.N:
                    continue
                # First GMS alarm in event window
                gms_alarms  = np.where(gms_model.gms[i, ev['t_start']:] >=
                                       gms_model.alpha)[0]
                base_alarms = np.where(baseline.label[i, ev['t_start']:] == 1)[0]

                t_gms  = gms_alarms[0]  if len(gms_alarms)  else np.inf
                t_base = base_alarms[0] if len(base_alarms) else np.inf
                if np.isfinite(t_base) and np.isfinite(t_gms):
                    leads.append(int(t_base - t_gms))
        return np.array(leads)

    def print_table(self, leads=None):
        """Print the performance comparison table (Table I in the paper)."""
        g = self.metrics_gms
        b = self.metrics_base

        def pct(v): return f"{v*100:.1f}%"

        print("\n" + "═"*62)
        print("  TABLE I  Performance Comparison: Proposed vs. Baseline")
        print("═"*62)
        print(f"  {'Metric':<22} {'Threshold':>12} {'GMS (Proposed)':>15}")
        print("─"*62)
        rows = [
            ("Accuracy",         b['accuracy'],  g['accuracy']),
            ("Precision",        b['precision'], g['precision']),
            ("Recall",           b['recall'],    g['recall']),
            ("False Alarm Rate", b['far'],       g['far']),
            ("F1-Score",         b['f1'],        g['f1']),
        ]
        for name, bv, gv in rows:
            imp = gv - bv if name != "False Alarm Rate" else bv - gv
            sign = "+" if imp >= 0 else ""
            print(f"  {name:<22} {pct(bv):>12} {pct(gv):>15}   ({sign}{imp*100:.1f}%)")
        print("═"*62)

        if leads is not None and len(leads):
            print(f"\n  Early detection lead  (mean ± std): "
                  f"{leads.mean():.1f} ± {leads.std():.1f} time steps")
            print(f"  Min lead: {leads.min()}  |  Max lead: {leads.max()}")
        print()
