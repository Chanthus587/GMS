"""
utils/metrics.py
─────────────────────────────────────────────────────────
Evaluation metrics and baseline comparison for the GMS model.

Classes
───────
  BaselineDetector   — conventional absolute-threshold detector
  Evaluator          — computes Accuracy, Precision, Recall, FAR
"""

import numpy as np
from config.settings import ABSOLUTE_THRESHOLD, Z_SCORE_THRESHOLD, EVENTS


class BaselineDetector:
    """
    Traditional approach: alarm fires when T_i(t) > abs_threshold.
    No spatial or temporal relationships are considered.

    label : ndarray (N, T)  — 0 = no alarm, 1 = alarm
    """

    def __init__(self, Temp, threshold=None):
        thr = threshold if threshold is not None else ABSOLUTE_THRESHOLD
        self.label = (Temp > thr).astype(int)
        self.threshold = thr


def build_ground_truth(N, T):
    """
    Construct a binary ground-truth matrix from config.EVENTS.
    gt[i, t] = 1 if node i is inside an injected instability window.
    """
    gt = np.zeros((N, T), dtype=int)
    for ev in EVENTS:
        for i in ev['nodes']:
            if i < N:
                gt[i, ev['t_start']:ev['t_end']] = 1
    return gt


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
