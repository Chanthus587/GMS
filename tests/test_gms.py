"""
tests/test_gms.py
─────────────────────────────────────────────────────────
Unit tests for the GMS model components.
Run from the project root:
    python -m pytest tests/ -v
or:
    python tests/test_gms.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from data.loader       import SimulatedData
from core.gms_model    import GMSModel
from evaluation.metrics import (BaselineDetector, Evaluator,
                                  build_ground_truth)


def make_env_and_model():
    env = SimulatedData()
    gms = GMSModel(env)
    return env, gms


# ── Test 1: Shapes ────────────────────────────────────────────────────────────
def test_output_shapes():
    env, gms = make_env_and_model()
    N, T = env.N, env.T
    assert gms.grad.shape  == (N, T), "grad shape mismatch"
    assert gms.mom.shape   == (N, T), "mom shape mismatch"
    assert gms.dur.shape   == (N, T), "dur shape mismatch"
    assert gms.nis.shape   == (N, T), "nis shape mismatch"
    assert gms.gms.shape   == (N, T), "gms shape mismatch"
    assert gms.label.shape == (N, T), "label shape mismatch"
    print("[PASS] test_output_shapes")


# ── Test 2: GMS range ─────────────────────────────────────────────────────────
def test_gms_range():
    _, gms = make_env_and_model()
    assert gms.gms.min() >= 0.0 - 1e-6, "GMS below 0"
    assert gms.gms.max() <= 1.0 + 1e-6, "GMS above 1"
    print("[PASS] test_gms_range")


# ── Test 3: Duration is a fraction in [0,1] ───────────────────────────────────
def test_duration_range():
    _, gms = make_env_and_model()
    assert gms.dur.min() >= 0.0, "Duration below 0"
    assert gms.dur.max() <= 1.0 + 1e-9, "Duration above 1"
    print("[PASS] test_duration_range")


# ── Test 4: Momentum is zero at t=0 ──────────────────────────────────────────
def test_momentum_t0_zero():
    _, gms = make_env_and_model()
    assert np.allclose(gms.mom[:, 0], 0.0), "Momentum at t=0 should be 0"
    print("[PASS] test_momentum_t0_zero")


# ── Test 5: NIS in [0,1] ─────────────────────────────────────────────────────
def test_nis_range():
    _, gms = make_env_and_model()
    assert gms.nis.min() >= 0.0 - 1e-9, "NIS below 0"
    assert gms.nis.max() <= 1.0 + 1e-9, "NIS above 1"
    print("[PASS] test_nis_range")


# ── Test 6: Labels are 0, 1, or 2 ────────────────────────────────────────────
def test_label_values():
    _, gms = make_env_and_model()
    unique = np.unique(gms.label)
    assert set(unique).issubset({0, 1, 2}), f"Unexpected labels: {unique}"
    print("[PASS] test_label_values")


# ── Test 7: Instability detected in event nodes ───────────────────────────────
def test_instability_detected_in_events():
    import config
    env, gms = make_env_and_model()
    for ev in config.EVENTS:
        for node in ev['nodes']:
            if node >= env.N:
                continue
            window_gms = gms.gms[node, ev['t_start']:ev['t_end']]
            max_gms = window_gms.max()
            assert max_gms > gms.alpha, (
                f"Event {ev['label']} node {node}: max GMS={max_gms:.3f} "
                f"did not exceed alpha={gms.alpha}")
    print("[PASS] test_instability_detected_in_events")


# ── Test 8: Adjacency is symmetric ───────────────────────────────────────────
def test_adjacency_symmetric():
    env, _ = make_env_and_model()
    for i in range(env.N):
        for j in env.adj[i]:
            assert i in env.adj[j], f"Adjacency not symmetric: {i}↔{j}"
    print("[PASS] test_adjacency_symmetric")


# ── Test 9: Baseline detector ─────────────────────────────────────────────────
def test_baseline():
    env, _ = make_env_and_model()
    bl = BaselineDetector(env, threshold=22.0)  # low threshold → lots of alarms
    assert bl.label.shape == (env.N, env.T)
    assert bl.label.max() == 1
    print("[PASS] test_baseline")


# ── Test 10: Evaluator metrics in [0,1] ──────────────────────────────────────
def test_evaluator_ranges():
    env, gms = make_env_and_model()
    bl       = BaselineDetector(env)
    gt       = build_ground_truth(env.N, env.T)
    pred     = (gms.gms >= gms.alpha).astype(int)
    ev       = Evaluator(gt, pred, bl.label)
    for key in ('accuracy','precision','recall','far','f1'):
        val = ev.metrics_gms[key]
        assert 0.0 <= val <= 1.0, f"{key} = {val} out of [0,1]"
    print("[PASS] test_evaluator_ranges")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tests = [
        test_output_shapes,
        test_gms_range,
        test_duration_range,
        test_momentum_t0_zero,
        test_nis_range,
        test_label_values,
        test_instability_detected_in_events,
        test_adjacency_symmetric,
        test_baseline,
        test_evaluator_ranges,
    ]
    print("\n" + "="*50)
    print("  GMS Unit Tests")
    print("="*50)
    passed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {fn.__name__}: {e}")
        except Exception as e:
            print(f"[ERROR] {fn.__name__}: {e}")
    print(f"\n  {passed}/{len(tests)} tests passed")
    print("="*50 + "\n")
