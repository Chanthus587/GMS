"""
main.py
─────────────────────────────────────────────────────────
Entry point for the GMS Microclimate Instability Detection System.

Run:
    python main.py                   # simulated data (default)
    python main.py --data nasa       # real NASA POWER data (needs internet)
    python main.py --node 3          # decompose node 3 instead of 0
    python main.py --no-plots        # skip figures, print metrics only
"""

import sys
import os
import argparse
import time

# ── ensure project root is on the path ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import config
from data.loader       import SimulatedData, NASAPowerData
from core.gms_model    import GMSModel
from evaluation.metrics import (BaselineDetector, Evaluator,
                                 build_ground_truth)
from visualization.plots import plot_all


def parse_args():
    p = argparse.ArgumentParser(
        description='GMS Microclimate Instability Detection')
    p.add_argument('--data',     choices=['sim','nasa'], default='sim',
                   help='Data source: sim (default) or nasa')
    p.add_argument('--node',     type=int, default=0,
                   help='Node index for component decomposition plot')
    p.add_argument('--no-plots', action='store_true',
                   help='Skip figure generation')
    p.add_argument('--out',      default='outputs',
                   help='Output directory for figures')
    return p.parse_args()


def main():
    args  = parse_args()
    start = time.time()

    print("=" * 65)
    print("  GMS — Gradient-Momentum Microclimate Instability System")
    print("  Paper: Gradient-Momentum Based Microclimate Instability")
    print("         Detection Using Spatial-Temporal Sensor Networks")
    print("=" * 65)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[Step 1/4]  Loading sensor data …")
    if args.data == 'nasa':
        env = NASAPowerData()
    else:
        env = SimulatedData()

    # ── 2. Run GMS model ─────────────────────────────────────────────────
    print("\n[Step 2/4]  Running GMS model …")
    gms = GMSModel(env)

    # Per-node report for the chosen node
    t_peak = int(gms.gms.mean(axis=0).argmax())
    gms.node_report(args.node, t_peak)

    # ── 3. Evaluate ──────────────────────────────────────────────────────
    print("\n[Step 3/4]  Evaluating performance …")
    baseline = BaselineDetector(env)
    gt       = build_ground_truth(env.N, env.T)

    gms_pred  = (gms.gms >= gms.alpha).astype(int)
    evaluator = Evaluator(gt, gms_pred, baseline.label)
    leads     = evaluator.early_detection_lead(env, gms, baseline)
    evaluator.print_table(leads)

    # ── 4. Visualise ─────────────────────────────────────────────────────
    if not args.no_plots:
        print("[Step 4/4]  Generating figures …")
        plot_all(env, gms, baseline, out_dir=args.out)
    else:
        print("[Step 4/4]  Skipping plots (--no-plots).")

    elapsed = time.time() - start
    print(f"\n{'='*65}")
    print(f"  Pipeline complete in {elapsed:.1f} s")
    print(f"  Figures saved to  →  {os.path.abspath(args.out)}/")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
