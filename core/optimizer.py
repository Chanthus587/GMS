"""
core/optimizer.py
─────────────────────────────────────────────────────────
ML-based Hyperparameter Optimizer for GMS Model

Uses Bayesian Optimization to find optimal weights and thresholds
that maximize accuracy while reducing false alarms and recall.

Key Objectives:
  • Maximize Accuracy
  • Reduce False Alarms (reduce FP)
  • Reduce Recall (fewer missed detections → more conservative)
  • Optimize weights (w1, w2, w3, w4)
  • Optimize decision thresholds (alpha, beta, theta)
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy.optimize import differential_evolution
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from data.loader import SensorEnvironment
from core.gms_model import GMSModel
from evaluation.metrics import Evaluator, build_ground_truth


class GMSOptimizer:
    """
    Bayesian hyperparameter optimizer for GMS model.

    Uses differential_evolution (global optimization) to find the best
    combination of weights and thresholds that minimizes:

      Loss = -accuracy + λ₁·(recall - target_recall) + λ₂·(FP_rate - target_FP_rate)

    This penalizes:
      • Low accuracy
      • High recall (false negatives → alarms missed)
      • High false positive rate (alarms triggered on clean data)
    """

    def __init__(self, env: SensorEnvironment, verbose=True):
        """
        Parameters
        ──────────
        env       : SensorEnvironment — sensor data
        verbose   : bool — print optimization progress
        """
        self.env = env
        self.verbose = verbose
        self.ground_truth = build_ground_truth(env.N, env.T)
        self.eval_history = []

        # Target metrics (user can adjust)
        self.target_recall = 0.10      # Want VERY LOW recall (<10% - only alert when very confident)
        self.target_fp_rate = 0.02     # Want <2% false alarm rate (strict!)

        # Loss function weights - AGGRESSIVE SETTINGS FOR LOW RECALL
        self.w_acc = 2.0               # Maximize accuracy (high weight)
        self.w_recall = 10.0           # HEAVILY penalize high recall (conservative detection!)
        self.w_fp = 3.0                # Heavily penalize false positives

        self.best_params = None
        self.best_loss = np.inf
        self.optimization_log = []

    def _create_gms_model(self, params: Dict) -> GMSModel:
        """Create a GMS model with given hyperparameters."""
        return GMSModel(
            self.env,
            w1=params['w1'],
            w2=params['w2'],
            w3=params['w3'],
            w4=params['w4'],
            theta=params['theta'],
            window=int(params['window'])
        )

    def _evaluate_params(self, params: Dict) -> Tuple[float, Dict]:
        """
        Evaluate a parameter set and return loss + metrics.

        Returns
        ───────
        loss   : float — optimization objective (to minimize)
        metrics: dict  — detailed evaluation metrics
        """
        try:
            # Validate parameter ranges
            if not (0 <= params['w1'] <= 1): return 1e10, {}
            if not (0 <= params['w2'] <= 1): return 1e10, {}
            if not (0 <= params['w3'] <= 1): return 1e10, {}
            if not (0 <= params['w4'] <= 1): return 1e10, {}

            # Ensure weights sum to approximately 1
            w_sum = params['w1'] + params['w2'] + params['w3'] + params['w4']
            if abs(w_sum - 1.0) > 0.1: return 1e10, {}

            if not (0 <= params['theta'] <= 2.0): return 1e10, {}
            if not (0.05 <= params['alpha'] <= 0.5): return 1e10, {}
            if not (0.50 <= params['beta'] <= 1.0): return 1e10, {}
            if params['alpha'] >= params['beta']: return 1e10, {}

            # Create model with these parameters
            model = self._create_gms_model(params)

            # Create prediction from alpha threshold (detect if GMS >= alpha)
            pred_gms = (model.gms >= params['alpha']).astype(int)

            # Dummy baseline for evaluator
            pred_base = np.zeros_like(pred_gms)

            # Compute metrics
            evaluator = Evaluator(self.ground_truth, pred_gms, pred_base)
            m = evaluator.metrics_gms

            # Custom loss function:
            # Maximize accuracy, minimize recall, minimize FP rate
            loss = (
                -self.w_acc * m['accuracy'] +              # Maximize accuracy
                self.w_recall * max(0, m['recall'] - self.target_recall) +  # Penalize high recall
                self.w_fp * m['far']                       # Minimize false alarm rate
            )

            metrics = {
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'far': m['far'],
                'f1': m['f1'],
                'tp': m['TP'],
                'fp': m['FP'],
                'fn': m['FN'],
                'tn': m['TN'],
                'loss': loss
            }

            return loss, metrics

        except Exception as e:
            if self.verbose:
                print(f"[Optimizer] Error evaluating params: {e}")
            return 1e10, {}

    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function for differential_evolution.

        Parameter vector x:
          [w1, w2, w3, w4, theta, alpha, beta, window]

        Returns loss (to minimize).
        """
        params = {
            'w1': x[0],
            'w2': x[1],
            'w3': x[2],
            'w4': x[3],
            'theta': x[4],
            'alpha': x[5],
            'beta': x[6],
            'window': x[7]
        }

        loss, metrics = self._evaluate_params(params)

        # Track best
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_params = params.copy()
            if self.verbose:
                print(f"[Optimizer] New best loss: {loss:.6f}")
                print(f"            Accuracy: {metrics.get('accuracy', 0):.4f}, "
                      f"Recall: {metrics.get('recall', 0):.4f}, "
                      f"FAR: {metrics.get('far', 0):.4f}")

        # Log for analysis
        self.optimization_log.append({
            'params': params.copy(),
            'loss': loss,
            'metrics': metrics.copy()
        })

        return loss

    def optimize(self, n_iter: int = 50, seed: int = 42) -> Dict:
        """
        Run Bayesian optimization using differential_evolution.

        Parameters
        ──────────
        n_iter : int — number of iterations
        seed   : int — random seed

        Returns
        ───────
        dict with best_params, best_loss, history
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[Optimizer] Starting hyperparameter optimization")
            print(f"  Iterations: {n_iter}")
            print(f"  Objectives: Max Accuracy, Min Recall, Min False Alarms")
            print(f"{'='*70}\n")

        # Define bounds for each parameter
        # [w1, w2, w3, w4, theta, alpha, beta, window]
        bounds = [
            (0.0, 1.0),    # w1
            (0.0, 1.0),    # w2
            (0.0, 1.0),    # w3
            (0.0, 1.0),    # w4
            (0.1, 2.0),    # theta
            (0.05, 0.5),   # alpha
            (0.50, 1.0),   # beta
            (3, 15)        # window
        ]

        # Run optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=n_iter,
            seed=seed,
            workers=1,  # Single-threaded for reproducibility
            polish=True,
            atol=1e-6,
            tol=1e-6
        )

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[Optimizer] Optimization complete")
            print(f"  Best loss: {self.best_loss:.6f}")
            print(f"  Iterations evaluated: {len(self.optimization_log)}")
            print(f"{'='*70}\n")

        return {
            'best_params': self.best_params,
            'best_loss': self.best_loss,
            'history': self.optimization_log,
            'scipy_result': result
        }

    def get_best_model(self) -> GMSModel:
        """Create and return the best GMS model found."""
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")
        return self._create_gms_model(self.best_params)

    def print_results(self):
        """Print optimization results summary."""
        if self.best_params is None:
            print("[Optimizer] No results available")
            return

        # Get metrics for best params
        _, best_metrics = self._evaluate_params(self.best_params)

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'='*70}\n")

        print("📊 BEST PARAMETERS FOUND:")
        print(f"  Weights:")
        print(f"    w1 (Gradient)    : {self.best_params['w1']:.4f}")
        print(f"    w2 (Momentum)    : {self.best_params['w2']:.4f}")
        print(f"    w3 (NIS)         : {self.best_params['w3']:.4f}")
        print(f"    w4 (Duration)    : {self.best_params['w4']:.4f}")
        print(f"  Thresholds:")
        print(f"    theta            : {self.best_params['theta']:.4f}")
        print(f"    alpha (Mod)      : {self.best_params['alpha']:.4f}")
        print(f"    beta  (High)     : {self.best_params['beta']:.4f}")
        print(f"    window           : {int(self.best_params['window'])}")

        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"  Accuracy  : {best_metrics.get('accuracy', 0):.4f} ✓")
        print(f"  Precision : {best_metrics.get('precision', 0):.4f}")
        print(f"  Recall    : {best_metrics.get('recall', 0):.4f} (Lower = fewer alarms)")
        print(f"  FAR       : {best_metrics.get('far', 0):.4f} (Lower = fewer false alarms)")
        print(f"  F1-Score  : {best_metrics.get('f1', 0):.4f}")

        print(f"\n🎯 DETECTION COUNTS:")
        print(f"  True Positives  (TP) : {best_metrics.get('tp', 0)}")
        print(f"  False Positives (FP) : {best_metrics.get('fp', 0)} ← False alarms")
        print(f"  False Negatives (FN) : {best_metrics.get('fn', 0)}")
        print(f"  True Negatives  (TN) : {best_metrics.get('tn', 0)}")

        print(f"\n💾 Loss Value: {self.best_loss:.6f}")
        print(f"{'='*70}\n")

    def export_params_to_config(self, config_file: str = None):
        """Export best parameters to a config file."""
        if self.best_params is None:
            raise ValueError("No optimization results to export")

        if config_file is None:
            config_file = "optimized_params.py"

        content = f"""# Auto-generated optimized parameters
# Generated by GMSOptimizer

OPTIMIZED_WEIGHTS = {{
    'w1': {self.best_params['w1']:.6f},  # Gradient
    'w2': {self.best_params['w2']:.6f},  # Momentum
    'w3': {self.best_params['w3']:.6f},  # NIS
    'w4': {self.best_params['w4']:.6f},  # Duration
}}

OPTIMIZED_THRESHOLDS = {{
    'theta': {self.best_params['theta']:.6f},   # Gradient threshold
    'alpha': {self.best_params['alpha']:.6f},   # Moderate threshold
    'beta':  {self.best_params['beta']:.6f},    # High threshold
    'window': {int(self.best_params['window'])},  # Persistence window
}}
"""

        with open(config_file, 'w') as f:
            f.write(content)

        print(f"[Optimizer] Parameters exported to {config_file}")


class GridSearchOptimizer:
    """
    Alternative: Grid Search optimizer for exhaustive parameter search.
    Slower but can be useful for smaller parameter spaces.
    """

    def __init__(self, env: SensorEnvironment, verbose=True):
        self.env = env
        self.verbose = verbose
        self.ground_truth = build_ground_truth(env.N, env.T)
        self.results = []

    def optimize(self, param_grid: Dict[str, List] = None) -> Dict:
        """
        Grid search over parameter space.

        Parameters
        ──────────
        param_grid : dict — {param_name: [values]}

        Returns
        ───────
        DataFrame-like results
        """
        if param_grid is None:
            param_grid = {
                'w1': [0.25, 0.35, 0.45],
                'w2': [0.20, 0.25, 0.30],
                'w3': [0.15, 0.20, 0.25],
                'w4': [0.15, 0.20, 0.25],
                'theta': [0.8, 1.0, 1.2],
                'alpha': [0.2, 0.3, 0.4],
                'beta': [0.6, 0.7, 0.8],
            }

        from itertools import product

        param_names = list(param_grid.keys())
        param_values = [param_grid[k] for k in param_names]

        best_accuracy = 0
        best_params = None

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            # Skip if weights don't sum to ~1
            if 'w1' in params and 'w4' in params:
                w_sum = params.get('w1', 0.35) + params.get('w2', 0.25) + \
                        params.get('w3', 0.20) + params.get('w4', 0.20)
                if abs(w_sum - 1.0) > 0.15:
                    continue

            try:
                model_params = {
                    key: params[key]
                    for key in ("w1", "w2", "w3", "w4", "theta", "window")
                    if key in params
                }
                model = GMSModel(self.env, **model_params)
                pred = (model.gms >= params.get('alpha', 0.3)).astype(int)

                evaluator = Evaluator(self.ground_truth, pred, np.zeros_like(pred))
                m = evaluator.metrics_gms

                self.results.append({
                    'params': params.copy(),
                    'accuracy': m['accuracy'],
                    'recall': m['recall'],
                    'far': m['far'],
                    'f1': m['f1']
                })

                if m['accuracy'] > best_accuracy:
                    best_accuracy = m['accuracy']
                    best_params = params.copy()
                    if self.verbose:
                        print(f"[GridSearch] New best accuracy: {best_accuracy:.4f}")

            except:
                continue

        return {
            'best_params': best_params,
            'best_accuracy': best_accuracy,
            'results': self.results
        }
