"""
backend/optimizer.py
─────────────────────────────────────────────────────────
ML-based Hyperparameter Optimizer for GMS Model

Uses Bayesian Optimization to find optimal weights and thresholds
that maximize accuracy while reducing false alarms and recall.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.optimize import differential_evolution
from utils.metrics import build_ground_truth, Evaluator


class GMSOptimizer:
    """
    Bayesian hyperparameter optimizer for GMS model.
    """

    def __init__(self, env, verbose=True):
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
        self.target_recall = 0.10
        self.target_fp_rate = 0.02

        # Loss function weights - AGGRESSIVE SETTINGS FOR LOW RECALL
        self.w_acc = 2.0
        self.w_recall = 10.0
        self.w_fp = 3.0

        self.best_params = None
        self.best_loss = np.inf
        self.optimization_log = []

    def _create_gms_model(self, params: Dict):
        """Create a GMS model with given hyperparameters."""
        from backend.gms_engine import GMSEngine

        # This is a simplified version - for actual use,
        # you'd need to refactor GMSEngine to accept parameters
        # For now, return a mock/simple version
        return None

    def _evaluate_params(self, params: Dict) -> Tuple[float, Dict]:
        """Evaluate a parameter set and return loss + metrics."""
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

            # Dummy metrics for now
            metrics = {
                'accuracy': 0.9,
                'precision': 0.85,
                'recall': 0.08,
                'far': 0.02,
                'f1': 0.16,
                'tp': 10,
                'fp': 2,
                'fn': 2,
                'tn': 100,
                'loss': 0.0
            }

            loss = -self.w_acc * metrics['accuracy']

            return loss, metrics

        except Exception as e:
            if self.verbose:
                print(f"[Optimizer] Error evaluating params: {e}")
            return 1e10, {}

    def objective_function(self, x: np.ndarray) -> float:
        """Objective function for differential_evolution."""
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

        self.optimization_log.append({'params': params.copy(), 'loss': loss, 'metrics': metrics.copy()})

        return loss

    def optimize(self, n_iter: int = 50, seed: int = 42) -> Dict:
        """Run Bayesian optimization using differential_evolution."""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[Optimizer] Starting hyperparameter optimization")
            print(f"  Iterations: {n_iter}")
            print(f"{'='*70}\n")

        # Define bounds for each parameter
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
            workers=1,
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

    def get_best_model(self):
        """Return the best parameters found."""
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")
        return self.best_params
