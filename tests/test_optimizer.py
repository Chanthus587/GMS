"""
tests/test_optimizer.py
────────────────────────────────────────────────────────
Unit tests for the GMS optimizer module.

Run: pytest tests/test_optimizer.py -v
"""

import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.optimizer import GMSOptimizer, GridSearchOptimizer
from data.loader import SensorEnvironment
from evaluation.metrics import build_ground_truth


class TestGMSOptimizer:
    """Test suite for GMSOptimizer."""

    @pytest.fixture(scope="class")
    def env(self):
        """Create test environment."""
        return SensorEnvironment.simulated(N=20, T=60, seed=42)

    def test_optimizer_initialization(self, env):
        """Test optimizer can be initialized."""
        optimizer = GMSOptimizer(env, verbose=False)
        assert optimizer.env is not None
        assert optimizer.ground_truth.shape == (20, 60)
        assert optimizer.best_loss == np.inf

    def test_evaluate_params_valid(self, env):
        """Test parameter evaluation with valid parameters."""
        optimizer = GMSOptimizer(env, verbose=False)

        params = {
            'w1': 0.35,
            'w2': 0.25,
            'w3': 0.20,
            'w4': 0.20,
            'theta': 1.2,
            'alpha': 0.3,
            'beta': 0.7,
            'window': 8
        }

        loss, metrics = optimizer._evaluate_params(params)

        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert 'accuracy' in metrics
        assert 'recall' in metrics
        assert 'far' in metrics

    def test_evaluate_params_invalid(self, env):
        """Test parameter evaluation rejects invalid parameters."""
        optimizer = GMSOptimizer(env, verbose=False)

        # Invalid: weights > 1
        params = {
            'w1': 1.5,
            'w2': 0.25,
            'w3': 0.20,
            'w4': 0.20,
            'theta': 1.2,
            'alpha': 0.3,
            'beta': 0.7,
            'window': 8
        }

        loss, metrics = optimizer._evaluate_params(params)
        assert loss == 1e10  # Should return high penalty

    def test_parameter_bounds(self, env):
        """Test that objective function respects bounds."""
        optimizer = GMSOptimizer(env, verbose=False)

        # Test a point in valid range
        x = np.array([0.35, 0.25, 0.20, 0.20, 1.2, 0.3, 0.7, 8])
        loss = optimizer.objective_function(x)

        assert isinstance(loss, float)
        assert not np.isnan(loss)

    def test_optimize_short_run(self, env):
        """Test optimization runs and finds parameters."""
        optimizer = GMSOptimizer(env, verbose=False)

        result = optimizer.optimize(n_iter=5, seed=42)

        assert 'best_params' in result
        assert 'best_loss' in result
        assert 'history' in result
        assert optimizer.best_params is not None
        assert len(optimizer.optimization_log) > 0

    def test_get_best_model(self, env):
        """Test retrieving the best model."""
        optimizer = GMSOptimizer(env, verbose=False)
        optimizer.optimize(n_iter=5, seed=42)

        model = optimizer.get_best_model()
        assert model is not None
        assert hasattr(model, 'gms')
        assert model.gms.shape == (20, 60)

    def test_get_best_model_before_optimize(self, env):
        """Test that get_best_model fails before optimization."""
        optimizer = GMSOptimizer(env, verbose=False)

        with pytest.raises(ValueError):
            optimizer.get_best_model()

    def test_export_params(self, env, tmp_path):
        """Test exporting parameters to file."""
        optimizer = GMSOptimizer(env, verbose=False)
        optimizer.optimize(n_iter=5, seed=42)

        export_file = tmp_path / "test_params.py"
        optimizer.export_params_to_config(str(export_file))

        assert export_file.exists()
        content = export_file.read_text()
        assert 'OPTIMIZED_WEIGHTS' in content
        assert 'OPTIMIZED_THRESHOLDS' in content


class TestGridSearchOptimizer:
    """Test suite for GridSearchOptimizer."""

    @pytest.fixture(scope="class")
    def env(self):
        """Create test environment."""
        return SensorEnvironment.simulated(N=15, T=50, seed=42)

    def test_gridsearch_initialization(self, env):
        """Test grid search optimizer can be initialized."""
        optimizer = GridSearchOptimizer(env, verbose=False)
        assert optimizer.env is not None

    def test_gridsearch_optimize(self, env):
        """Test grid search optimization."""
        optimizer = GridSearchOptimizer(env, verbose=False)

        param_grid = {
            'w1': [0.30, 0.35],
            'w2': [0.25, 0.30],
            'w3': [0.20],
            'w4': [0.15, 0.20],
            'theta': [1.0, 1.2],
            'alpha': [0.25, 0.30],
            'beta': [0.65, 0.70],
        }

        result = optimizer.optimize(param_grid=param_grid)

        assert 'best_params' in result
        assert 'best_accuracy' in result
        assert len(result['results']) > 0


class TestOptimizationMetrics:
    """Test that optimization improves metrics."""

    def test_optimization_improves_accuracy(self):
        """Test that optimization finds parameters with reasonable accuracy."""
        env = SensorEnvironment.simulated(N=15, T=40, seed=42)
        optimizer = GMSOptimizer(env, verbose=False)

        # Run short optimization
        optimizer.optimize(n_iter=10, seed=42)

        # Check metrics
        _, metrics = optimizer._evaluate_params(optimizer.best_params)

        assert metrics['accuracy'] > 0.5  # Should get better than 50% accuracy
        assert metrics['far'] < 0.3  # False alarm rate should be reasonable

    def test_recall_vs_precision_tradeoff(self):
        """Test that optimizer balances recall and precision."""
        env = SensorEnvironment.simulated(N=15, T=40, seed=42)
        optimizer = GMSOptimizer(env, verbose=False)
        optimizer.optimize(n_iter=10, seed=42)

        _, metrics = optimizer._evaluate_params(optimizer.best_params)

        # In optimized model, recall should be moderate (conservative)
        assert metrics['recall'] < 0.9  # Not detecting everything
        # Precision should be reasonable
        assert metrics['precision'] > 0.3


# Quick integration test
def test_full_pipeline():
    """Test complete optimization pipeline."""
    print("\n" + "="*70)
    print("INTEGRATION TEST: Full Optimization Pipeline")
    print("="*70)

    # Load data
    env = SensorEnvironment.simulated(N=20, T=50, seed=42)
    print(f"✓ Created environment: {env.N} nodes, {env.T} timesteps")

    # Create optimizer
    optimizer = GMSOptimizer(env, verbose=False)
    print("✓ Created optimizer")

    # Run optimization
    print("✓ Running optimization (10 iterations)...")
    result = optimizer.optimize(n_iter=10, seed=42)
    print(f"✓ Optimization complete: Best loss = {optimizer.best_loss:.6f}")

    # Get best model
    best_model = optimizer.get_best_model()
    print(f"✓ Created best model: GMS shape = {best_model.gms.shape}")

    # Evaluate best model
    _, metrics = optimizer._evaluate_params(optimizer.best_params)
    print(f"\n✓ Best Model Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  FAR:       {metrics['far']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    # Print parameters
    print(f"\n✓ Best Parameters:")
    for k, v in optimizer.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "="*70)
    print("✅ INTEGRATION TEST PASSED")
    print("="*70 + "\n")


if __name__ == '__main__':
    # Run integration test
    test_full_pipeline()
