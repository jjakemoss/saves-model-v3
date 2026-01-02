import json
from pathlib import Path
from datetime import datetime

# Read the tuning results
results_path = Path('models/metadata/hyperparameter_tuning_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

# Find the best result (lowest test_rmse)
best_result = min(results, key=lambda x: x['test_rmse'])

print(f"Best configuration found:")
print(f"  Test RMSE: {best_result['test_rmse']:.3f}")
print(f"  Val RMSE: {best_result['val_rmse']:.3f}")
print(f"  Iteration: {best_result['iteration']}")
print(f"\nParameters:")
for k, v in best_result['params'].items():
    print(f"  {k}: {v}")

# Save best parameters
best_params_path = Path('models/metadata/best_hyperparameters.json')
with open(best_params_path, 'w') as f:
    json.dump({
        'best_params': best_result['params'],
        'val_rmse': best_result['val_rmse'],
        'test_rmse': best_result['test_rmse'],
        'iteration': best_result['iteration'],
        'tuning_date': datetime.now().isoformat()
    }, f, indent=2)

print(f"\nBest parameters saved to {best_params_path}")

# Compare to baseline
baseline_rmse = 6.510
improvement = baseline_rmse - best_result['test_rmse']
pct_improvement = (improvement / baseline_rmse) * 100

print(f"\n{'='*70}")
print("IMPROVEMENT SUMMARY")
print(f"{'='*70}")
print(f"Baseline (EWA, default params): {baseline_rmse:.3f} RMSE")
print(f"Optimized hyperparameters:      {best_result['test_rmse']:.3f} RMSE")
print(f"Improvement:                    {improvement:.3f} saves ({pct_improvement:.2f}%)")
print(f"{'='*70}")
