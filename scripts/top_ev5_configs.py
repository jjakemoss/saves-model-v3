import re

log_file = r"s:\Documents\GitHub\saves-model-v3\data\tune_comprehensive_cleaned.log"

print("Finding top EV=5% configs by combined performance...\n")

configs = []
current_config = None

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(r'\[(\d+)/6144\]\s+Testing:\s+weights=(\w+),\s+depth=(\d+),\s+child=(\d+),\s+gamma=([\d.]+),\s+lr=([\d.]+),\s+alpha=(\d+),\s+lambda=(\d+)', line)
        if match:
            if current_config is not None and current_config.get('ev_results'):
                configs.append(current_config)

            current_config = {
                'config_num': int(match.group(1)),
                'weights': match.group(2) == 'True',
                'max_depth': int(match.group(3)),
                'min_child_weight': int(match.group(4)),
                'gamma': float(match.group(5)),
                'learning_rate': float(match.group(6)),
                'reg_alpha': int(match.group(7)),
                'reg_lambda': int(match.group(8)),
                'ev_results': []
            }
            continue

        if current_config is not None:
            ev_match = re.search(r'EV=([\d]+)%:\s+Val=([-+\d.]+)%\s+\((\d+)\),\s+Test=([-+\d.]+)%\s+\((\d+)\)', line)
            if ev_match:
                current_config['ev_results'].append({
                    'ev_threshold': int(ev_match.group(1)),
                    'val_roi': float(ev_match.group(2)),
                    'val_bets': int(ev_match.group(3)),
                    'test_roi': float(ev_match.group(4)),
                    'test_bets': int(ev_match.group(5))
                })

if current_config is not None and current_config.get('ev_results'):
    configs.append(current_config)

# Filter for EV=5% and calculate combined metrics
ev5_results = []
for config in configs:
    for ev_result in config['ev_results']:
        if ev_result['ev_threshold'] == 2:
            total_bets = ev_result['val_bets'] + ev_result['test_bets']
            combined_roi = (ev_result['val_roi'] * ev_result['val_bets'] +
                           ev_result['test_roi'] * ev_result['test_bets']) / total_bets

            ev5_results.append({
                'config': config,
                'ev_data': ev_result,
                'combined_roi': combined_roi
            })

print(f"Found {len(ev5_results)} configs with EV=5% results\n")

# Sort by combined ROI and get top 10
print("=" * 80)
print("TOP 10 CONFIGS FOR EV=5% BY COMBINED ROI")
print("=" * 80)
best_combined = sorted(ev5_results, key=lambda x: x['combined_roi'], reverse=True)[:10]

for i, result in enumerate(best_combined, 1):
    config = result['config']
    ev = result['ev_data']

    print(f"\n{i}. Config #{config['config_num']}")
    print(f"   Combined ROI: {result['combined_roi']:+.2f}% (weighted by bets)")
    print(f"   Val: {ev['val_roi']:+.2f}% ({ev['val_bets']} bets) | Test: {ev['test_roi']:+.2f}% ({ev['test_bets']} bets)")
    print(f"   Total bets: {ev['val_bets'] + ev['test_bets']}")
    print(f"   Hyperparameters:")
    print(f"     use_sample_weights: {config['weights']}")
    print(f"     max_depth: {config['max_depth']}")
    print(f"     min_child_weight: {config['min_child_weight']}")
    print(f"     gamma: {config['gamma']}")
    print(f"     learning_rate: {config['learning_rate']}")
    print(f"     reg_alpha: {config['reg_alpha']}")
    print(f"     reg_lambda: {config['reg_lambda']}")
    print(f"   Python dict format:")
    print(f"     {{'max_depth': {config['max_depth']}, 'min_child_weight': {config['min_child_weight']}, 'gamma': {config['gamma']}, 'learning_rate': {config['learning_rate']}, 'reg_alpha': {config['reg_alpha']}, 'reg_lambda': {config['reg_lambda']}, 'use_sample_weights': {config['weights']}}}")

print("\n" + "=" * 80)
