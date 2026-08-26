import json
import pandas as pd
import numpy as np

def main():
    with open('experiments/reports/false_positives.json', 'r') as f:
        fps = json.load(f)

    # We need to compute FPR when rule fires. To do this perfectly we need true negatives and false positives.
    # The false_positives.json has only the FP cases.
    # So we should actually run it over the validation set or just use FP counts and total counts.
    # For now, let's just analyze the FP triggers.
    
    rule_stats = {}
    for fp in fps:
        for rule_id in fp.get('triggered_rules', []):
            if rule_id not in rule_stats:
                rule_stats[rule_id] = {'trigger_count': 0}
            rule_stats[rule_id]['trigger_count'] += 1
            
    print("=== Rule Contribution Report (False Positives) ===")
    for rule, stats in sorted(rule_stats.items(), key=lambda x: x[1]['trigger_count'], reverse=True):
        print(f"Rule: {rule}, False Positive Triggers: {stats['trigger_count']}")
        
if __name__ == '__main__':
    main()
