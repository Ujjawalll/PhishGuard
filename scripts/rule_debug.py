import pandas as pd
from ml.rules.engine import RuleEngine
from ml.features.lexical import extract_lexical_features

def main():
    df = pd.read_csv("ml/data/splits/random/val.csv")
    engine = RuleEngine()
    
    triggers_pos = {}
    triggers_neg = {}
    
    for _, row in df.iterrows():
        url = row['url']
        label = row['label']
        features = extract_lexical_features(url)
        res = engine.evaluate(url, features)
        
        for rule in res["triggered_rules"]:
            r_id = rule["rule_id"]
            if label == 1:
                triggers_pos[r_id] = triggers_pos.get(r_id, 0) + 1
            else:
                triggers_neg[r_id] = triggers_neg.get(r_id, 0) + 1
                
    print("Rule triggers on POSITIVE (phishing) examples (total 14):")
    for r, count in sorted(triggers_pos.items(), key=lambda x: x[1], reverse=True):
        print(f"  {r}: {count}")
        
    print("\nRule triggers on NEGATIVE (legit) examples (total 999):")
    for r, count in sorted(triggers_neg.items(), key=lambda x: x[1], reverse=True):
        print(f"  {r}: {count}")

if __name__ == "__main__":
    main()
