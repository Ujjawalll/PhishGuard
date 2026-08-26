import inspect
from ml.features.lexical import extract_lexical_features

def test_feature_extraction_no_label_argument():
    # Ensure that the feature extraction interface does not accept 'label', 'y', etc.
    sig = inspect.signature(extract_lexical_features)
    for param_name in sig.parameters:
        assert param_name.lower() not in ['label', 'y', 'target', 'ground_truth', 'phishing'], f"Label leakage guard failed: found parameter {param_name}"
