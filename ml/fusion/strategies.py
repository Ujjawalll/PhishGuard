import numpy as np
from sklearn.linear_model import LogisticRegression

class WeightedSumFusion:
    def __init__(self, alpha: float = 0.5, threshold: float = 0.5):
        self.alpha = alpha
        self.threshold = threshold

    def predict_proba(self, ml_prob: np.ndarray, rule_score: np.ndarray, known_malicious: np.ndarray = None) -> np.ndarray:
        prob = self.alpha * ml_prob + (1 - self.alpha) * rule_score
        if known_malicious is not None:
            prob = np.where(known_malicious, 1.0, prob)
        return prob

    def predict(self, ml_prob: np.ndarray, rule_score: np.ndarray, known_malicious: np.ndarray = None) -> np.ndarray:
        return (self.predict_proba(ml_prob, rule_score, known_malicious) >= self.threshold).astype(int)

class MetaClassifierFusion:
    def __init__(self):
        self.meta_model = LogisticRegression(random_state=42)
        self.is_fitted = False

    def _build_features(self, ml_prob: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        return np.column_stack([ml_prob, rule_score])

    def fit(self, ml_prob: np.ndarray, rule_score: np.ndarray, y_true: np.ndarray):
        X = self._build_features(ml_prob, rule_score)
        self.meta_model.fit(X, y_true)
        self.is_fitted = True
        return self

    def predict_proba(self, ml_prob: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("MetaClassifierFusion must be fitted before predict_proba")
        X = self._build_features(ml_prob, rule_score)
        return self.meta_model.predict_proba(X)[:, 1]

    def predict(self, ml_prob: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("MetaClassifierFusion must be fitted before predict")
        X = self._build_features(ml_prob, rule_score)
        return self.meta_model.predict(X)

class OrLogicFusion:
    def __init__(self, ml_threshold: float = 0.5, rule_threshold: float = 0.5):
        self.ml_threshold = ml_threshold
        self.rule_threshold = rule_threshold

    def predict(self, ml_prob: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        ml_flag = ml_prob >= self.ml_threshold
        rule_flag = rule_score >= self.rule_threshold
        return (ml_flag | rule_flag).astype(int)

    def predict_proba(self, ml_prob: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        # Faux probabilities based on the maximum of the two risk signals
        return np.maximum(ml_prob, rule_score)

class HierarchicalFusion:
    def __init__(self, rule_safe_max: float = 0.1, rule_high_min: float = 0.5, ml_threshold: float = 0.5):
        self.rule_safe_max = rule_safe_max
        self.rule_high_min = rule_high_min
        self.ml_threshold = ml_threshold

    def predict(self, ml_prob: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        preds = np.zeros_like(ml_prob, dtype=int)
        for i in range(len(ml_prob)):
            if rule_score[i] >= self.rule_high_min:
                preds[i] = 1  # Definite phishing by rules
            elif rule_score[i] <= self.rule_safe_max:
                preds[i] = 0  # Definite safe by rules
            else:
                preds[i] = 1 if ml_prob[i] >= self.ml_threshold else 0
        return preds

    def predict_proba(self, ml_prob: np.ndarray, rule_score: np.ndarray) -> np.ndarray:
        probas = np.zeros_like(ml_prob)
        for i in range(len(ml_prob)):
            if rule_score[i] >= self.rule_high_min:
                probas[i] = np.clip(0.8 + rule_score[i]*0.2, 0.8, 1.0)
            elif rule_score[i] <= self.rule_safe_max:
                probas[i] = np.clip(rule_score[i], 0.0, 0.2)
            else:
                probas[i] = ml_prob[i]
        return probas
