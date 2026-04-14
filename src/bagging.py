import numpy as np
from collections import Counter
from src.decision_tree import DecisionTree

class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.estimators = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.estimators = []

        for _ in range(self.n_estimators):
            # Draw a bootstrap sample (random sample with replacement)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(tree)

    def predict(self, X):
        # Collect predictions from all trees: shape (n_estimators, n_samples)
        all_preds = np.array([tree.predict(X) for tree in self.estimators])

        # Majority vote across estimators for each sample
        n_samples = X.shape[0]
        final_preds = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            votes = Counter(all_preds[:, i])
            final_preds[i] = votes.most_common(1)[0][0]

        return final_preds
