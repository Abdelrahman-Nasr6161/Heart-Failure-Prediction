import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
from src.decision_tree import DecisionTree

def _train_tree(args):
    X, y, min_samples_split, max_depth, n_samples = args
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_sampled, y_sampled = X[indices], y[indices]
    clf = DecisionTree(min_samples_split=min_samples_split, max_depth=max_depth, n_features="sqrt")
    clf.fit(X_sampled, y_sampled)
    return clf

class ParallelRandomForest:
    def __init__(self, num_of_classifiers=10, min_samples_split=2, max_depth=5, n_jobs=-1):
        self.num_of_classifiers = num_of_classifiers
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.classifiers = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        task_args = [
            (X, y, self.min_samples_split, self.max_depth, n_samples)
            for _ in range(self.num_of_classifiers)
        ]
        
        with Pool(processes=self.n_jobs) as pool:
            self.classifiers = pool.map(_train_tree, task_args)

    def predict(self, X):
        all_tree_preds = np.array([clf.predict(X) for clf in self.classifiers])
        all_tree_preds = all_tree_preds.T
        
        final_preds = []
        for sample_preds in all_tree_preds:
            most_common = Counter(sample_preds).most_common(1)[0][0]
            final_preds.append(most_common)
            
        return np.array(final_preds)