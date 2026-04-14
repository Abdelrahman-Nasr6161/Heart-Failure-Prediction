from src.decision_tree import DecisionTree
import numpy as np
from collections import Counter
class RandomForest:
    def __init__(self,num_of_classifiers=10, min_samples_split=2, max_depth=5):
        self.num_of_classifiers=num_of_classifiers
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
    def fit(self,X,y):
        n_samples = X.shape[0]
        self.classifiers = []
        for i in range(self.num_of_classifiers):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sampled, y_sampled = X[indices], y[indices] 
            clf = DecisionTree(self.min_samples_split, self.max_depth, "sqrt")
            clf.fit(X_sampled, y_sampled)
            self.classifiers.append(clf)
    def predict(self, X):
        all_tree_preds = np.array([clf.predict(X) for clf in self.classifiers])
        
        all_tree_preds = all_tree_preds.T
        
        final_preds = []
        for sample_preds in all_tree_preds:
            most_common = Counter(sample_preds).most_common(1)[0][0]
            final_preds.append(most_common)
            
        return np.array(final_preds)