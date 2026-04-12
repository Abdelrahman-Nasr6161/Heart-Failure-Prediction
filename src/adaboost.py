import numpy as np
from src.decision_tree import DecisionTree 

class AdaBoost:
    def __init__(self, no_of_weak_learners=50, depth_of_weak_learners=1):
        self.no_of_weak_learners = no_of_weak_learners
        self.depth_of_weak_learners = depth_of_weak_learners
        self.weak_learners = []
        self.alphas = []

    def fit(self, X, y):
        no_of_samples = X.shape[0]

        y_ = np.where(y == 0, -1, 1)    # 'y' as is (0 and 1) for the DecisionTree, 'y_' for AdaBoost math

        # Initialize weights to 1/N
        w = np.full(no_of_samples, (1 / no_of_samples))

        self.weak_learners = []
        self.alphas = []

        for _ in range(self.no_of_weak_learners):
            weak_learner = DecisionTree(max_depth=self.depth_of_weak_learners, min_samples_split=2)

            # Draw a bootstrap sample from the training data according to the weights w
            indices = np.random.choice(no_of_samples, no_of_samples, p=w, replace=True)
            X_resampled = X[indices]
            
            # Fit weak learner on resampled data
            y_resampled = y[indices]
            weak_learner.fit(X_resampled, y_resampled)

            # Predict on the original training data to evaluate errors
            predictions = weak_learner.predict(X)
            preds_ = np.where(predictions == 0, -1, 1)

            # Calculate weighted error using the {-1, 1} mapped predictions
            misclassified = np.where(preds_ != y_, 1, 0)
            error = np.sum(w * misclassified)

            # Early stop / skip if the weak learner is perfect or worse than random guessing (i.e., error >= 0.5)
            if error == 0:
                alpha = 10.0
            elif error >= 0.5:
                continue
            else:
                # Calculate the weight of this weak learner (alpha)
                alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))

            # Update sample weights based on alpha and predictions
            w = w * np.exp(-alpha * y_ * preds_)
            
            # Normalize the sample weights so they sum to 1
            w /= np.sum(w)

            # Save the classifier and its weight
            self.weak_learners.append(weak_learner)
            self.alphas.append(alpha)

    def predict(self, X):
        # Initialize an array to hold the weighted sum
        y_pred = np.zeros(X.shape[0])
        
        # Calculate the weighted vote
        for alpha, weak_learner in zip(self.alphas, self.weak_learners):
            weak_learner_pred = weak_learner.predict(X)
            weak_learner_pred_ = np.where(weak_learner_pred == 0, -1, 1)
            y_pred += alpha * weak_learner_pred_

        return np.where(y_pred <= 0, 0, 1)