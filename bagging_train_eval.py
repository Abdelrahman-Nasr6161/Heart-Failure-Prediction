import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.bagging import BaggingClassifier
from src.preprocessing import load_and_preprocess
import itertools

# --- Hyperparameter Tuning ---
best_val_f1 = 0
best_n_estimators = 10
best_depth = 10
best_n = 0.95

n_estimators_list = [10, 20, 50, 100]
depths_list = [3, 5, 10, 15]
n_components_val_list = [0.85, 0.9, 0.95, 0.995]

print("Tuning Bagging Hyperparameters...")
for n_estimators, max_depth, n_components in itertools.product(n_estimators_list, depths_list, n_components_val_list):
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', n_components, True)

    bg = BaggingClassifier(n_estimators=n_estimators, max_depth=max_depth)
    bg.fit(X_train, y_train)

    preds = bg.predict(X_val)
    score = f1_score(y_val, preds)

    print(f"Estimators: {n_estimators} | Depth: {max_depth} | n_components: {n_components} | Validation F1: {score:.4f}")

    if score > best_val_f1:
        best_val_f1 = score
        best_n_estimators = n_estimators
        best_depth = max_depth
        best_n = n_components

print(f"\nBest Estimators: {best_n_estimators}, Best Depth: {best_depth}, Best n_components: {best_n}")

# --- Final Evaluation ---
# Train final model on training set with finalized hyperparameters
final_model = BaggingClassifier(n_estimators=best_n_estimators, max_depth=best_depth)
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', best_n, True)
final_model.fit(X_train, y_train)

# Test only on the unseen test set
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

result_dir = f'results/bagging/nclf_{best_n_estimators}_depth_{best_depth}'
os.makedirs(result_dir, exist_ok=True)

output_path = os.path.join(result_dir, 'metrics.txt')
with open(output_path, 'w') as f:
    f.write("Bagging Classification Results\n")
    f.write("==============================\n")
    f.write(f"Best Hyperparameters:\n")
    f.write(f" - Number of Estimators: {best_n_estimators}\n")
    f.write(f" - Base Learner Max Depth: {best_depth}\n")
    f.write(f" - PCA n_components: {best_n}\n\n")
    f.write(f"Test Set Performance (20% split):\n")
    f.write(f" - Accuracy: {acc:.4f}\n")
    f.write(f" - F1-Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

print(f"\nFinal Results on Test Set (20% split)")
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Bagging Confusion Matrix (Est. {best_n_estimators}, Depth {best_depth})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(result_dir, 'cm.png'))
plt.close()
