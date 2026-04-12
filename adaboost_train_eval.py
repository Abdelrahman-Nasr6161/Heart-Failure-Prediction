import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.adaboost import AdaBoost
from src.preprocessing import load_and_preprocess
import itertools

# --- Hyperparameter Tuning ---
best_val_f1 = 0
best_weak_learner = 10
best_depth = 1
best_n = 0.95

weak_learners_no_list = [10, 30, 50, 100]         
weak_learners_depth_list = [1, 2, 3]                   # Using stumps (1) or very shallow trees (2, 3)
n_components_val_list = [0.85, 0.9, 0.95, 0.995]

print("Tuning AdaBoost Hyperparameters...")
for no_of_weak_learners, depth_of_weak_learners, n_components in itertools.product(weak_learners_no_list, weak_learners_depth_list, n_components_val_list):
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', n_components, True)
    
    ab = AdaBoost(no_of_weak_learners=no_of_weak_learners, depth_of_weak_learners=depth_of_weak_learners)
    ab.fit(X_train, y_train)
    
    preds = ab.predict(X_val)
    score = f1_score(y_val, preds) 
    
    print(f"Estimators: {no_of_weak_learners} | Depth: {depth_of_weak_learners} | n_component: {n_components} | Validation F1: {score:.4f}")
    
    if score > best_val_f1:
        best_val_f1 = score
        best_weak_learner = no_of_weak_learners
        best_depth = depth_of_weak_learners
        best_n = n_components

print(f"\nBest Estimators: {best_weak_learner}, Best Depth: {best_depth}, Best n_components: {best_n}")

# --- Final Evaluation ---
# Train final model on training set with finalized hyperparameters 
final_model = AdaBoost(no_of_weak_learners=best_weak_learner, depth_of_weak_learners=best_depth)
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', best_n, True)
final_model.fit(X_train, y_train)

# Test only on the unseen test set
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred) 
f1 = f1_score(y_test, y_pred) 
cm = confusion_matrix(y_test, y_pred) 

result_dir = f'results/adaboost/nclf_{best_weak_learner}_depth_{best_depth}'
os.makedirs(result_dir, exist_ok=True)

output_path = os.path.join(result_dir, 'metrics.txt')
with open(output_path, 'w') as f:
    f.write("AdaBoost Classification Results\n")
    f.write("===============================\n")
    f.write(f"Best Hyperparameters:\n")
    f.write(f" - Number of Estimators: {best_weak_learner}\n")
    f.write(f" - Weak Learner Max Depth: {best_depth}\n")
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
plt.title(f'AdaBoost Confusion Matrix (Est. {best_weak_learner}, Depth {best_depth})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(result_dir, 'cm.png'))
plt.close()