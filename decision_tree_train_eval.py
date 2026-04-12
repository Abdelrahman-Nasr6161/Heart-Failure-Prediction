import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.decision_tree import DecisionTree # Your scratch implementation
from src.preprocessing import load_and_preprocess
import itertools

# Load preprocessed data

# --- Hyperparameter Tuning ---
best_val_f1 = 0
best_depth = 0
depths = [3, 5, 10, 15, 20]
n_components_vals = [0.85,0.9,0.95,0.995]
print("Tuning Decision Tree Hyperparameters...")
for depth,n_components in itertools.product(depths, n_components_vals):
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', n_components, True)
    dt = DecisionTree(max_depth=depth, min_samples_split=2)
    dt.fit(X_train, y_train)
    
    preds = dt.predict(X_val)
    score = f1_score(y_val, preds)
    
    print(f"Depth: {depth} | n_component: {n_components} | Validation F1: {score:.4f}")
    
    if score > best_val_f1:
        best_val_f1 = score
        best_depth = depth
        best_n = n_components

print(f"\nBest Depth found: {best_depth}, Best n_components: {n_components}")

# --- Final Evaluation ---
# Train final model with best hyperparameter on training set 
final_model = DecisionTree(max_depth=best_depth)
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', n_components)
final_model.fit(X_train, y_train)

# Test on the unseen test set
y_pred = final_model.predict(X_test)

# Compute Metrics 
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"\nFinal Results on Test Set:")
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plot Confusion Matrix 
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Decision Tree Confusion Matrix (Depth {best_depth})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(f'results/decision_tree/depth_{best_depth}.png')