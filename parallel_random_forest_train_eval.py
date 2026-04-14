import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.parallel_random_forest import ParallelRandomForest 
from src.preprocessing import load_and_preprocess
import itertools
import time
# --- Hyperparameter Tuning ---
best_val_f1 = 0
best_depth = 0
best_min_samples = 2
best_n = 0.95
best_num_of_classifiers = 5
depths = [3, 5, 10, 15]
min_samples_split_vals = [2, 5, 10, 20]
n_components_vals = [0.85, 0.9, 0.95, 0.995]
num_of_classifiers_vals = [5, 10, 15, 25]
print("Tuning Parallel Random Forest Tree Hyperparameters...")
start = time.time()
for  num_of_classifiers, depth, min_samples, n_components in itertools.product(num_of_classifiers_vals, depths, min_samples_split_vals, n_components_vals):
    # Using fixed random seed 42 and stratified split (70/10/20)
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', n_components, True)
    
    bdt = ParallelRandomForest(num_of_classifiers=num_of_classifiers,max_depth=depth, min_samples_split=min_samples)
    bdt.fit(X_train, y_train)
    
    preds = bdt.predict(X_val)
    score = f1_score(y_val, preds) 
    
    print(f"Num of Classifiers: {num_of_classifiers} | Depth: {depth} | MinSamples: {min_samples} | n_component: {n_components} | Validation F1: {score:.4f}")
    
    if score > best_val_f1:
        best_val_f1 = score
        best_depth = depth
        best_min_samples = min_samples
        best_n = n_components
        best_num_of_classifiers = best_num_of_classifiers
end = time.time()
print(f"\nBest Number of Classfiers found: {best_num_of_classifiers}, Best Depth found: {best_depth}, Best Min Samples: {best_min_samples}, Best n_components: {best_n}")

# --- Final Evaluation ---
# Train final model on training set with finalized hyperparameters 
final_model = ParallelRandomForest(num_of_classifiers=best_num_of_classifiers,max_depth=best_depth, min_samples_split=best_min_samples)
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', best_n)
final_model.fit(X_train, y_train)

# Test only on the unseen test set
y_pred = final_model.predict(X_test)

# Compute mandatory performance metrics 
acc = accuracy_score(y_test, y_pred) 
f1 = f1_score(y_test, y_pred) 
cm = confusion_matrix(y_test, y_pred) 

# --- Directory and File Management ---
# Create the specific directory for this hyperparameter combination
result_dir = f'results/parallel_random_forest/num_of_classifiers_{best_num_of_classifiers}_depth_{best_depth}_min_{best_min_samples}'
os.makedirs(result_dir, exist_ok=True)

# Save metrics to text file for the report 
output_path = os.path.join(result_dir, 'metrics.txt')
with open(output_path, 'w') as f:
    f.write("Parallel Random Forest Classification Results\n")
    f.write("====================================\n")
    f.write(f"training time : {(end-start):.4f} seconds\n")
    f.write(f"Best Hyperparameters:\n")
    f.write(f" - Max Depth: {best_depth}\n")
    f.write(f" - Min Samples Split: {best_min_samples}\n")
    f.write(f" - PCA n_components: {best_n}\n\n")
    f.write(f"Test Set Performance (20% split):\n")
    f.write(f" - Accuracy: {acc:.4f}\n")
    f.write(f" - F1-Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

print(f"\nFinal Results on Test Set (20% split)")
print(f"Accuracy: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")

# Plot and save Confusion Matrix figure 
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Parallel Random Forest Confusion Matrix (Depth {best_depth})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(result_dir, 'cm.png'))
plt.close() # Close figure to free memory