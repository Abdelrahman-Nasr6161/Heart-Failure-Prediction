import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.preprocessing import load_and_preprocess

# --- Configuration ---
BEST_N_COMPONENTS = 0.95 # From your previous tuning
RESULT_DIR = 'results/xgboost'
os.makedirs(RESULT_DIR, exist_ok=True)

# Load data using your established preprocessing pipeline
X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess('data/heart.csv', BEST_N_COMPONENTS)

# Merge train and val for GridSearchCV (which uses internal cross-validation)
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))

# --- Hyperparameter Tuning ---
print("Tuning XGBoost Hyperparameters...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Using GridSearchCV to find the optimal combination
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train_full, y_train_full)

best_params = grid_search.best_params_
print(f"\nBest Parameters Found: {best_params}")

# --- Final Evaluation ---
final_model = grid_search.best_estimator_
y_pred = final_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# --- Save Results ---
with open(os.path.join(RESULT_DIR, 'metrics.txt'), 'w') as f:
    f.write("XGBoost Classification Results\n")
    f.write("==============================\n")
    f.write(f"Best Hyperparameters: {best_params}\n\n")
    f.write(f"Test Set Performance:\n")
    f.write(f" - Accuracy: {acc:.4f}\n")
    f.write(f" - F1-Score: {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title('XGBoost Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(RESULT_DIR, 'cm.png'))

print(f"\nFinal Results - Accuracy: {acc:.4f}, F1: {f1:.4f}")