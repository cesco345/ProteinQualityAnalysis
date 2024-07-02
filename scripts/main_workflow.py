import sys
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetch_data import fetch_data
from data.preprocess_data import preprocess_data
from data.split_data import split_data
from models.train_model import train_model
from models.tune_hyperparameters import tune_hyperparameters
from models.evaluate_model import evaluate_model
from models.advanced_models import train_xgboost, train_stacking
from utils.handle_imbalance import handle_imbalance
from utils.feature_engineering import select_features
from utils.visualization import plot_feature_importance

# Fetch and preprocess data
X, y = fetch_data()
X_scaled = preprocess_data(X)

# Handle class imbalance
X_resampled, y_resampled = handle_imbalance(X_scaled, y)

# Feature selection
X_selected, selector = select_features(X_resampled, y_resampled)

# Split data
X_train, X_test, y_train, y_test = split_data(X_selected, y_resampled)

# Train model with hyperparameter tuning
best_model = tune_hyperparameters(X_train, y_train)

# Evaluate model
accuracy, cm = evaluate_model(best_model, X_test, y_test)
print(f'Best Model Accuracy: {accuracy * 100:.2f}%')

# Plot feature importance
plot_feature_importance(best_model, selector.get_feature_names_out())

# Train and evaluate advanced models
xgb_model = train_xgboost(X_train, y_train)
accuracy_xgb, cm_xgb = evaluate_model(xgb_model, X_test, y_test)
print(f'XGBoost Model Accuracy: {accuracy_xgb * 100:.2f}%')

stacking_model = train_stacking(X_train, y_train)
accuracy_stack, cm_stack = evaluate_model(stacking_model, X_test, y_test)
print(f'Stacking Model Accuracy: {accuracy_stack * 100:.2f}%')

# Plot confusion matrix for XGBoost
plt.figure(figsize=(10, 7))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - XGBoost')
plt.show()

# Plot confusion matrix for stacking model
plt.figure(figsize=(10, 7))
sns.heatmap(cm_stack, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Stacking')
plt.show()

