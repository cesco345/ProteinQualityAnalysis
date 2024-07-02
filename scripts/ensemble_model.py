import sys
import os

# Add the parent directory to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data.fetch_data import fetch_data
from data.preprocess_data import preprocess_data
from data.split_data import split_data
from utils.handle_imbalance import handle_imbalance
from utils.feature_engineering import select_features
from utils.visualization import plot_feature_importance
from models.advanced_models import train_xgboost, train_stacking

# Train and evaluate advanced models as needed
stacking_model = train_stacking(X_train, y_train)
accuracy_stack, cm_stack = evaluate_model(stacking_model, X_test, y_test)
print(f'Stacking Model Accuracy: {accuracy_stack * 100:.2f}%')


def train_ensemble_model(X_train, y_train):
    # Define individual models
    rf = RandomForestClassifier(random_state=42)
    svc = SVC(probability=True, random_state=42)
    gbc = GradientBoostingClassifier(random_state=42)

    # Combine models into a voting classifier
    ensemble_model = VotingClassifier(estimators=[
        ('rf', rf),
        ('svc', svc),
        ('gbc', gbc)
    ], voting='soft')

    # Train the ensemble model
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    return accuracy, cm

if __name__ == "__main__":
    # Fetch and preprocess data
    X, y = fetch_data()
    X_scaled = preprocess_data(X)

    # Handle class imbalance
    X_resampled, y_resampled = handle_imbalance(X_scaled, y)

    # Feature selection
    X_selected, selector = select_features(X_resampled, y_resampled)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_selected, y_resampled)

    # Train ensemble model
    ensemble_model = train_ensemble_model(X_train, y_train)

    # Evaluate model
    accuracy, cm = evaluate_model(ensemble_model, X_test, y_test)
    print(f'Ensemble Model Accuracy: {accuracy * 100:.2f}%')

    # Plot feature importance (only applicable for models that support it, like RandomForest)
    if hasattr(ensemble_model.estimators_[0], 'feature_importances_'):
        plot_feature_importance(ensemble_model.estimators_[0], selector.get_feature_names_out())

