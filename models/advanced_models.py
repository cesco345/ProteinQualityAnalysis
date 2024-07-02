import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_stacking(X_train, y_train):
    estimators = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('gbc', GradientBoostingClassifier(random_state=42))
    ]
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression()
    )
    stacking_model.fit(X_train, y_train)
    return stacking_model

if __name__ == "__main__":
    from data.fetch_data import fetch_data
    from data.preprocess_data import preprocess_data
    from data.split_data import split_data
    from utils.handle_imbalance import handle_imbalance
    from utils.feature_engineering import select_features

    # Fetch and preprocess data
    X, y = fetch_data()
    X_scaled = preprocess_data(X)

    # Handle class imbalance
    X_resampled, y_resampled = handle_imbalance(X_scaled, y)

    # Feature selection
    X_selected, selector = select_features(X_resampled, y_resampled)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X_selected, y_resampled)

    # Train XGBoost model
    xgb_model = train_xgboost(X_train, y_train)
    print(f'Trained XGBoost model: {xgb_model}')

    # Train stacking model
    stacking_model = train_stacking(X_train, y_train)
    print(f'Trained Stacking model: {stacking_model}')

