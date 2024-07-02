import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Fetch dataset
yeast = fetch_ucirepo(id=110)

# Data (as pandas dataframes)
X = yeast.data.features
y = yeast.data.targets

# Print metadata and variable information (optional)
print(yeast.metadata)
print(yeast.variables)

# Preprocess data
# Encode the localization_site (target variable) into numerical values if needed
y = y['localization_site'].astype('category').cat.codes

# Select features
features = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc']
X = X[features]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Map numeric predictions back to localization site labels
localization_site_mapping = dict(enumerate(yeast.data.targets['localization_site'].astype('category').cat.categories))
new_data = pd.DataFrame({
    'mcg': [0.58],
    'gvh': [0.54],
    'alm': [0.32],
    'mit': [0.00],
    'erl': [0.00],
    'pox': [0.00],
    'vac': [0.01],
    'nuc': [0.02]
})
prediction = model.predict(new_data)
print(f'Predicted Localization Site: {localization_site_mapping[prediction[0]]}')

# Feature Importance Visualization
feature_importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importance')
plt.show()

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

