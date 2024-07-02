import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names):
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=feature_names)
    plt.title('Feature Importance')
    plt.show()

if __name__ == "__main__":
    from data.fetch_data import fetch_data
    from models.train_model import train_model
    X, y = fetch_data()
    model = train_model(X, y)
    plot_feature_importance(model, X.columns)

