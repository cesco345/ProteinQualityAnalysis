from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    from data.split_data import split_data
    from train_model import train_model
    X_train, X_test, y_train, y_test = split_data()
    model = train_model(X_train, y_train)
    accuracy, cm = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

