from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    from data.split_data import split_data
    X_train, X_test, y_train, y_test = split_data()
    model = train_model(X_train, y_train)
    print(model)

