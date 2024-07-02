from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from fetch_data import fetch_data
    from preprocess_data import preprocess_data
    X, y = fetch_data()
    X_scaled = preprocess_data(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    print(X_train.shape, X_test.shape)

