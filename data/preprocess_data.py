from sklearn.preprocessing import StandardScaler

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

if __name__ == "__main__":
    from fetch_data import fetch_data
    X, y = fetch_data()
    X_scaled = preprocess_data(X)
    print(X_scaled[:5])

