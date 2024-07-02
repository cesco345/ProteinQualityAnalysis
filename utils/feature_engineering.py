from sklearn.feature_selection import SelectKBest, f_classif

def select_features(X, y, k=5):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector

if __name__ == "__main__":
    from data.fetch_data import fetch_data
    X, y = fetch_data()
    X_selected, selector = select_features(X, y)
    print(X_selected[:5])

