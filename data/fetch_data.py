from ucimlrepo import fetch_ucirepo
import pandas as pd

def fetch_data():
    yeast = fetch_ucirepo(id=110)
    X = yeast.data.features
    y = yeast.data.targets['localization_site'].astype('category').cat.codes
    return X, y

if __name__ == "__main__":
    X, y = fetch_data()
    print(X.head(), y.head())
