from imblearn.over_sampling import SMOTE
import numpy as np

def handle_imbalance(X, y):
    # Determine the number of samples in the smallest class
    class_counts = np.bincount(y)
    min_class_count = np.min(class_counts)
    
    # Set k_neighbors to be less than the number of samples in the smallest class
    k_neighbors = min(min_class_count - 1, 5)  # Default is 5, but it cannot exceed min_class_count - 1

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

if __name__ == "__main__":
    from fetch_data import fetch_data
    X, y = fetch_data()
    X_resampled, y_resampled = handle_imbalance(X, y)
    print(X_resampled.shape, y_resampled.shape)

