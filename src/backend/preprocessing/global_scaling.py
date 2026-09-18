from sklearn.preprocessing import StandardScaler

def apply_global_scaling(X_train, X_test):
    """Fits the StandardScaler on training data and transforms both datasets."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler