from sklearn.preprocessing import StandardScaler

def apply_global_scaling(X_train, X_test):
    """
    Applies Global Standard Scaling across all 396 dimensions.
    Fitted ONLY on the training data to prevent data leakage.
    """
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled