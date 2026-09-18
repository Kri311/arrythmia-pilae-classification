import numpy as np
from imblearn.over_sampling import SMOTE

def apply_augmentation(X_train, y_train):
    """Applies SMOTE to balance minority classes strictly in the training set."""
    print(f"Original training distribution: {np.unique(y_train, return_counts=True)}")
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Resampled training distribution: {np.unique(y_resampled, return_counts=True)}")
    return X_resampled, y_resampled