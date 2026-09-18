import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
import logging

logger = logging.getLogger(__name__)

def add_gaussian_noise(signal, noise_factor=0.05):
    noise = np.random.normal(0, noise_factor, signal.shape)
    return signal + noise

def shift_signal(signal, shift_max=5):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(signal, shift)

def augment_data(X_train, y_train):
    """
    Manages severe data imbalance dynamically based on class counts.
    """
    class_counts = Counter(y_train)
    min_samples = min(class_counts.values())
    
    logger.info(f"Class distribution before augmentation: {dict(class_counts)}")
    
    # Dynamically determine k_neighbors (must be less than the smallest class size)
    if min_samples > 1:
        k_neighbors = min(5, min_samples - 1)
        logger.info(f"Using SMOTE with k_neighbors={k_neighbors}")
        resampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
    else:
        logger.warning("A class has only 1 sample! Falling back to RandomOverSampler.")
        resampler = RandomOverSampler(random_state=42)
        
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    logger.info(f"Class distribution after augmentation: {dict(Counter(y_resampled))}")
    
    augmented_X = []
    
    for i in range(len(X_resampled)):
        sample = X_resampled[i].copy()
        
        if np.random.rand() > 0.7:
            if np.random.rand() > 0.5:
                sample = add_gaussian_noise(sample)
            else:
                sample = shift_signal(sample)
                
        augmented_X.append(sample)
        
    return np.array(augmented_X), y_resampled