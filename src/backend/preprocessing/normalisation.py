import numpy as np
import config

def per_beat_zscore_normalization(features):
    """
    Applies independent Z-score normalization for Lead 1 and Lead 2 segments
    within each 396-dimensional combined feature vector to eliminate external noise variations.
    """
    normalized_features = []
    half = config.TOTAL_SAMPLES  # 198
    
    for feature in features:
        lead1 = feature[:half].copy()
        lead2 = feature[half:].copy()
        
        # Normalize Lead 1 independently
        m1, s1 = np.mean(lead1), np.std(lead1)
        if s1 > 0:
            lead1 = (lead1 - m1) / s1
            
        # Normalize Lead 2 independently
        m2, s2 = np.mean(lead2), np.std(lead2)
        if s2 > 0:
            lead2 = (lead2 - m2) / s2
            
        # Re-concatenate the normalized leads
        normalized_features.append(np.concatenate((lead1, lead2)))
        
    return np.array(normalized_features)