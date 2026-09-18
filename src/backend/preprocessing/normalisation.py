import numpy as np

def per_beat_zscore(segment):
    """Applies Z-score normalization independently to a 1D signal segment."""
    mean = np.mean(segment)
    std = np.std(segment)
    if std == 0:
        std = 1e-8
    return (segment - mean) / std

def normalize_and_concatenate(segments_lead_1, segments_lead_2):
    """Normalizes each lead independently, then concatenates them."""
    X_data = []
    for seg_1, seg_2 in zip(segments_lead_1, segments_lead_2):
        norm_1 = per_beat_zscore(seg_1)
        norm_2 = per_beat_zscore(seg_2)
        
        # Concatenate 198 from lead 1 and 198 from lead 2 to make 396 dimensions
        combined = np.concatenate((norm_1, norm_2))
        X_data.append(combined)
        
    return np.array(X_data)