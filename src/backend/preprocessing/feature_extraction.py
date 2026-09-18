import numpy as np
import config

def extract_beats(filtered_signals, r_peaks, beat_types, sig_names):
    """
    Extracts fixed window segments around R-peaks for two leads.
    """
    features = []
    labels = []
    
    # Handle inconsistencies in Lead Names
    lead1_idx, lead2_idx = 0, 1
    for idx, name in enumerate(sig_names):
        if 'MLII' in name or 'II' in name:
            lead1_idx = idx
        elif 'V1' in name or 'V2' in name or 'V5' in name or 'V4' in name:
            lead2_idx = idx
            
    num_samples = filtered_signals.shape[0]
    
    # Iterate skipping the first and last beat to safely check adjacent RR intervals
    for i in range(1, len(r_peaks) - 1):
        r_peak = r_peaks[i]
        beat_type = beat_types[i]
        
        if beat_type not in config.AAMI_MAPPING:
            continue
            
        # Dynamic check for high heart rates:
        # If the beat is too close to the previous/next beat, it overlaps morphologically.
        prev_r = r_peaks[i-1]
        next_r = r_peaks[i+1]
        
        if (r_peak - prev_r < config.WINDOW_LEFT) or (next_r - r_peak < config.WINDOW_RIGHT):
            # Skip this beat as it overlaps heavily and corrupts the feature vector
            continue
            
        # Ensure window is within signal bounds
        if r_peak - config.WINDOW_LEFT < 0 or r_peak + config.WINDOW_RIGHT >= num_samples:
            continue
            
        # Extract peak-99 to peak+99
        lead1_segment = filtered_signals[r_peak - config.WINDOW_LEFT : r_peak + config.WINDOW_RIGHT, lead1_idx]
        lead2_segment = filtered_signals[r_peak - config.WINDOW_LEFT : r_peak + config.WINDOW_RIGHT, lead2_idx]
        
        combined_segment = np.concatenate((lead1_segment, lead2_segment))
        features.append(combined_segment)
        labels.append(config.AAMI_MAPPING[beat_type])
        
    return features, labels