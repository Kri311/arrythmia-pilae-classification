import numpy as np
from src.config import WINDOW_SIZE

def apply_dynamic_window(segment, is_close_to_adjacent):
    """Applies a Hann window to smooth the edges if an adjacent beat is too close."""
    if is_close_to_adjacent:
        window = np.hanning(len(segment))
        return segment * window
    return segment

def extract_beats(signal_lead_1, signal_lead_2, peaks):
    """
    Extracts fixed windows around R-peaks for both leads.
    """
    segments_lead_1 = []
    segments_lead_2 = []
    valid_peaks = []
    
    for i, peak in enumerate(peaks):
        if peak >= WINDOW_SIZE and peak < (len(signal_lead_1) - WINDOW_SIZE):
            
            # Dynamic check: is the distance to the adjacent peak dangerously short?
            is_close = False
            if i > 0 and (peak - peaks[i-1]) < WINDOW_SIZE * 1.2:
                is_close = True
            if i < len(peaks) - 1 and (peaks[i+1] - peak) < WINDOW_SIZE * 1.2:
                is_close = True
                
            # Extract [peak - 99 : peak + 99] for BOTH leads
            seg_1 = signal_lead_1[peak - WINDOW_SIZE : peak + WINDOW_SIZE]
            seg_2 = signal_lead_2[peak - WINDOW_SIZE : peak + WINDOW_SIZE]
            
            seg_1 = apply_dynamic_window(seg_1, is_close)
            seg_2 = apply_dynamic_window(seg_2, is_close)
            
            segments_lead_1.append(seg_1)
            segments_lead_2.append(seg_2)
            valid_peaks.append(peak)
            
    return segments_lead_1, segments_lead_2, valid_peaks