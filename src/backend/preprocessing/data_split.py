"""
Data splitting module for inter-patient ECG records.

Loads ECG recordings, processes individual beats (filtering, segmentation,
normalization), and partitions data into train, validation, and test sets
following the inter-patient De Chazal protocol.
"""
import logging
from collections import Counter
import numpy as np
import wfdb

import config
from feature_extraction import extract_beats
from noise_removal import apply_bandpass_filter
from normalisation import per_beat_zscore_normalization

logger = logging.getLogger(__name__)


def load_and_split_data():
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    all_records = config.DS1_TRAIN + config.DS1_VAL + config.DS2_TEST
    
    for record_id in all_records:
        record_path = f"{config.DATA_DIR}/{record_id}"
        
        try:
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')
        except Exception:
            continue
            
        if record.p_signal.shape[1] < 2:
            continue
            
        logger.info(f"--- Processing Patient Record {record_id} ---")
        
        # 1. Raw Data Input
        raw_signal = record.p_signal
        logger.info(f"Phase 1 (Input): Loaded raw continuous ECG signal. Shape: {raw_signal.shape}. "
                    f"Represents ~30 mins of 2-lead recordings at 360Hz.")
        
        # 2. Noise Removal
        filtered_signals = apply_bandpass_filter(raw_signal)
        logger.info("Phase 2 (Filtering): Applied 0.5-45Hz Butterworth filter. "
                    "Effect: Baseline wander (patient breathing drift) and electrical noise eliminated. "
                    "The signal's isoelectric baseline is now completely flat.")
        
        # 3. Feature Extraction
        r_peaks = annotation.sample
        beat_types = annotation.symbol
        sig_names = record.sig_name
        
        raw_features, labels = extract_beats(filtered_signals, r_peaks, beat_types, sig_names)
        
        if not raw_features:
            logger.warning(f"Patient {record_id}: No valid beats extracted (likely extreme heart rate overlap).")
            continue
            
        logger.info(f"Phase 3 (Extraction): Segmented {len(raw_features)} valid beats using R-peaks. "
                    f"Effect: Morphed continuous signal into discrete 198-sample windows. "
                    f"Combined Lead 1 and Lead 2 into a {len(raw_features[0])}-dimensional morphological vector.")
        
        # 4. Normalization
        norm_features = per_beat_zscore_normalization(raw_features)
        logger.info("Phase 4 (Normalization): Applied independent Z-score normalization per lead. "
                    "Effect: Scaled varying peak amplitudes down to a standard range. "
                    "The model will now learn the physical *shape* of the heartbeat rather than its raw voltage size.")
        
        # 5. Patient-Wise Split
        if record_id in config.DS1_TRAIN:
            X_train.extend(norm_features)
            y_train.extend(labels)
            logger.info("Phase 5 (Routing): Patient assigned to TRAIN set (DS1).")
        elif record_id in config.DS1_VAL:
            X_val.extend(norm_features)
            y_val.extend(labels)
            logger.info("Phase 5 (Routing): Patient assigned to VALIDATION set (DS1-VAL).")
        else:
            X_test.extend(norm_features)
            y_test.extend(labels)
            logger.info("Phase 5 (Routing): Patient assigned to TEST set (DS2).")

    # Log raw class distribution for each split
    logger.info(f"Train class distribution (raw): {dict(sorted(Counter(y_train).items()))}")
    logger.info(f"Val class distribution (raw): {dict(sorted(Counter(y_val).items()))}")
    logger.info(f"Test class distribution (raw): {dict(sorted(Counter(y_test).items()))}")
            
    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)