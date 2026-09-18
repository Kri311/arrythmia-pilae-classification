import wfdb
import numpy as np
from tqdm import tqdm
from src.config import TARGET_LEADS, SYMBOL_MAP, CLASSES, TRAIN_PATIENTS
from .noise_removal import apply_bandpass_filter
from .feature_extraction import extract_beats
from .normalisation import normalize_and_concatenate

def load_patient_data(valid_path, patient_id):
    try:
        record = wfdb.rdrecord(f'{valid_path}/{patient_id}')
        annotation = wfdb.rdann(f'{valid_path}/{patient_id}', 'atr')
        
        # Check for lead inconsistencies
        sig_names = record.sig_name
        if TARGET_LEADS[0] not in sig_names or TARGET_LEADS[1] not in sig_names:
            print(f"\nSkipping Patient {patient_id}: Missing required leads {TARGET_LEADS}.")
            return None, None
            
        idx_lead_1 = sig_names.index(TARGET_LEADS[0])
        idx_lead_2 = sig_names.index(TARGET_LEADS[1])
        
        # Apply Bandpass Filter
        sig_1 = apply_bandpass_filter(record.p_signal[:, idx_lead_1])
        sig_2 = apply_bandpass_filter(record.p_signal[:, idx_lead_2])
        
        peaks = annotation.sample
        labels = annotation.symbol
        
        valid_peaks, valid_labels = [], []
        for p, l in zip(peaks, labels):
            if l in SYMBOL_MAP:
                valid_peaks.append(p)
                valid_labels.append(CLASSES.index(SYMBOL_MAP[l]))
                
        # Extract features & normalize
        seg_1, seg_2, final_peaks = extract_beats(sig_1, sig_2, valid_peaks)
        X_patient = normalize_and_concatenate(seg_1, seg_2)
        
        # Map labels
        y_patient = np.array([valid_labels[valid_peaks.index(p)] for p in final_peaks])
        return X_patient, y_patient
        
    except Exception as e:
        print(f"Error processing {patient_id}: {e}")
        return None, None

def patient_wise_split(valid_path, patient_ids):
    X_train, y_train, X_test, y_test = [], [], [], []
    
    for pid in tqdm(patient_ids):
        X_p, y_p = load_patient_data(valid_path, pid)
        if X_p is not None:
            if pid in TRAIN_PATIENTS:
                X_train.append(X_p)
                y_train.append(y_p)
            else:
                X_test.append(X_p)
                y_test.append(y_p)
                
    return (np.vstack(X_train), np.concatenate(y_train), 
            np.vstack(X_test), np.concatenate(y_test))