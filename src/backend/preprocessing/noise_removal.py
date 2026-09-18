import numpy as np
from scipy.signal import butter, filtfilt
from src.config import LOWCUT, HIGHCUT, FS

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data):
    """Applies a Butterworth bandpass filter to a continuous ECG signal."""
    b, a = butter_bandpass(LOWCUT, HIGHCUT, FS)
    y = filtfilt(b, a, data)
    return y