from scipy.signal import butter, filtfilt
import config

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal_data):
    """
    Applies a Butterworth bandpass filter to the continuous ECG signal.
    """
    b, a = butter_bandpass(config.LOWCUT, config.HIGHCUT, config.FS, order=config.ORDER)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)
    return filtered_signal