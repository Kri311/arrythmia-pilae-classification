# AAMI EC57 standard patient split (Inter-patient paradigm)
TRAIN_PATIENTS = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']

# Preprocessing parameters
WINDOW_SIZE = 99
TARGET_LEADS = ['MLII', 'V1'] # Strict lead requirement to handle inconsistencies

# Butterworth Bandpass Filter parameters
LOWCUT = 0.5
HIGHCUT = 45.0
FS = 360.0

# Augmentation params
NOISE_FACTOR = 0.01
SHIFT_SAMPLES = 5