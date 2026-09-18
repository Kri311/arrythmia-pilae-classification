import os

# Get the absolute path of the directory containing config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths (Update this if your raw dataset is located elsewhere)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'data', 'mit-bih-arrhythmia-database-1.0.0'))
OUTPUT_DIR = 'processed_data'

# Window Size Parameters (peak-99 to peak+99 = 198 samples)
WINDOW_LEFT = 99
WINDOW_RIGHT = 99
TOTAL_SAMPLES = WINDOW_LEFT + WINDOW_RIGHT

# Butterworth Filter Parameters
LOWCUT = 0.5
HIGHCUT = 45.0
FS = 360.0
ORDER = 1

# AAMI Class Mapping (Normal, Supraventricular, Ventricular, Fusion, Unknown)
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0, 
    'A': 1, 'a': 1, 'J': 1, 'S': 1,         
    'V': 2, 'E': 2,                             
    'F': 3,                                     
    '/': 4, 'f': 4, 'Q': 4                      
}

# Standard Inter-Patient Split (De Chazal Split) 
# DS1 for Train (to learn morphology), DS2 for Test (to test generalization)
DS1_TRAIN = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230']
DS2_TEST = ['100', '103', '104', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234']