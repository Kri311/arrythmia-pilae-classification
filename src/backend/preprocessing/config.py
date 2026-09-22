"""
Configuration for ECG Morphology & Preprocessing Pipeline.

Contains dataset paths, signal processing parameters, AAMI class mapping,
and inter-patient split definitions (De Chazal protocol) with a validation set.
"""
import os

# Get the absolute path of the directory containing config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data Paths
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'data', 'mit-bih-arrhythmia-database-1.0.0'))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', 'processed_data'))

# Window Size Parameters (peak-99 to peak+99 = 198 samples per lead)
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

AAMI_CLASS_NAMES = {0: 'Normal (N)', 1: 'Supraventricular (S)', 2: 'Ventricular (V)',
                    3: 'Fusion (F)', 4: 'Unknown (Q)'}

# ---------------------------------------------------------------------------
# Standard Inter-Patient Split (De Chazal Split)
# DS1 for Train + Validation, DS2 for Test (to test generalization)
# ---------------------------------------------------------------------------

# Full DS1 pool (original 22 records)
DS1_FULL = ['101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
            '122', '124', '201', '203', '205', '207', '208', '209', '215', '220',
            '223', '230']

# Validation records held out from DS1.
# Chosen for arrhythmia class diversity (200-series records contain more minority classes).
DS1_VAL = ['201', '207', '209', '223']

# Training records = DS1_FULL minus DS1_VAL (18 records)
DS1_TRAIN = [r for r in DS1_FULL if r not in DS1_VAL]

# Test records (23 records — never touched until final evaluation)
DS2_TEST = ['100', '103', '104', '105', '111', '113', '117', '121', '123',
            '200', '202', '210', '212', '213', '214', '219', '221', '222',
            '228', '231', '232', '233', '234']