import os
import logging
from datetime import datetime
import numpy as np
import config
from data_split import load_and_split_data
from data_augmentation import augment_data
from global_scaling import apply_global_scaling

# --- Setup Narrative Logging ---
os.makedirs('logs', exist_ok=True)
log_filename = datetime.now().strftime('logs/morphology_pipeline_%Y%m%d_%H%M%S.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s', # Simplified format for narrative reading
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_preprocessing():
    logger.info("======================================================")
    logger.info("STARTING ECG MORPHOLOGY & PREPROCESSING PIPELINE")
    logger.info("======================================================")
    
    # Phases 1-5 happen inside here
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    logger.info("======================================================")
    logger.info(f"GLOBAL TRAIN SET AGGREGATED. Shape: {X_train.shape}")
    logger.info("======================================================")
    
    # Phase 6: Data Augmentation
    logger.info("Phase 6 (Augmentation): Balancing dataset via SMOTE & Transforms...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train)
    logger.info("Effect: SMOTE mathematically synthesized new morphological 'twins' for rare arrhythmias. "
                "Random Gaussian noise and signal shifting were injected to force the model to ignore minor artifacts. "
                f"New balanced Train Shape: {X_train_aug.shape}.")
    
    # Phase 7: Global Scaling
    logger.info("Phase 7 (Global Scaling): Fitting StandardScaler on training data...")
    X_train_scaled, X_test_scaled = apply_global_scaling(X_train_aug, X_test)
    logger.info("Effect: Shifted the entire 396-dimensional space to zero-mean and unit-variance. "
                "This ensures the downstream SVM and VQC classifiers process all time-steps fairly, "
                "morphing the final data into a perfectly standardized input tensor.")
    
    logger.info("Phase 8 (Export): Saving final morphed matrices to disk...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(config.OUTPUT_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(config.OUTPUT_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_train.npy'), y_train_aug)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_test.npy'), y_test)
    
    logger.info(f"PIPELINE COMPLETE! Final data exported to {config.OUTPUT_DIR}/")
    logger.info(f"Full transformation log saved to {log_filename}")

if __name__ == '__main__':
    run_preprocessing()