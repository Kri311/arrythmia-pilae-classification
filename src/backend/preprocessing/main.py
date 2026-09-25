"""
ECG Morphology & Preprocessing Pipeline — Main Orchestrator.

Runs the full preprocessing pipeline: record loading, beat extraction,
patient-wise 3-way split, optional augmentation, and global scaling.

Usage:
    python main.py --augmentation class_weight
    python main.py --augmentation moderate_smote
    python main.py --augmentation none
    python main.py --augmentation smote_plus_weight
"""
import os
import sys
import argparse
import logging
from datetime import datetime
from collections import Counter
import numpy as np
import config
from data_split import load_and_split_data
from data_augmentation import augment_data
from global_scaling import apply_global_scaling

# --- Setup Narrative Logging ---
PROJECT_ROOT = os.path.abspath(os.path.join(config.BASE_DIR, '..', '..', '..'))
    log_dir = os.path.join(PROJECT_ROOT, 'logs', 'preprocessing')
    os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, datetime.now().strftime('morphology_pipeline_%Y%m%d_%H%M%S.log'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def log_class_distribution(y, split_name):
    """Logs the class distribution for a given split."""
    counts = dict(sorted(Counter(y).items()))
    total = len(y)
    logger.info(f"{split_name} class distribution: {counts} (total={total})")
    for cls_id, cls_name in sorted(config.AAMI_CLASS_NAMES.items()):
        count = counts.get(cls_id, 0)
        pct = count / total * 100 if total > 0 else 0.0
        logger.info(f"  Class {cls_id} ({cls_name}): {count} ({pct:.1f}%)")


def run_preprocessing(augmentation_strategy='class_weight', task='multi'):
    logger.info("=" * 60)
    logger.info("STARTING ECG MORPHOLOGY & PREPROCESSING PIPELINE")
    logger.info(f"Augmentation Strategy: {augmentation_strategy} | Task: {task}")
    logger.info("=" * 60)
    
    # --- Phase 1-5: Load records, filter, extract, normalize, split ---
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data()
    
    if task == 'binary':
        logger.info("Remapping labels for binary classification (Normal=0, Abnormal=1)...")
        y_train = np.where(y_train > 0, 1, 0)
        y_val = np.where(y_val > 0, 1, 0)
        y_test = np.where(y_test > 0, 1, 0)
        config.AAMI_CLASS_NAMES = {0: 'Normal (N)', 1: 'Abnormal (A)'}
        config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, 'binary')
    else:
        config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, 'multi')
    
    logger.info("=" * 60)
    logger.info("ALL RECORDS PROCESSED — RAW CLASS DISTRIBUTIONS")
    logger.info("=" * 60)
    log_class_distribution(y_train, "Train (raw)")
    log_class_distribution(y_val, "Val (raw)")
    log_class_distribution(y_test, "Test (raw)")
    logger.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    
    # --- Phase 6: Data Augmentation (TRAIN ONLY) ---
    logger.info("=" * 60)
    logger.info(f"Phase 6 (Augmentation): Strategy = '{augmentation_strategy}' — applied to TRAIN set ONLY")
    logger.info("=" * 60)
    
    X_train_aug, y_train_aug, class_weights = augment_data(X_train, y_train, strategy=augmentation_strategy)
    
    logger.info(f"Train shape after augmentation: {X_train_aug.shape}")
    log_class_distribution(y_train_aug, "Train (after augmentation)")
    
    if class_weights is not None:
        logger.info(f"Class weights computed: {dict(enumerate(class_weights))}")
    
    # --- Phase 7: Global Scaling (fit on train only) ---
    logger.info("=" * 60)
    logger.info("Phase 7 (Global Scaling): Fitting StandardScaler on TRAINING data only")
    logger.info("=" * 60)
    
    X_train_scaled, X_val_scaled, X_test_scaled = apply_global_scaling(
        X_train_aug, X_val, X_test, output_dir=config.OUTPUT_DIR
    )
    
    logger.info("Effect: Shifted the entire 396-dimensional space to zero-mean and unit-variance. "
                "Scaler fitted on training data only — no data leakage.")
    
    # --- Phase 8: Export ---
    logger.info("=" * 60)
    logger.info("Phase 8 (Export): Saving final processed matrices to disk")
    logger.info("=" * 60)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    np.save(os.path.join(config.OUTPUT_DIR, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(config.OUTPUT_DIR, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(config.OUTPUT_DIR, 'X_test.npy'), X_test_scaled)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_train.npy'), y_train_aug)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(config.OUTPUT_DIR, 'y_test.npy'), y_test)
    
    if class_weights is not None:
        np.save(os.path.join(config.OUTPUT_DIR, 'class_weights.npy'), class_weights)
        logger.info(f"Saved class_weights.npy to {config.OUTPUT_DIR}/")
    
    logger.info(f"PIPELINE COMPLETE! Final data exported to {config.OUTPUT_DIR}/")
    logger.info(f"  Train: {X_train_scaled.shape}  Val: {X_val_scaled.shape}  Test: {X_test_scaled.shape}")
    logger.info(f"Full transformation log saved to {log_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ECG Preprocessing Pipeline')
    parser.add_argument('--augmentation', type=str, default='class_weight',
                        choices=['none', 'class_weight', 'moderate_smote', 'smote_plus_weight'],
                        help='Data augmentation strategy for the training set')
    parser.add_argument('--task', type=str, default='multi', choices=['multi', 'binary'],
                        help='Classification task: multi (5-class) or binary (Normal vs Abnormal)')
    args = parser.parse_args()
    
    run_preprocessing(augmentation_strategy=args.augmentation, task=args.task)