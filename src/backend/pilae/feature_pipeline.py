"""
PILAE + PCA Feature Extraction Pipeline.

Loads preprocessed ECG data (train/val/test), compresses through the PILAE
autoencoder, applies PCA for dimensionality reduction, and exports
classification-ready feature vectors. All fitting is done on training data only.

Usage:
    python feature_pipeline.py                     # Default: 95% variance PCA
    python feature_pipeline.py --pca_components 8  # Fixed 8 dimensions
    python feature_pipeline.py --pca_components 42 # Fixed 42 dimensions
"""
import os
import sys
import argparse
import numpy as np
import logging
import joblib
from datetime import datetime
from pilae_model import PILAE
from pca_dim_reduction import apply_pca

# --- Setup Bulletproof Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
PREPROC_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'extracted_features')

# --- Setup Narrative Logging ---
log_dir = os.path.join(PROJECT_ROOT, 'logs', 'pilae')
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, datetime.now().strftime('pilae_pipeline_%Y%m%d_%H%M%S.log'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_feature_extraction(pca_components=0.95, task='multi'):
    logger.info("=" * 60)
    logger.info(f"STARTING PILAE + PCA FEATURE EXTRACTION PIPELINE (Task: {task})")
    logger.info(f"PCA components setting: {pca_components}")
    logger.info("=" * 60)
    
    preproc_dir = PREPROC_DIR
    output_dir = OUTPUT_DIR
    
    if task == 'binary':
        preproc_dir = os.path.join(preproc_dir, 'binary')
        output_dir = os.path.join(output_dir, 'binary')
    else:
        preproc_dir = os.path.join(preproc_dir, 'multi')
        output_dir = os.path.join(output_dir, 'multi')
    
    # 1. Load Preprocessed Data (3-way split)
    logger.info(f"Loading standardized data from {preproc_dir}...")
    X_train = np.load(os.path.join(preproc_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(preproc_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(preproc_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(preproc_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(preproc_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(preproc_dir, 'y_test.npy'))
    
    logger.info(f"Loaded Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 2. PILAE Feature Extraction (fit on TRAIN only)
    hidden_dim = 250  # Enforce a bottleneck to compress the 396-dim input
    logger.info(f"Phase 1 (PILAE): Initializing non-iterative AutoEncoder with {hidden_dim} hidden neurons.")
    logger.info("NOTE: PILAE encoder fitted on TRAINING data only — no data leakage.")
    
    pilae = PILAE(hidden_dim=hidden_dim)
    pilae.fit(X_train)
    
    logger.info("Transforming train, val, and test sets into PILAE hidden representations...")
    Z_train = pilae.transform(X_train)
    Z_val = pilae.transform(X_val)
    Z_test = pilae.transform(X_test)
    logger.info(f"Effect: Compressed high-dimensional ECG into robust {Z_train.shape[1]}-dimensional space.")
    
    # 3. PCA Dimensionality Reduction (fit on TRAIN only)
    logger.info("Phase 2 (PCA): Applying PCA on PILAE features to eliminate redundancy.")
    logger.info("NOTE: PCA fitted on TRAINING data only — no data leakage.")
    Z_train_reduced, Z_val_reduced, Z_test_reduced, pca_model = apply_pca(
        Z_train, Z_val, Z_test, n_components=pca_components
    )
    
    # 4. Export the final features for the classification models
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Phase 3 (Export): Saving final learned feature vectors...")
    np.save(os.path.join(output_dir, 'Z_train_reduced.npy'), Z_train_reduced)
    np.save(os.path.join(output_dir, 'Z_val_reduced.npy'), Z_val_reduced)
    np.save(os.path.join(output_dir, 'Z_test_reduced.npy'), Z_test_reduced)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # Save fitted models for reproducibility
    models_dir = os.path.join(output_dir, 'fitted_models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(pilae, os.path.join(models_dir, 'pilae_model.pkl'))
    joblib.dump(pca_model, os.path.join(models_dir, 'pca_model.pkl'))
    logger.info(f"Saved fitted PILAE and PCA models to {models_dir}/")
    
    logger.info(f"PIPELINE COMPLETE! Final classification-ready features saved to {output_dir}/")
    logger.info(f"  Train: {Z_train_reduced.shape}  Val: {Z_val_reduced.shape}  Test: {Z_test_reduced.shape}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PILAE + PCA Feature Extraction')
    parser.add_argument('--pca_components', type=str, default='0.95',
                        help='PCA components: float (0-1) for variance retention, '
                             'or int for fixed dimensions. Default: 0.95')
    parser.add_argument('--task', type=str, default='multi', choices=['multi', 'binary'],
                        help='Classification task: multi (5-class) or binary (Normal vs Abnormal)')
    args = parser.parse_args()
    
    # Parse the pca_components argument
    try:
        pca_val = int(args.pca_components)
    except ValueError:
        pca_val = float(args.pca_components)
    
    run_feature_extraction(pca_components=pca_val, task=args.task)