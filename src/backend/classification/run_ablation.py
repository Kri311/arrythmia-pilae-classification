"""
Ablation Study Orchestrator.

Runs all ablation experiments for the hybrid PILAE-VQC arrhythmia classification
pipeline and collects results into a summary table for publication.

Usage:
    python run_ablation.py                # Run all experiments
    python run_ablation.py --experiment 1 # Run specific experiment
"""
import os
import sys
import subprocess
import argparse
import csv
import logging
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..'))
PREPROC_DIR = os.path.join(PROJECT_ROOT, 'src', 'backend', 'preprocessing')
PILAE_DIR = os.path.join(PROJECT_ROOT, 'src', 'backend', 'pilae')
CLASSICAL_DIR = os.path.join(BASE_DIR, 'classical')
VQC_DIR = os.path.join(BASE_DIR, 'vqc')

# --- Setup Logging ---
os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
log_filename = datetime.now().strftime(os.path.join(BASE_DIR, 'logs', 'ablation_%Y%m%d_%H%M%S.log'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Results output
RESULTS_DIR = os.path.join(BASE_DIR, 'ablation_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_step(description, cmd, cwd):
    """Run a subprocess and log its output."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"CMD:  {' '.join(cmd)}")
    logger.info(f"CWD:  {cwd}")
    logger.info(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=7200)
    
    if result.stdout:
        for line in result.stdout.strip().split('\n')[-50:]:  # Last 50 lines
            logger.info(f"  | {line}")
    
    if result.returncode != 0:
        logger.error(f"FAILED (exit code {result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().split('\n')[-20:]:
                logger.error(f"  ! {line}")
        return False
    
    logger.info(f"COMPLETED SUCCESSFULLY")
    return True


# ---------------------------------------------------------------------------
# Experiment Definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    # --- Data augmentation ablation ---
    1: {
        'name': 'No SMOTE (raw imbalanced)',
        'preproc_args': ['--augmentation', 'none'],
        'classical': True,
        'hybrid': None,
    },
    2: {
        'name': 'Class weighting only',
        'preproc_args': ['--augmentation', 'class_weight'],
        'classical': True,
        'hybrid': None,
    },
    3: {
        'name': 'Moderate SMOTE',
        'preproc_args': ['--augmentation', 'moderate_smote'],
        'classical': True,
        'hybrid': None,
    },
    4: {
        'name': 'Moderate SMOTE + class weighting',
        'preproc_args': ['--augmentation', 'smote_plus_weight'],
        'classical': True,
        'hybrid': None,
    },
    
    # --- Hybrid routing ablation (uses best augmentation from above) ---
    5: {
        'name': 'Hybrid: LinearSVC router (τ sweep)',
        'preproc_args': None,  # Skip preprocessing (reuse from best augmentation)
        'classical': False,
        'hybrid': ['--router', 'linear_svm', '--sweep_tau'],
    },
    6: {
        'name': 'Hybrid: RBF-SVM router (τ sweep)',
        'preproc_args': None,
        'classical': False,
        'hybrid': ['--router', 'rbf_svm', '--sweep_tau'],
    },
    7: {
        'name': 'Hybrid: Logistic Regression router (τ sweep)',
        'preproc_args': None,
        'classical': False,
        'hybrid': ['--router', 'logistic_regression', '--sweep_tau'],
    },
    8: {
        'name': 'Hybrid: RBF-SVM entropy routing (τ sweep)',
        'preproc_args': None,
        'classical': False,
        'hybrid': ['--router', 'rbf_svm', '--sweep_tau', '--routing_strategy', 'entropy'],
    },
    
    # --- Qubit ablation ---
    9: {
        'name': 'VQC: 4 qubits (RBF-SVM router)',
        'preproc_args': None,
        'classical': False,
        'hybrid': ['--router', 'rbf_svm', '--sweep_tau', '--n_qubits', '4'],
    },
    10: {
        'name': 'VQC: 8 qubits (RBF-SVM router)',
        'preproc_args': None,
        'classical': False,
        'hybrid': ['--router', 'rbf_svm', '--sweep_tau', '--n_qubits', '8'],
    },
}


def run_experiment(exp_id, python_exe='python'):
    """Run a single ablation experiment."""
    exp = EXPERIMENTS[exp_id]
    logger.info(f"\n{'#'*70}")
    logger.info(f"EXPERIMENT {exp_id}: {exp['name']}")
    logger.info(f"{'#'*70}")
    
    # Step 1: Preprocessing (if needed)
    if exp['preproc_args'] is not None:
        ok = run_step(
            f"Preprocessing ({exp['name']})",
            [python_exe, 'main.py'] + exp['preproc_args'],
            cwd=PREPROC_DIR
        )
        if not ok:
            logger.error(f"Experiment {exp_id} FAILED at preprocessing")
            return False
        
        # Step 2: Feature extraction
        ok = run_step(
            f"PILAE + PCA Feature Extraction",
            [python_exe, 'feature_pipeline.py'],
            cwd=PILAE_DIR
        )
        if not ok:
            logger.error(f"Experiment {exp_id} FAILED at feature extraction")
            return False
    
    # Step 3: Classical baselines
    if exp['classical']:
        ok = run_step(
            f"Classical Baselines",
            [python_exe, 'train_classical.py'],
            cwd=CLASSICAL_DIR
        )
        if not ok:
            logger.error(f"Experiment {exp_id} FAILED at classical training")
            return False
    
    # Step 4: Hybrid pipeline
    if exp['hybrid'] is not None:
        ok = run_step(
            f"Hybrid Pipeline ({exp['name']})",
            [python_exe, 'train_hybrid.py'] + exp['hybrid'],
            cwd=VQC_DIR
        )
        if not ok:
            logger.error(f"Experiment {exp_id} FAILED at hybrid training")
            return False
    
    logger.info(f"EXPERIMENT {exp_id} COMPLETED SUCCESSFULLY")
    return True


def run_all_experiments(python_exe='python'):
    """Run all ablation experiments sequentially."""
    logger.info("=" * 70)
    logger.info("STARTING COMPLETE ABLATION STUDY")
    logger.info(f"Total experiments: {len(EXPERIMENTS)}")
    logger.info("=" * 70)
    
    results = {}
    for exp_id in sorted(EXPERIMENTS.keys()):
        success = run_experiment(exp_id, python_exe)
        results[exp_id] = success
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 70)
    for exp_id, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  Exp {exp_id:>2}: {status} — {EXPERIMENTS[exp_id]['name']}")
    
    logger.info(f"\nDetailed results are in the individual log files under each module's logs/ directory.")
    logger.info(f"Ablation log saved to: {log_filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ablation Study Orchestrator')
    parser.add_argument('--experiment', type=int, default=None,
                        help='Run a specific experiment by ID (1-10). Default: run all.')
    parser.add_argument('--python', type=str, default='python',
                        help='Python executable path')
    args = parser.parse_args()
    
    if args.experiment is not None:
        if args.experiment not in EXPERIMENTS:
            print(f"Error: Experiment {args.experiment} not found. Valid: {sorted(EXPERIMENTS.keys())}")
            sys.exit(1)
        run_experiment(args.experiment, python_exe=args.python)
    else:
        run_all_experiments(python_exe=args.python)
