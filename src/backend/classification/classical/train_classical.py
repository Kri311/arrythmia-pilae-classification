"""
Classical Classification Benchmarking.

Trains and evaluates 6 classical models on PILAE+PCA features with
comprehensive metrics, timing, and confusion matrix analysis.
All models use class_weight='balanced' where supported.

Usage:
    python train_classical.py
"""
import os
import time
import numpy as np
import logging
from datetime import datetime
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Classification Models
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation Metrics
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             balanced_accuracy_score, f1_score, matthews_corrcoef)

# --- Setup Bulletproof Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..'))
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'extracted_features')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'classical')

# --- Setup Narrative Logging ---
log_dir = os.path.join(PROJECT_ROOT, 'logs', 'classical')
    os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, datetime.now().strftime('classical_training_%Y%m%d_%H%M%S.log'))

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create a folder to save the graphs
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'plots', 'classical')
os.makedirs(PLOTS_DIR, exist_ok=True)

AAMI_CLASSES_MULTI = ['N', 'S', 'V', 'F', 'Q']
AAMI_CLASSES_BINARY = ['N', 'A']
AAMI_CLASSES = AAMI_CLASSES_MULTI


def save_confusion_matrix_heatmap(y_true, y_pred, model_name, split_name='test', task='multi'):
    """Generates and saves a seaborn heatmap for the confusion matrix."""
    labels = [0,1,2,3,4] if task == 'multi' else [0,1]
    classes = AAMI_CLASSES_MULTI if task == 'multi' else AAMI_CLASSES_BINARY
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    
    plt.title(f'Confusion Matrix: {model_name} ({split_name})')
    plt.ylabel('True Heartbeat Class')
    plt.xlabel('Predicted Heartbeat Class')
    plt.tight_layout()
    
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plot_path = os.path.join(PLOTS_DIR, f"{safe_name}_{split_name}_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved Confusion Matrix Heatmap to: {plot_path}")


def evaluate_model(model_name, y_true, y_pred, split_name='test', task='multi'):
    """Calculates and logs all standard evaluation metrics."""
    labels = [0,1,2,3,4] if task == 'multi' else [0,1]
    classes = AAMI_CLASSES_MULTI if task == 'multi' else AAMI_CLASSES_BINARY
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION RESULTS: {model_name} ({split_name.upper()} SET)")
    logger.info(f"{'='*60}")
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    logger.info(f"Overall Accuracy:     {acc * 100:.2f}%")
    logger.info(f"Balanced Accuracy:    {bal_acc * 100:.2f}%")
    logger.info(f"Macro-F1:             {macro_f1:.4f}")
    logger.info(f"Weighted-F1:          {weighted_f1:.4f}")
    logger.info(f"MCC:                  {mcc:.4f}")
    
    logger.info("\nPer-Class Classification Report:")
    logger.info(classification_report(y_true, y_pred, labels=labels,
                                      target_names=classes, digits=4, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info("Confusion Matrix:")
    logger.info(np.array_str(cm))
    
    # Log dominant error transitions
    logger.info("\nDominant Error Transitions:")
    if task == 'multi':
        class_pairs = [(0, 1, 'N→S'), (0, 3, 'N→F'), (1, 0, 'S→N'), (4, 2, 'Q→V'),
                       (1, 2, 'S→V'), (2, 0, 'V→N'), (3, 0, 'F→N')]
    else:
        class_pairs = [(0, 1, 'N→A'), (1, 0, 'A→N')]
        
    for true_cls, pred_cls, label in class_pairs:
        if true_cls < cm.shape[0] and pred_cls < cm.shape[1]:
            count = cm[true_cls, pred_cls]
            total = cm[true_cls].sum()
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"  {label}: {count}/{total} ({pct:.1f}%)")
    
    logger.info(f"{'-'*60}\n")
    
    # Save the heatmap
    save_confusion_matrix_heatmap(y_true, y_pred, model_name, split_name, task=task)
    
    return {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'mcc': mcc
    }


def run_classical_training(task='multi'):
    logger.info("=" * 60)
    logger.info(f"STARTING CLASSICAL CLASSIFICATION BENCHMARKING (Task: {task})")
    logger.info("=" * 60)
    
    features_dir = FEATURES_DIR
    models_dir = MODELS_DIR
    global PLOTS_DIR
    plots_dir_base = PLOTS_DIR
    
    if task == 'binary':
        features_dir = os.path.join(FEATURES_DIR, 'binary')
        models_dir = os.path.join(MODELS_DIR, 'binary')
        PLOTS_DIR = os.path.join(PLOTS_DIR, 'binary')
    else:
        features_dir = os.path.join(FEATURES_DIR, 'multi')
        models_dir = os.path.join(MODELS_DIR, 'multi')
        PLOTS_DIR = os.path.join(PLOTS_DIR, 'multi')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # 1. Load the PILAE+PCA features (3-way split)
    logger.info(f"Loading compressed features from {features_dir}...")
    try:
        X_train = np.load(os.path.join(features_dir, 'Z_train_reduced.npy'))
        X_val = np.load(os.path.join(features_dir, 'Z_val_reduced.npy'))
        X_test = np.load(os.path.join(features_dir, 'Z_test_reduced.npy'))
        y_train = np.load(os.path.join(features_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(features_dir, 'y_val.npy'))
        y_test = np.load(os.path.join(features_dir, 'y_test.npy'))
    except FileNotFoundError:
        logger.error("Feature files not found! Make sure the PILAE pipeline finished successfully.")
        return
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Load class weights if available
    class_weights_path = os.path.join(PROJECT_ROOT, 'data', 'processed', task, 'class_weights.npy')
    xgb_weights = None
    if os.path.exists(class_weights_path):
        cw = np.load(class_weights_path)
        xgb_weights = {i: w for i, w in enumerate(cw)}
        logger.info(f"Loaded class weights: {xgb_weights}")
    
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Define models to benchmark
    models = {
        "Linear SVM (Primary Pre-Classifier)": LinearSVC(
            random_state=42, max_iter=10000, dual=False, class_weight='balanced'
        ),
        "RBF SVM (Calibrated)": SVC(
            kernel='rbf', probability=True, class_weight='balanced', random_state=42, cache_size=1000
        ),
        "Logistic Regression": LogisticRegression(
            random_state=42, max_iter=1000, class_weight='balanced'
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42, n_estimators=200, n_jobs=-1, class_weight='balanced'
        ),
        "XGBoost": XGBClassifier(
            random_state=42, eval_metric='mlogloss' if task == 'multi' else 'logloss', n_jobs=-1,
            sample_weight=None  # Will handle separately
        ),
    }
    
    results_summary = []
    
    # 3. Train and Evaluate each model
    for name, model in models.items():
        logger.info(f"\n{'#'*60}")
        logger.info(f"Training {name}...")
        logger.info(f"{'#'*60}")
        
        # --- Training ---
        train_start = time.perf_counter()
        
        if "XGBoost" in name and xgb_weights is not None:
            # XGBoost uses sample_weight instead of class_weight
            sample_weights = np.array([xgb_weights.get(y, 1.0) for y in y_train])
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        
        train_time = time.perf_counter() - train_start
        logger.info(f"Training time: {train_time:.2f}s")
        
        # --- Inference timing ---
        infer_start = time.perf_counter()
        y_pred_test = model.predict(X_test)
        infer_time = time.perf_counter() - infer_start
        logger.info(f"Inference time (test set): {infer_time:.4f}s")
        
        # --- Model size info ---
        if hasattr(model, 'n_estimators'):
            logger.info(f"Number of estimators/trees: {model.n_estimators}")
        if hasattr(model, 'support_vectors_'):
            logger.info(f"Number of support vectors: {model.support_vectors_.shape[0]}")
        
        # --- Evaluate on validation set ---
        y_pred_val = model.predict(X_val)
        val_metrics = evaluate_model(name, y_val, y_pred_val, split_name='val', task=task)
        
        # --- Evaluate on test set ---
        test_metrics = evaluate_model(name, y_test, y_pred_test, split_name='test', task=task)
        
        results_summary.append({
            'model': name,
            'train_time': train_time,
            'infer_time': infer_time,
            **{f'val_{k}': v for k, v in val_metrics.items()},
            **{f'test_{k}': v for k, v in test_metrics.items()}
        })
        
        # --- Save the trained model ---
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        model_path = os.path.join(models_dir, f"{safe_name}.pkl")
        joblib.dump(model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # --- SVM Confidence Score Extraction ---
        if "Linear SVM" in name or "RBF SVM" in name:
            logger.info(">>> EXTRACTING CONFIDENCE SCORES FOR FUTURE VQC ROUTING <<<")
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test[:3])
                logger.info("Example calibrated probabilities for first 3 test samples:")
                logger.info(f"{np.round(proba, 3)}")
            if hasattr(model, 'decision_function'):
                confidence_scores = model.decision_function(X_test[:3])
                logger.info("Example raw decision function values for first 3 test samples:")
                logger.info(f"{np.round(confidence_scores, 3)}")
    
    # --- Summary Table ---
    logger.info("\n" + "=" * 80)
    logger.info("CLASSICAL BASELINES — SUMMARY TABLE")
    logger.info("=" * 80)
    header = f"{'Model':<40} {'Val Acc':>8} {'Val MF1':>8} {'Test Acc':>8} {'Test MF1':>8} {'Test MCC':>8} {'Train(s)':>8}"
    logger.info(header)
    logger.info("-" * 80)
    for r in results_summary:
        row = (f"{r['model']:<40} "
               f"{r['val_accuracy']*100:>7.2f}% "
               f"{r['val_macro_f1']:>8.4f} "
               f"{r['test_accuracy']*100:>7.2f}% "
               f"{r['test_macro_f1']:>8.4f} "
               f"{r['test_mcc']:>8.4f} "
               f"{r['train_time']:>7.2f}s")
        logger.info(row)
    
    logger.info("\nAll classical models trained, evaluated, and saved to disk.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Classical Benchmarking')
    parser.add_argument('--task', type=str, default='multi', choices=['multi', 'binary'],
                        help='Classification task: multi (5-class) or binary (Normal vs Abnormal)')
    args = parser.parse_args()
    run_classical_training(task=args.task)