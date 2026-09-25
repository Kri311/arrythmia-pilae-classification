"""
Hybrid Classical + VQC Classification Pipeline.

Implements the calibrated uncertainty-based routing mechanism:
  - Confident samples → Classical classifier prediction
  - Uncertain samples  → VQC prediction

Supports τ threshold tuning on the validation set, multiple routing strategies,
comprehensive evaluation metrics, and proper VQC training with mini-batches,
learning rate scheduling, weighted loss, and early stopping.

Usage:
    python train_hybrid.py --router rbf_svm --tau 0.80
    python train_hybrid.py --router logistic_regression --tau 0.70
    python train_hybrid.py --router rbf_svm --sweep_tau
    python train_hybrid.py --router rbf_svm --tau 0.80 --n_qubits 4
"""
import os
import sys
import time
import argparse
import numpy as np
import logging
from datetime import datetime
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score, matthews_corrcoef)

from quantum_model import VQC

# --- Setup Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..', '..', '..'))
FEATURES_DIR = os.path.join(PROJECT_ROOT, 'data', 'extracted_features')
CLASSICAL_MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'classical')

# --- Setup Narrative Logging ---
log_dir = os.path.join(PROJECT_ROOT, 'logs', 'vqc')
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, datetime.now().strftime('hybrid_vqc_%Y%m%d_%H%M%S.log'))

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
    logging.FileHandler(log_filename), logging.StreamHandler()
])
logger = logging.getLogger(__name__)

PLOTS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'plots', 'vqc')
os.makedirs(PLOTS_DIR, exist_ok=True)

AAMI_CLASSES_MULTI = ['N', 'S', 'V', 'F', 'Q']
AAMI_CLASSES_BINARY = ['N', 'A']

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
    plot_path = os.path.join(PLOTS_DIR_TASK, f"{safe_name}_{split_name}_heatmap.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved heatmap: {plot_path}")


def compute_metrics(y_true, y_pred, prefix=''):
    """Compute all evaluation metrics and return as a dict."""
    return {
        f'{prefix}accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        f'{prefix}macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        f'{prefix}weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        f'{prefix}mcc': matthews_corrcoef(y_true, y_pred),
    }


def log_detailed_metrics(y_true, y_pred, model_name, split_name='test', task='multi'):
    """Log detailed metrics including confusion matrix and error transitions."""
    metrics = compute_metrics(y_true, y_pred)
    labels = [0,1,2,3,4] if task == 'multi' else [0,1]
    classes = AAMI_CLASSES_MULTI if task == 'multi' else AAMI_CLASSES_BINARY
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATION: {model_name} ({split_name.upper()})")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy:     {metrics['accuracy']*100:.2f}%")
    logger.info(f"Balanced Accuracy:    {metrics['balanced_accuracy']*100:.2f}%")
    logger.info(f"Macro-F1:             {metrics['macro_f1']:.4f}")
    logger.info(f"Weighted-F1:          {metrics['weighted_f1']:.4f}")
    logger.info(f"MCC:                  {metrics['mcc']:.4f}")
    
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_true, y_pred, labels=labels,
                                      target_names=classes, digits=4, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    logger.info("Confusion Matrix:")
    logger.info(np.array_str(cm))
    
    # Dominant error transitions
    logger.info("\nDominant Error Transitions:")
    if task == 'multi':
        class_pairs = [(0,1,'N→S'), (0,3,'N→F'), (1,0,'S→N'),
                       (4,2,'Q→V'), (1,2,'S→V'), (2,0,'V→N')]
    else:
        class_pairs = [(0,1,'N→A'), (1,0,'A→N')]
        
    for true_cls, pred_cls, label in class_pairs:
        if true_cls < cm.shape[0] and pred_cls < cm.shape[1]:
            count = cm[true_cls, pred_cls]
            total = cm[true_cls].sum()
            pct = count / total * 100 if total > 0 else 0
            logger.info(f"  {label}: {count}/{total} ({pct:.1f}%)")
    
    save_confusion_matrix_heatmap(y_true, y_pred, model_name, split_name, task=task)
    return metrics


def get_routing_mask(router_model, X, tau, strategy='probability'):
    """
    Compute routing decisions: which samples go to classical vs VQC.
    
    Args:
        router_model: A fitted classifier with predict_proba or decision_function.
        X: Feature matrix.
        tau: Confidence threshold.
        strategy: 'probability' (uses p_max) or 'entropy' (uses H(x)).
    
    Returns:
        confident_mask: Boolean array (True = classical, False = VQC)
        classical_preds: Predictions from the classical model for all samples
        confidence_scores: The confidence metric for each sample
    """
    classical_preds = router_model.predict(X)
    
    if strategy == 'probability':
        if hasattr(router_model, 'predict_proba'):
            proba = router_model.predict_proba(X)
            p_max = np.max(proba, axis=1)
            confident_mask = p_max > tau
            confidence_scores = p_max
        else:
            # Fallback for LinearSVC: use absolute decision function
            df = np.abs(router_model.decision_function(X))
            if df.ndim > 1:
                confidence_scores = np.max(df, axis=1)
            else:
                confidence_scores = df
            confident_mask = confidence_scores > tau
    
    elif strategy == 'entropy':
        if not hasattr(router_model, 'predict_proba'):
            raise ValueError("Entropy routing requires predict_proba")
        proba = router_model.predict_proba(X)
        n_classes = proba.shape[1]
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        max_entropy = np.log(n_classes)
        normalized_entropy = entropy / max_entropy
        # Low entropy = confident → classical; High entropy = uncertain → VQC
        confident_mask = normalized_entropy < tau
        confidence_scores = normalized_entropy
    
    else:
        raise ValueError(f"Unknown routing strategy: {strategy}")
    
    return confident_mask, classical_preds, confidence_scores


def train_vqc(X_train, y_train, X_val, y_val, n_qubits, n_layers, n_classes,
              device, epochs=100, batch_size=32, lr=0.01, patience=15):
    """
    Train the VQC on the provided (ambiguous) training data.
    Uses mini-batch training, weighted loss, LR scheduling, and early stopping.
    
    Returns:
        model: Trained VQC model
        q_pca: Fitted PCA model for quantum dimensionality reduction
        history: dict with training and validation loss per epoch
    """
    logger.info(f"Initializing VQC: {n_qubits} qubits, {n_layers} layers, {n_classes} classes")
    
    # --- PCA for quantum (reduce to n_qubits dimensions) ---
    feature_dim = X_train.shape[1]
    if feature_dim > n_qubits:
        logger.info(f"Applying VQC PCA: {feature_dim} → {n_qubits} dimensions (fit on VQC training data)")
        q_pca = PCA(n_components=n_qubits, random_state=42)
        X_train_q = q_pca.fit_transform(X_train)
        X_val_q = q_pca.transform(X_val) if len(X_val) > 0 else np.array([])
    else:
        q_pca = None
        X_train_q = X_train
        X_val_q = X_val
    
    # --- Convert to tensors ---
    X_train_tensor = torch.tensor(X_train_q, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    
    # --- Class-weighted loss ---
    class_counts = np.bincount(y_train, minlength=n_classes).astype(float)
    class_counts[class_counts == 0] = 1.0  # Avoid div by zero
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"VQC class weights: {dict(enumerate(np.round(weights, 3)))}")
    
    # --- Model, loss, optimizer, scheduler ---
    model = VQC(n_qubits=n_qubits, n_layers=n_layers, n_classes=n_classes, scale_to_pi=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    # --- DataLoader ---
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # --- Training loop with early stopping ---
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    logger.info(f"Training VQC on {len(X_train)} samples over up to {epochs} epochs (patience={patience})...")
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_train_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_train_loss)
        scheduler.step()
        
        # --- Validation loss ---
        val_loss = float('inf')
        if len(X_val_q) > 0:
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.tensor(X_val_q, dtype=torch.float32).to(device)
                y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
        history['val_loss'].append(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f})")
                model.load_state_dict(best_state)
                break
    
    logger.info(f"VQC Training Complete. Final train loss: {history['train_loss'][-1]:.4f}, "
                f"Best val loss: {best_val_loss:.4f}")
    
    return model, q_pca, history


def run_hybrid_pipeline(router_name='rbf_svm', tau=0.80, sweep_tau=False,
                         n_qubits=8, n_layers=2, routing_strategy='probability',
                         epochs=100, batch_size=32, task='multi'):
    """Run the complete hybrid classification pipeline."""
    
    logger.info("=" * 60)
    logger.info(f"HYBRID {router_name.upper()} + VQC CLASSIFICATION (Task: {task})")
    logger.info(f"Router: {router_name} | τ: {tau} | Routing: {routing_strategy}")
    logger.info(f"VQC: {n_qubits} qubits, {n_layers} layers, {epochs} max epochs")
    logger.info("=" * 60)
    
    features_dir = FEATURES_DIR
    classical_models_dir = CLASSICAL_MODELS_DIR
    
    if task == 'binary':
        features_dir = os.path.join(FEATURES_DIR, 'binary')
        classical_models_dir = os.path.join(CLASSICAL_MODELS_DIR, 'binary')
    else:
        features_dir = os.path.join(FEATURES_DIR, 'multi')
        classical_models_dir = os.path.join(CLASSICAL_MODELS_DIR, 'multi')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"PyTorch Compute Device: {device.type.upper()}")
    
    # 1. Load features
    logger.info(f"Loading features from {features_dir}...")
    X_train = np.load(os.path.join(features_dir, 'Z_train_reduced.npy'))
    X_val = np.load(os.path.join(features_dir, 'Z_val_reduced.npy'))
    X_test = np.load(os.path.join(features_dir, 'Z_test_reduced.npy'))
    y_train = np.load(os.path.join(features_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(features_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(features_dir, 'y_test.npy'))
    
    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}, Classes: {n_classes}")
    
    # 2. Load the pre-trained classical router
    router_filename_map = {
        'rbf_svm': 'rbf_svm_calibrated.pkl',
        'logistic_regression': 'logistic_regression.pkl',
        'linear_svm': 'linear_svm_primary_pre-classifier.pkl',
        'random_forest': 'random_forest.pkl',
    }
    router_file = router_filename_map.get(router_name, f'{router_name}.pkl')
    router_path = os.path.join(classical_models_dir, router_file)
    logger.info(f"Loading router model: {router_path}")
    router_model = joblib.load(router_path)
    
    # 3. Route training data
    train_confident_mask, train_classical_preds, train_scores = get_routing_mask(
        router_model, X_train, tau, strategy=routing_strategy
    )
    ambiguous_indices = np.where(~train_confident_mask)[0]
    
    X_train_ambiguous = X_train[ambiguous_indices]
    y_train_ambiguous = y_train[ambiguous_indices]
    
    logger.info(f"\n--- TRAIN SET ROUTING (τ={tau}) ---")
    logger.info(f"Total:     {len(X_train)}")
    logger.info(f"Classical: {train_confident_mask.sum()} ({train_confident_mask.mean()*100:.1f}%)")
    logger.info(f"VQC:       {len(ambiguous_indices)} ({len(ambiguous_indices)/len(X_train)*100:.1f}%)")
    
    if len(X_train_ambiguous) < 10:
        logger.warning(f"Only {len(X_train_ambiguous)} ambiguous samples! τ is too low. Aborting VQC training.")
        # Fall back to classical-only evaluation
        y_pred_test = router_model.predict(X_test)
        log_detailed_metrics(y_test, y_pred_test, f"Classical-Only ({router_name})", 'test', task=task)
        return
    
    # 4. Route validation data for VQC early stopping
    val_confident_mask, val_classical_preds, val_scores = get_routing_mask(
        router_model, X_val, tau, strategy=routing_strategy
    )
    val_ambiguous_indices = np.where(~val_confident_mask)[0]
    X_val_ambiguous = X_val[val_ambiguous_indices] if len(val_ambiguous_indices) > 0 else np.array([]).reshape(0, X_val.shape[1])
    y_val_ambiguous = y_val[val_ambiguous_indices] if len(val_ambiguous_indices) > 0 else np.array([])
    
    logger.info(f"\n--- VAL SET ROUTING (τ={tau}) ---")
    logger.info(f"Total:     {len(X_val)}")
    logger.info(f"Classical: {val_confident_mask.sum()} ({val_confident_mask.mean()*100:.1f}%)")
    logger.info(f"VQC:       {len(val_ambiguous_indices)} ({len(val_ambiguous_indices)/len(X_val)*100:.1f}%)")
    
    # 5. Train VQC
    train_start = time.perf_counter()
    vqc_model, q_pca, history = train_vqc(
        X_train_ambiguous, y_train_ambiguous,
        X_val_ambiguous, y_val_ambiguous,
        n_qubits=n_qubits, n_layers=n_layers, n_classes=n_classes,
        device=device, epochs=epochs, batch_size=batch_size
    )
    train_time = time.perf_counter() - train_start
    logger.info(f"VQC training time: {train_time:.2f}s")
    
    # 6. τ Sweep on validation set (if requested)
    if sweep_tau:
        logger.info("\n" + "=" * 60)
        logger.info("τ THRESHOLD SWEEP ON VALIDATION SET")
        logger.info("=" * 60)
        
        tau_values = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]
        sweep_results = []
        
        header = f"{'τ':>5} {'ClassPct':>8} {'VQCPct':>8} {'Accuracy':>8} {'BalAcc':>8} {'MacroF1':>8} {'MCC':>8}"
        logger.info(header)
        logger.info("-" * 65)
        
        for t in tau_values:
            y_pred_val = _hybrid_predict(
                router_model, vqc_model, q_pca, X_val, t,
                routing_strategy, device, n_classes
            )
            
            conf_mask, _, _ = get_routing_mask(router_model, X_val, t, strategy=routing_strategy)
            vqc_pct = (~conf_mask).mean() * 100
            class_pct = conf_mask.mean() * 100
            
            m = compute_metrics(y_val, y_pred_val)
            sweep_results.append({'tau': t, **m, 'vqc_pct': vqc_pct})
            
            row = (f"{t:>5.2f} {class_pct:>7.1f}% {vqc_pct:>7.1f}% "
                   f"{m['accuracy']*100:>7.2f}% {m['balanced_accuracy']*100:>7.2f}% "
                   f"{m['macro_f1']:>8.4f} {m['mcc']:>8.4f}")
            logger.info(row)
        
        # Select best τ by macro-F1
        best = max(sweep_results, key=lambda x: x['macro_f1'])
        tau = best['tau']
        logger.info(f"\nBest τ = {tau} (Macro-F1 = {best['macro_f1']:.4f} on val)")
    
    # 7. Final evaluation on TEST set
    logger.info("\n" + "=" * 60)
    logger.info(f"FINAL TEST EVALUATION (τ = {tau})")
    logger.info("=" * 60)
    
    infer_start = time.perf_counter()
    y_pred_test = _hybrid_predict(
        router_model, vqc_model, q_pca, X_test, tau,
        routing_strategy, device, n_classes
    )
    infer_time = time.perf_counter() - infer_start
    
    # Routing statistics for test
    test_conf_mask, test_class_preds, test_scores = get_routing_mask(
        router_model, X_test, tau, strategy=routing_strategy
    )
    vqc_count = (~test_conf_mask).sum()
    
    logger.info(f"\n--- TEST SET ROUTING (τ={tau}) ---")
    logger.info(f"Total:     {len(X_test)}")
    logger.info(f"Classical: {test_conf_mask.sum()} ({test_conf_mask.mean()*100:.1f}%)")
    logger.info(f"VQC:       {vqc_count} ({vqc_count/len(X_test)*100:.1f}%)")
    
    # Classical-only accuracy on test (for comparison)
    classical_only_preds = router_model.predict(X_test)
    classical_only_acc = accuracy_score(y_test, classical_only_preds)
    logger.info(f"\nClassical-only accuracy (no VQC): {classical_only_acc*100:.2f}%")
    
    # VQC-only accuracy on routed samples
    if vqc_count > 0:
        vqc_indices = np.where(~test_conf_mask)[0]
        vqc_only_acc = accuracy_score(y_test[vqc_indices], y_pred_test[vqc_indices])
        logger.info(f"VQC accuracy (on routed samples): {vqc_only_acc*100:.2f}%")
    
    # Full hybrid metrics
    model_label = f"Hybrid_{router_name}_tau{tau}"
    log_detailed_metrics(y_test, y_pred_test, model_label, 'test', task=task)
    
    logger.info(f"Training time:  {train_time:.2f}s")
    logger.info(f"Inference time: {infer_time:.4f}s")
    
    # Save the VQC model
    vqc_save_path = os.path.join(PROJECT_ROOT, 'models', 'vqc', 'binary') if task == 'binary' else os.path.join(PROJECT_ROOT, 'models', 'vqc', 'multi')
    os.makedirs(vqc_save_path, exist_ok=True)
    os.makedirs(vqc_save_path, exist_ok=True)
    torch.save(vqc_model.state_dict(), os.path.join(vqc_save_path, f'vqc_{router_name}_tau{tau}.pt'))
    if q_pca is not None:
        joblib.dump(q_pca, os.path.join(vqc_save_path, f'vqc_pca_{router_name}.pkl'))
    logger.info(f"VQC model saved to {vqc_save_path}/")


def _hybrid_predict(router_model, vqc_model, q_pca, X, tau, strategy, device, n_classes):
    """Generate hybrid predictions: classical for confident, VQC for uncertain."""
    confident_mask, classical_preds, _ = get_routing_mask(router_model, X, tau, strategy=strategy)
    
    y_pred = classical_preds.copy()
    
    ambiguous_indices = np.where(~confident_mask)[0]
    if len(ambiguous_indices) > 0:
        X_ambiguous = X[ambiguous_indices]
        
        # Apply VQC PCA if needed
        if q_pca is not None:
            X_ambiguous_q = q_pca.transform(X_ambiguous)
        else:
            X_ambiguous_q = X_ambiguous
        
        # VQC inference
        vqc_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_ambiguous_q, dtype=torch.float32).to(device)
            vqc_logits = vqc_model(X_tensor)
            vqc_preds = torch.argmax(vqc_logits, dim=1).cpu().numpy()
        
        y_pred[ambiguous_indices] = vqc_preds
    
    return np.array(y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid SVM + VQC Pipeline')
    parser.add_argument('--router', type=str, default='rbf_svm',
                        choices=['rbf_svm', 'logistic_regression', 'linear_svm', 'random_forest'],
                        help='Classical router model for uncertainty estimation')
    parser.add_argument('--tau', type=float, default=0.80,
                        help='Confidence threshold τ (probability for rbf_svm/logreg, '
                             'margin for linear_svm)')
    parser.add_argument('--sweep_tau', action='store_true',
                        help='Sweep τ values on validation set to find optimal')
    parser.add_argument('--routing_strategy', type=str, default='probability',
                        choices=['probability', 'entropy'],
                        help='Routing strategy: probability (p_max > τ) or entropy (H(x) < τ)')
    parser.add_argument('--n_qubits', type=int, default=8,
                        help='Number of qubits for the VQC')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of StronglyEntanglingLayers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max training epochs for VQC')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Mini-batch size for VQC training')
    parser.add_argument('--task', type=str, default='multi', choices=['multi', 'binary'],
                        help='Classification task: multi (5-class) or binary (Normal vs Abnormal)')
    args = parser.parse_args()
    
    run_hybrid_pipeline(
        router_name=args.router,
        tau=args.tau,
        sweep_tau=args.sweep_tau,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        routing_strategy=args.routing_strategy,
        epochs=args.epochs,
        batch_size=args.batch_size,
        task=args.task
    )