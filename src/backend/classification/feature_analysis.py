"""
Diagnostic script to investigate feature-space separability for an ECG arrhythmia classification project.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(BASE_DIR, 'evaluation_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

AAMI_CLASSES = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

def load_data():
    feature_dir = os.path.abspath(os.path.join(BASE_DIR, '../pilae/extracted_features'))
    
    Z_train = np.load(os.path.join(feature_dir, 'Z_train_reduced.npy'))
    y_train = np.load(os.path.join(feature_dir, 'y_train.npy'))
    
    Z_val = np.load(os.path.join(feature_dir, 'Z_val_reduced.npy'))
    y_val = np.load(os.path.join(feature_dir, 'y_val.npy'))
    
    Z_test = np.load(os.path.join(feature_dir, 'Z_test_reduced.npy'))
    y_test = np.load(os.path.join(feature_dir, 'y_test.npy'))
    
    # Combine train and val for analysis
    Z_train_val = np.vstack((Z_train, Z_val))
    y_train_val = np.concatenate((y_train, y_val))
    
    return Z_train_val, y_train_val, Z_test, y_test

def plot_tsne(Z, y):
    logging.info("Running t-SNE visualization...")
    # Subsample to max ~5000 points
    n_samples = Z.shape[0]
    if n_samples > 5000:
        indices = np.random.choice(n_samples, 5000, replace=False)
        Z_sub = Z[indices]
        y_sub = y[indices]
    else:
        Z_sub = Z
        y_sub = y
        
    tsne = TSNE(n_components=2, random_state=42)
    Z_tsne = tsne.fit_transform(Z_sub)
    
    plt.figure(figsize=(10, 8))
    for class_id, class_name in AAMI_CLASSES.items():
        idx = np.where(y_sub == class_id)[0]
        plt.scatter(Z_tsne[idx, 0], Z_tsne[idx, 1], label=class_name, alpha=0.6, s=10)
        
    plt.title('t-SNE of PILAE+PCA Features')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(PLOT_DIR, 'tsne_visualization.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("t-SNE visualization saved.")

def inter_class_distance(Z, y):
    logging.info("Computing Inter-class distance matrix...")
    centroids = {}
    for class_id in AAMI_CLASSES:
        idx = np.where(y == class_id)[0]
        if len(idx) > 0:
            centroids[class_id] = np.mean(Z[idx], axis=0)
        else:
            centroids[class_id] = np.zeros(Z.shape[1])
            
    num_classes = len(AAMI_CLASSES)
    dist_matrix = np.zeros((num_classes, num_classes))
    
    for i in range(num_classes):
        for j in range(num_classes):
            dist_matrix[i, j] = np.linalg.norm(centroids[i] - centroids[j])
            
    logging.info("Inter-class distance matrix:")
    header = "    " + " ".join([f"{AAMI_CLASSES[i]:>6}" for i in range(num_classes)])
    logging.info(header)
    for i in range(num_classes):
        row_str = f"{AAMI_CLASSES[i]:<4}" + " ".join([f"{dist_matrix[i, j]:>6.2f}" for j in range(num_classes)])
        logging.info(row_str)
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(dist_matrix, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=[AAMI_CLASSES[i] for i in range(num_classes)],
                yticklabels=[AAMI_CLASSES[i] for i in range(num_classes)])
    plt.title('Inter-Class Euclidean Distance Matrix')
    plt.savefig(os.path.join(PLOT_DIR, 'inter_class_distances.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Inter-class distance matrix heatmap saved.")

def fisher_ratio(Z, y):
    logging.info("Computing Fisher's Linear Discriminant Ratios...")
    pairs = [(0, 1), (1, 2), (2, 3), (2, 4), (0, 3)]
    pair_names = [("N", "S"), ("S", "V"), ("V", "F"), ("V", "Q"), ("N", "F")]
    
    for (c1, c2), (n1, n2) in zip(pairs, pair_names):
        idx1 = np.where(y == c1)[0]
        idx2 = np.where(y == c2)[0]
        
        if len(idx1) == 0 or len(idx2) == 0:
            logging.info(f"Fisher Ratio ({n1}, {n2}): Not enough data")
            continue
            
        mu1 = np.mean(Z[idx1], axis=0)
        mu2 = np.mean(Z[idx2], axis=0)
        var1 = np.var(Z[idx1], axis=0)
        var2 = np.var(Z[idx2], axis=0)
        
        # (mu1 - mu2)^2 / (sigma1^2 + sigma2^2)
        # add epsilon to avoid division by zero
        epsilon = 1e-10
        f_ratio = ((mu1 - mu2)**2) / (var1 + var2 + epsilon)
        avg_f_ratio = np.mean(f_ratio)
        
        logging.info(f"Fisher Ratio ({n1}, {n2}): {avg_f_ratio:.4f}")

def bhattacharyya_distance(mu1, cov1, mu2, cov2):
    # Bhattacharyya distance for multivariate normal distributions
    try:
        cov = (cov1 + cov2) / 2
        inv_cov = np.linalg.inv(cov)
        diff = mu1 - mu2
        
        term1 = 0.125 * np.dot(np.dot(diff.T, inv_cov), diff)
        
        det1 = np.linalg.det(cov1)
        det2 = np.linalg.det(cov2)
        det = np.linalg.det(cov)
        
        if det1 <= 0 or det2 <= 0 or det <= 0:
            return np.inf
            
        term2 = 0.5 * np.log(det / np.sqrt(det1 * det2))
        return term1 + term2
    except np.linalg.LinAlgError:
        return np.inf

def q_v_deep_dive(Z, y):
    logging.info("Running Q vs V deep-dive...")
    idx_q = np.where(y == 4)[0]
    idx_v = np.where(y == 2)[0]
    
    if len(idx_q) == 0 or len(idx_v) == 0:
        logging.warning("Not enough data for Q or V classes.")
        return
        
    Z_q = Z[idx_q]
    Z_v = Z[idx_v]
    
    # Combine and run PCA top-2
    Z_qv = np.vstack((Z_q, Z_v))
    pca = PCA(n_components=2)
    Z_qv_pca = pca.fit_transform(Z_qv)
    
    Z_q_pca = Z_qv_pca[:len(idx_q)]
    Z_v_pca = Z_qv_pca[len(idx_q):]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(Z_q_pca[:, 0], Z_q_pca[:, 1], label='Q', alpha=0.6, s=15, color='purple')
    plt.scatter(Z_v_pca[:, 0], Z_v_pca[:, 1], label='V', alpha=0.6, s=15, color='orange')
    plt.title('Q vs V in Top-2 PCA Components')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(PLOT_DIR, 'q_vs_v_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logging.info("Q vs V PCA plot saved.")
    
    # Bhattacharyya distance
    mu_q = np.mean(Z_q, axis=0)
    mu_v = np.mean(Z_v, axis=0)
    cov_q = np.cov(Z_q, rowvar=False)
    cov_v = np.cov(Z_v, rowvar=False)
    
    b_dist = bhattacharyya_distance(mu_q, cov_q, mu_v, cov_v)
    logging.info(f"Bhattacharyya distance (Q, V): {b_dist:.4f}")

def main():
    logging.info("Starting feature analysis...")
    
    try:
        Z_train_val, y_train_val, Z_test, y_test = load_data()
        logging.info(f"Loaded train+val data: {Z_train_val.shape}, {y_train_val.shape}")
    except FileNotFoundError as e:
        logging.error(f"Could not load data: {e}")
        return
        
    plot_tsne(Z_train_val, y_train_val)
    inter_class_distance(Z_train_val, y_train_val)
    fisher_ratio(Z_train_val, y_train_val)
    q_v_deep_dive(Z_train_val, y_train_val)
    
    logging.info("Feature analysis complete.")

if __name__ == "__main__":
    main()
