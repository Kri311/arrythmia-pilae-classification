"""
PCA Dimensionality Reduction Module.

Applies PCA to PILAE hidden representations to remove redundancy.
Fit on training data only to prevent data leakage.
"""
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


def apply_pca(Z_train, Z_val, Z_test, n_components=0.95):
    """
    Applies PCA to further reduce dimensionality and remove redundancy.
    
    Args:
        Z_train: PILAE-transformed training features
        Z_val: PILAE-transformed validation features
        Z_test: PILAE-transformed test features
        n_components: float (0-1) for variance retention, or int for fixed dims.
                      Default 0.95 retains 95% of morphological variance.
    
    Returns:
        Z_train_reduced, Z_val_reduced, Z_test_reduced, pca_model
    """
    if isinstance(n_components, float) and 0 < n_components < 1:
        logger.info(f"Applying PCA (retaining {n_components * 100}% of the variance)...")
    else:
        logger.info(f"Applying PCA (fixed {n_components} components)...")
    
    pca = PCA(n_components=n_components, random_state=42)
    
    # Fit on training data ONLY
    Z_train_reduced = pca.fit_transform(Z_train)
    Z_val_reduced = pca.transform(Z_val)
    Z_test_reduced = pca.transform(Z_test)
    
    logger.info(f"PCA complete. Reduced dimensions from {Z_train.shape[1]} down to {Z_train_reduced.shape[1]}.")
    logger.info(f"Explained variance ratio (cumulative): {pca.explained_variance_ratio_.sum():.4f}")
    
    return Z_train_reduced, Z_val_reduced, Z_test_reduced, pca