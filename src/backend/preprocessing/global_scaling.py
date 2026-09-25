"""Global feature scaling module for ECG morphological features."""

import logging
import os
import joblib
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def apply_global_scaling(X_train, X_val, X_test, output_dir=None):
    """
    Applies Global Standard Scaling across all 396 dimensions.
    Fitted ONLY on the training data to prevent data leakage.
    """
    logger.info("Fitting StandardScaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    logger.info("Transforming validation and test sets using fitted scaler...")
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        scaler_path = os.path.join(output_dir, "standard_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        logger.info(f"Saved fitted StandardScaler to {scaler_path}")

    return X_train_scaled, X_val_scaled, X_test_scaled