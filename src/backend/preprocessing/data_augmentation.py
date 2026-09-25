"""
Multi-strategy data augmentation and class imbalance management module.

Provides four strategies for handling class imbalance in ECG heartbeat classification:
1. 'none': Returns features and labels unchanged with class_weights=None.
2. 'class_weight': Returns features and labels unchanged, computing balanced class weights.
3. 'moderate_smote': Applies capped SMOTE and RandomOverSampler to prevent extreme
   synthetic sample generation while balancing minority classes moderately.
4. 'smote_plus_weight': Applies moderate SMOTE followed by balanced class weighting
   on the resampled distribution.
"""

from collections import Counter
import logging
from typing import Optional, Tuple

from imblearn.over_sampling import RandomOverSampler, SMOTE
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)


def _compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Computes balanced class weights using sklearn.utils.class_weight.compute_class_weight.

    Returns:
        numpy array indexed by class label.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    max_label = int(np.max(classes)) if len(classes) > 0 else 0
    class_weights = np.zeros(max_label + 1, dtype=np.float64)
    for cls, weight in zip(classes, weights):
        class_weights[int(cls)] = weight
    return class_weights


def _apply_moderate_smote(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies capped SMOTE and RandomOverSampler to address class imbalance moderately.

    Rules:
    - For each minority class, target count is min(majority_count * 0.3, minority_count * 5).
    - If a class has fewer than 10 real samples, exclude it from SMOTE and handle it via
      RandomOverSampler with a cap of 50 samples total.
    - Dynamically set k_neighbors = min(5, min_smote_class_size - 1) where min_smote_class_size
      is the smallest class being SMOTEd (>= 10 samples).
    """
    class_counts = Counter(y_train)
    if len(class_counts) <= 1:
        logger.warning("Fewer than 2 classes present; skipping resampling.")
        return np.asarray(X_train), np.asarray(y_train)

    majority_count = max(class_counts.values())
    smote_strategy = {}
    ros_strategy = {}

    for cls, count in class_counts.items():
        if count >= majority_count:
            continue

        target_count = int(round(min(majority_count * 0.3, count * 5)))
        if target_count <= count:
            continue

        if count < 10:
            ros_target = min(50, target_count)
            if ros_target > count:
                ros_strategy[cls] = ros_target
        else:
            smote_strategy[cls] = target_count

    X_resampled, y_resampled = X_train, y_train

    if smote_strategy:
        min_smote_class_size = min(class_counts[cls] for cls in smote_strategy)
        k_neighbors = max(1, min(5, min_smote_class_size - 1))
        logger.info(
            f"Applying SMOTE with k_neighbors={k_neighbors} "
            f"(min_smote_class_size={min_smote_class_size}) for classes: {smote_strategy}"
        )
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=k_neighbors, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_resampled, y_resampled)

    if ros_strategy:
        logger.info(
            f"Applying RandomOverSampler (<10 samples, capped at 50) for classes: {ros_strategy}"
        )
        ros = RandomOverSampler(sampling_strategy=ros_strategy, random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_resampled, y_resampled)

    return np.asarray(X_resampled), np.asarray(y_resampled)


def augment_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = 'class_weight'
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Manages data imbalance using the specified strategy.

    Args:
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        strategy: One of 'none', 'class_weight', 'moderate_smote', 'smote_plus_weight'

    Returns:
        X_augmented: Augmented (or unchanged) training features
        y_augmented: Augmented (or unchanged) training labels
        class_weights: numpy array of per-class weights (or None if not applicable)
    """
    valid_strategies = {'none', 'class_weight', 'moderate_smote', 'smote_plus_weight'}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown augmentation strategy: '{strategy}'. "
            f"Supported strategies: {sorted(valid_strategies)}"
        )

    X_arr = np.asarray(X_train)
    y_arr = np.asarray(y_train)

    logger.info(f"Data augmentation strategy: '{strategy}'")
    logger.info(f"Class distribution before augmentation: {dict(Counter(y_arr))}")

    if strategy == 'none':
        X_aug, y_aug, class_weights = X_arr, y_arr, None
    elif strategy == 'class_weight':
        class_weights = _compute_class_weights(y_arr)
        X_aug, y_aug = X_arr, y_arr
    elif strategy == 'moderate_smote':
        X_aug, y_aug = _apply_moderate_smote(X_arr, y_arr)
        class_weights = None
    elif strategy == 'smote_plus_weight':
        X_aug, y_aug = _apply_moderate_smote(X_arr, y_arr)
        class_weights = _compute_class_weights(y_aug)
    else:
        raise ValueError(f"Unhandled strategy: '{strategy}'")

    logger.info(f"Class distribution after augmentation: {dict(Counter(y_aug))}")
    if class_weights is not None:
        logger.info(f"Class weights computed: {dict(enumerate(class_weights))}")

    return X_aug, y_aug, class_weights