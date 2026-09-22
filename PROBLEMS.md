ValueError: Expected 2D array, got 1D array instead:
array=[].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

Fix: data directory isn't properly defined in the config.py
____
ValueError: Expected n_neighbors <= n_samples_fit, but n_neighbors = 6, n_samples_fit = 4, n_samples = 4

Fix: SMOTE: it uses a default of k_neighbors=5 to generate synthetic data. However, in the MIT-BIH dataset, the "Unknown" (Q) class is extremely rare for the patients in DS1_TRAIN. In fact, it only has exactly 4 samples. Because 4 is less than 6 (5 neighbors + 1 point), SMOTE crashes. To fix this, we need to dynamically calculate k_neighbors based on the smallest class size, and safely fall back to RandomOverSampler if a class is just too small.

Because we added the dynamic check, SMOTE successfully used k_neighbors=3 for the "Unknown" class. It synthesized morphological twins and balanced every single class up to the majority number (45,402).
____
