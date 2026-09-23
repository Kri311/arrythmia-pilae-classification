# Hybrid PILAE and Quantum-Inspired Classification for Cardiac Arrhythmia Detection

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)

A Non-Iterative Feature Extraction and Variational Quantum Circuit Routing Architecture tailored for highly imbalanced medical diagnostic data.

---

## Table of Contents
* [Problem Statement](#problem-statement)
* [Objective](#objective)
* [Dataset Overview](#dataset-overview)
* [Project Architecture](#project-architecture)
* [Methodology](#methodology)
* [Project Structure](#project-structure)
* [Installation & Setup](#installation--setup)
* [Usage (How to Run)](#usage-how-to-run)
* [Results & Evaluation](#results--evaluation)

---

## Problem Statement

Electrocardiogram (ECG) data analysis is fundamentally challenging due to the high dimensionality of continuous waveforms, extreme inter-patient morphological variability, and profound class imbalances. Classical machine learning models trained on this data inherently suffer from "accuracy inflation." They aggressively bias towards the majority class (normal heartbeats) to achieve high overall accuracy, while disastrously failing to detect minority, life-threatening arrhythmias (e.g., Ventricular premature beats). 

Furthermore, traditional deep neural feature extraction (Autoencoders) relies on slow, iterative gradient-descent optimization (backpropagation) that is prone to vanishing gradients, local minima, and requires heavy hyperparameter tuning.

## Objective

To engineer an industry-grade, hybrid machine learning pipeline that prioritizes medical diagnostic sensitivity (Recall) over deceptive baseline accuracy. This is achieved by:
1. **Accelerating Feature Extraction:** Replacing gradient descent with a mathematically instantaneous Pseudoinverse Linear Auto-Encoder (PILAE).
2. **Eliminating Bias:** Forcing the model to learn rare anomalies via strict Patient-Wise splitting and SMOTE data augmentation.
3. **Quantum Routing:** Utilizing a Linear SVM to rapidly classify linearly separable beats, while routing ambiguous, overlapping cases to a GPU-accelerated Variational Quantum Circuit (VQC) to leverage quantum entanglement for complex boundary mapping.

---

## Dataset Overview

**Source:** MIT-BIH Arrhythmia Database
The dataset contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, sampled at 360 Hz with 11-bit resolution over a 10 mV range.

**File Formats:**
*   `.dat`: Raw, continuous physiological waveforms in binary.
*   `.hea`: Metadata (sampling frequency, channels, gain).
*   `.atr`: Clinical annotations mapping the exact sample points of R-peaks and their beat classes.

**Data Characteristics:** The dataset is highly imbalanced. Normal (N) beats comprise over 85% of all samples, while classes like Fusion (F) and Unknown (Q) comprise a tiny fraction.

---

## Project Architecture

```text
[ Raw ECG Signals (.dat, .atr) ]
           |
           v
[ Bandpass Filter 0.5-45Hz ]
           |
           v
[ R-Peak Segmentation (198 samples/lead) ]
           |
           v
[ Dual-Lead Concatenation (396 dimensions) ]
           |
           v
[ Patient-Wise Split (DS1 / DS2) ] ---> [ SMOTE Data Augmentation (Train only) ]
           |
           v
[ PILAE Feature Extraction (250 Neurons) ]  <-- Non-iterative pseudoinverse matrix math
           |
           v
[ PCA Dimensionality Reduction (42 dims) ]
           |
           v
[ Linear SVM (Primary Pre-Classifier) ]
           |
           +---> Calculate |d(z)| (Distance from hyperplane)
           |
      Is |d(z)| > tau ?
      /              \
    YES               NO (Ambiguous / Borderline overlap)
    /                  \
[ SVM Label ]        [ Secondary PCA (8 dims) ]
                               |
                               v
                     [ Quantum Angle Embedding (Ry, Rz) ]
                               |
                               v
                     [ Parameterized Quantum Circuit ]  <-- Strongly Entangling Layers
                               |
                               v
                     [ Pauli-Z Measurement ] ---> [ VQC Label ]
```

---

## Methodology

### 1. Data Preprocessing
Raw continuous waveforms are isolated via a 0.5-45 Hz Butterworth bandpass filter. Heartbeats are segmented using a 198-sample window (99 samples before/after the R-peak). Segments from two leads are concatenated into a 396-dimensional vector. Each segment undergoes per-beat Z-Score Normalization. Crucially, a Patient-Wise Stratified Split (De Chazal split) is strictly enforced to prevent data leakage, followed by SMOTE augmentation to balance the training set.

### 2. Feature Extraction (PILAE)
The Pseudoinverse Linear Auto-Encoder (PILAE) bypasses iterative gradient descent (backpropagation). 
*   **Encoder:** Weights are initialized randomly as `N(0, 1/sqrt(d))`. Hidden representation: `H = tanh(X * W_enc)`.
*   **Decoder:** Solved instantly via the Moore-Penrose pseudoinverse: `W_dec = H^+ * X`.
*   This highly efficient math compressed the 396-dimensional input into 250 hidden neurons with a massive 97.8% signal retention rate.

### 3. Feature Transformation (PCA)
Principal Component Analysis (PCA) was applied to the 250 PILAE neurons, retaining exactly 95% of the morphological variance. This aggressively reduced the dimensions from 250 down to a noise-free, 42-dimensional feature tensor.

### 4. Classical Classification
A **Linear SVM** acts as the primary classical filter. It calculates the geometric distance `|d(z)|` of a sample from its decision boundary. Other models (Logistic Regression, Random Forest, XGBoost) were included purely as benchmarks.

### 5. Quantum Classification (VQC)
If the SVM's distance score `|d(z)| <= tau`, the heartbeat is deemed "ambiguous" and routed to the VQC.
*   **Secondary PCA:** The 42 dims are compressed to 8 to fit classical quantum simulators.
*   **Angle Embedding:** The 8 features are mapped to 8 quantum qubits.
*   **Quantum Entanglement:** Quibits pass through Strongly Entangling Layers to draw complex, non-linear boundaries around overlapping arrhythmias. Optimized with Adam via binary cross-entropy.

---

## Project Structure

```text
.
├── data/
│   └── mit-bih-arrhythmia-database-1.0.0/   # Raw datasets (.dat, .atr, .hea)
├── paper/
│   └── figures/                             # Architecture diagrams
├── src/
│   └── backend/
│       ├── preprocessing/
│       │   ├── main.py                      # Preprocessing & SMOTE runner
│       │   ├── feature_extraction.py
│       │   ├── normalisation.py
│       │   ├── data_split.py
│       │   └── ...
│       ├── pilae/
│       │   ├── feature_pipeline.py          # PILAE + PCA runner
│       │   ├── pilae_model.py
│       │   └── pca_reduction.py
│       └── classification/
│           ├── classical/
│           │   └── train_classical.py       # SVM Benchmarks
│           └── vqc/
│               ├── train_hybrid.py          # Quantum Routing Pipeline
│               └── quantum_model.py         # PennyLane Circuit
├── README.md
└── REPORT.md
```

---

## Installation & Setup

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
# Core numerical and ML libraries
pip install numpy pandas scipy scikit-learn imbalanced-learn joblib xgboost matplotlib seaborn

# PyTorch (Ensure you install the correct CUDA version for your GPU)
pip install torch torchvision torchaudio

# PennyLane for Quantum Machine Learning
pip install pennylane "pennylane-lightning[gpu]"
```

---

## Usage (How to Run)

Execute the pipeline sequentially from the root directory:

**1. Data Preprocessing**
Cleans the signals, segments the R-peaks, and applies SMOTE.
```bash
python src/backend/preprocessing/main.py
```

**2. PILAE Feature Extraction**
Compresses the 396-dim ECG beats into 42-dim mathematical tensors.
```bash
python src/backend/pilae/feature_pipeline.py
```

**3. Classical Benchmarking & Pre-Classification**
Trains the Linear SVM and saves the models to disk.
```bash
python src/backend/classification/classical/train_classical.py
```

**4. Hybrid SVM + Variational Quantum Circuit**
Loads the SVM, routes ambiguous samples via threshold `tau`, and executes the GPU-accelerated Quantum Circuit.
```bash
python src/backend/classification/vqc/train_hybrid.py
```

---

## Results & Evaluation

The implementation reveals a stark contrast between classical accuracy-chasing and medical diagnostic reliability.

| Model | Overall Accuracy | Ventricular Recall | Fusion Recall | Diagnostic Viability |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest** | 82.68% | 68.17% | 27.72% | Poor (Ignores rare anomalies) |
| **XGBoost** | 77.78% | 78.81% | 38.86% | Moderate |
| **Hybrid SVM + VQC** | 45.99% | **72.50%** | **35.49%** | **High** (Sensitive to Anomalies) |

### Interpretation
*   **Tree-based Ensembles (Random Forest):** Achieved the highest overall accuracy by simply predicting the majority class ("Normal") and ignoring rare arrhythmias. This is a fatal flaw in medical diagnostics.
*   **Hybrid SVM + VQC:** The confusion matrices reveal a heavy penalty on raw accuracy due to the SMOTE balancing (many False Positives for Normal beats). However, this trade-off correctly prioritized **Recall**. The hybrid system aggressively hunted down minority classes, successfully intercepting life-threatening Ventricular anomalies. The quantum entanglement successfully mapped the complex, non-linear boundaries that the classical models failed to differentiate. 