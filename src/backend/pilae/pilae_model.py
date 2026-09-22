import numpy as np
import logging

logger = logging.getLogger(__name__)

class PILAE:
    def __init__(self, hidden_dim):
        """
        Pseudoinverse Linear AutoEncoder (PILAE)
        :param hidden_dim: Number of hidden neurons (p) for the bottleneck
        """
        self.hidden_dim = hidden_dim
        self.W_enc = None
        self.W_dec = None
        
    def fit(self, X):
        d = X.shape[1]
        
        # 1. Initialize encoder weights randomly and scale down appropriately
        # W_enc ~ N(0, 1/sqrt(d)) - prevents tanh saturation
        logger.info(f"Initializing encoder weights with scaling factor 1/sqrt({d})")
        np.random.seed(42) # For reproducibility
        self.W_enc = np.random.normal(0, 1/np.sqrt(d), (d, self.hidden_dim))
        
        # 2. Compute hidden state representation: H = tanh(X * W_enc)
        logger.info("Computing hidden state representation (H)...")
        H = np.tanh(np.dot(X, self.W_enc))
        
        # 3. Calculate Decoder weights analytically to minimize reconstruction error
        # W_dec = H^+ * X (where H^+ is the Moore-Penrose pseudoinverse)
        logger.info("Computing pseudoinverse (H^+) to analytically solve for decoder weights...")
        H_pinv = np.linalg.pinv(H)
        self.W_dec = np.dot(H_pinv, X)
        
        # 4. Calculate and log the reconstruction error
        X_reconstructed = np.dot(H, self.W_dec)
        reconstruction_error = np.mean(np.square(X - X_reconstructed))
        logger.info(f"PILAE Training Complete. Mean Squared Reconstruction Error: {reconstruction_error:.6f}")
        
    def transform(self, X):
        """
        Transforms high-dimensional ECG data into the compressed PILAE feature space.
        """
        return np.tanh(np.dot(X, self.W_enc))