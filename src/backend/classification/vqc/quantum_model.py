"""
Variational Quantum Circuit (VQC) Model.

Implements a hybrid classical-quantum neural network using PennyLane.
Features are angle-embedded into qubits, processed through parameterized
entangling layers, measured, and mapped to class logits via a classical layer.
"""
import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler


class VQC(nn.Module):
    def __init__(self, n_qubits, n_layers, n_classes, scale_to_pi=True):
        """
        Args:
            n_qubits: Number of qubits (must match input feature dimension)
            n_layers: Number of StronglyEntanglingLayers
            n_classes: Number of output classes
            scale_to_pi: If True, scale inputs to [0, π] before angle embedding
        """
        super(VQC, self).__init__()
        self.n_qubits = n_qubits
        self.scale_to_pi = scale_to_pi
        
        # Attempt to load the GPU-accelerated quantum simulator
        try:
            self.dev = qml.device("lightning.gpu", wires=n_qubits)
        except Exception:
            # Fallback to CPU simulator if lightning.gpu fails to load
            self.dev = qml.device("default.qubit", wires=n_qubits)
            
        # --- Define the Quantum Node ---
        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            # 1. Feature Encoding (Equation 13): Angle Embedding
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Z')
                
            # 2. Parameterized Quantum Circuit (Equation 14)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # 3. Measurement (Equation 15): Expectation values of Pauli-Z
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Bridge the Quantum circuit into PyTorch
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        
        # Classical fully-connected layer
        self.clayer = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        if self.scale_to_pi:
            # Scale inputs to [0, π] for optimal angle embedding
            # This is done per-batch using min-max of the batch
            x_min = x.min(dim=0, keepdim=True)[0]
            x_max = x.max(dim=0, keepdim=True)[0]
            denom = x_max - x_min
            denom = torch.clamp(denom, min=1e-8)  # Prevent division by zero
            x = (x - x_min) / denom * np.pi
        
        q_out = self.qlayer(x)
        logits = self.clayer(q_out)
        return logits