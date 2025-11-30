# DeepFense Component Reference

This document details the key components available in the DeepFense framework.

## 1. Frontends

Frontends are responsible for extracting features from raw audio. They take `(batch, time)` input and return `(batch, channels, time)` or similar feature maps.

### **WavLM** (`type: "wavlm"`)
A wrapper around the WavLM pre-trained model.

*   **Source**: `deepfense/models/frontends/wavlm.py`
*   **Config Arguments**:
    *   `ckpt_path` (str): Path to the pre-trained `.pt` checkpoint file.
*   **Input**: Raw audio waveform.
*   **Output**: Feature sequence.

---

## 2. Backends

Backends take features from the frontend and aggregate them into a fixed-size embedding vector.

### **AASIST** (`type: "AASIST"`)
The AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks) backend.

*   **Source**: `deepfense/models/backends/aasist.py`
*   **Config Arguments**:
    *   `filts` (list): Filter configuration for RawNet encoder.
    *   `gat_dims` (list): Dimensions for Graph Attention layers.
    *   `pool_ratios` (list): Pooling ratios.
    *   `temperatures` (list): Temperatures for Softmax in attention.
    *   `pool` (tuple, optional): Final pooling kernel size (default: `(1, 1)`).
*   **Output**: Embedding vector (dimension depends on `gat_dims` and concatenation logic, typically 160ish depending on config, but projected to 128 or similar inside).

---

## 3. Unified Losses

Losses in DeepFense now **combine** the projection/classification layer (formerly "Mapper") and the loss calculation.

### **CrossEntropy** (`type: "CrossEntropy"`)
Standard Cross Entropy loss with a linear projection layer.

*   **Source**: `deepfense/models/losses/cross_entropy.py`
*   **Config Arguments**:
    *   `embedding_dim` (int): Input dimension of the embeddings (must match backend output).
    *   `n_classes` (int): Number of classes (usually 2: spoof vs bonafide).
    *   `class_weights` (list[float], optional): Weights for handling class imbalance (e.g., `[0.1, 0.9]`).
    *   `reduction` (str, optional): 'mean' or 'sum' (default: 'mean').

### **AMSoftmax** (`type: "AMSoftmax"`)
Additive Margin Softmax loss for discriminative embedding learning.

*   **Source**: `deepfense/models/losses/am_softmax.py`
*   **Config Arguments**:
    *   `embedding_dim` (int): Input embedding dimension.
    *   `n_classes` (int): Number of classes.
    *   `s` (float): Scale factor (inverse temperature).
    *   `m` (float): Margin value.
    *   `class_weights` (list[float], optional): Weights for the underlying CrossEntropy loss.

### **OCSoftmax** (`type: "OCSoftmax"`)
One-Class Softmax loss, often used for generalized spoofing detection.

*   **Source**: `deepfense/models/losses/oc_softmax.py`
*   **Config Arguments**:
    *   `embedding_dim` (int): Input embedding dimension.
    *   `w_posi` (float): Weight center for positive class.
    *   `w_nega` (float): Weight center for negative class.
    *   `alpha` (float): Scaling factor.

---

## 4. Metrics

Metrics are calculated using the `Evaluator` class.

*   **EER**: Equal Error Rate.
*   **minDCF**: Minimum Detection Cost Function (requires `Pspoof`, `Cmiss`, `Cfa` in config).
*   **actDCF**: Actual Detection Cost Function.
*   **F1_SCORE**: Macro/Micro F1 score.
*   **ACC**: Accuracy.

