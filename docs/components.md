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

### **Wav2Vec2** (`type: "wav2vec2"`)
Wrapper for Wav2Vec2-based models (e.g., XLSR).

*   **Source**: `deepfense/models/frontends/wav2vec2.py`
*   **Config Arguments**:
    *   `ckpt_path` (str): Path to `.pt` checkpoint.
    *   `freeze` (bool): Whether to freeze the frontend weights.

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
*   **Output**: Embedding vector.

### **MLP** (`type: "MLP"`)
A flexible Multi-Layer Perceptron backend with configurable pooling and normalization.

*   **Source**: `deepfense/models/backends/mlp.py`
*   **Config Arguments**:
    *   `input_dim` (int): Dimension of input features (automatically handled if base class used).
    *   `projection` (list[int]): List of hidden layer dimensions (e.g., `[512, 256]`).
    *   `activation` (str): Activation function (`relu`, `selu`, `tanh`, `sigmoid`).
    *   `norm_type` (str): Normalization type (`batch`, `layer`, or `none`).
    *   `pooling_type` (str): Pooling strategy (`mean`, `tap`, `stats`, `asp`, `mha`).
    *   `output_dim` (int, optional): Final output dimension. If specified, adds a final linear projection.

### **Nes2Net** (`type: "Nes2Net"`)
Backend using Nested Res2Net blocks.

*   **Source**: `deepfense/models/backends/nes2net.py`
*   **Config Arguments**:
    *   `nes_ratio` (list[int]): Ratios for nested blocks (e.g., `[8, 8]`).
    *   `dilation` (int): Dilation factor.
    *   `se_ratio` (int): Squeeze-Excitation ratio.
    *   `pooling_type` (str): Pooling strategy.

### **TCM** (`type: "TCM"`)
Transformer-Coupled Module (Conformer-based) backend.

*   **Source**: `deepfense/models/backends/tcm.py`
*   **Config Arguments**:
    *   `emb_size` (int): Embedding size.
    *   `heads` (int): Number of attention heads.
    *   `num_encoders` (int): Number of Conformer blocks.

---

## 3. Transforms (Augmentations)

Augmentations are applied to the raw waveform during data loading.

### **AugmentationPipeline** (`type: "augmentation_pipeline"`)
A meta-transform that manages a list of other transforms with advanced selection and execution strategies.

*   **Source**: `deepfense/data/transforms/augmentations.py`
*   **Config Arguments**:
    *   `mode` (str): **Selection Strategy**.
        *   `"sequential"`: Selects `k` (or all) transforms.
        *   `"parallel"`: Selects exactly 1 transform (`OneOf`).
    *   `k` (int, optional): Number of transforms to select if mode is "sequential". If `None`, selects all.
    *   `execution` (str): **Application Strategy**.
        *   `"chain"`: Applies selected transforms in sequence to the same audio (`x -> t1 -> t2`).
        *   `"independent"`: Applies selected transforms separately (`[t1(x), t2(x)]`).
    *   `concat_original` (bool): If `True`, includes the original signal in the output stack (`[Original, Aug1, Aug2...]`). Useful for training on both clean and augmented versions simultaneously.
    *   `p` (float): Probability of applying the pipeline itself.
    *   `transforms` (list): List of transform configurations to manage.

### **Standard Transforms**
*   **`rawboost`**: Applies RawBoost algorithms.
*   **`rir`**: Convolves with Room Impulse Responses (requires CSV of RIR paths).
*   **`add_noise`**: Adds additive noise (requires CSV of noise paths).
*   **`speed_perturb`**: Resamples audio to change speed/pitch.
*   **`do_clip`**: Clips signal amplitude.
*   **`pad`**: Pads or crops signal to fixed length.

---

## 4. Unified Losses

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

## 5. Metrics & Visualization

### **Metrics**
Metrics are calculated using the `Evaluator` class.

*   **EER**: Equal Error Rate.
*   **minDCF**: Minimum Detection Cost Function (requires `Pspoof`, `Cmiss`, `Cfa` in config).
*   **actDCF**: Actual Detection Cost Function.
*   **F1_SCORE**: Macro/Micro F1 score.
*   **ACC**: Accuracy.

### **Visualization** (`deepfense/utils/visualization.py`)
The visualization module provides simple, effective trend plotting for all tracked metrics.
*   **Automatic**: No extra config flags needed; just list metrics in the `metrics` section.
*   **Plots**: Generates `trend_{MetricName}.png` (e.g., `trend_EER.png`) in the experiment output folder.
*   **Comparison**: Automatically overlays "Train" and "Val" lines where applicable (e.g., Loss).
