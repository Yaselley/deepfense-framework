# Backend Components

Backends take the features extracted by the Frontend and map them to a fixed-dimensional embedding vector.

## Available Backends

### 1. AASIST (`AASIST`)
A Graph Attention Network (GAT) based architecture designed for ASV spoofing.
*   **Config Type**: `AASIST`
*   **Arguments**:
    *   `filts`: Filter configuration.
    *   `gat_dims`: Graph attention dimensions.

### 2. ECAPA-TDNN (`ECAPA_TDNN`)
**State-of-the-Art** backend for speaker verification, adapted for Deepfake Detection. Features channel attention (SE-Blocks) and multi-scale feature aggregation.
*   **Config Type**: `ECAPA_TDNN`
*   **Arguments**:
    *   `channels` (int): Number of channels in Res2Net blocks (default: 512).
    *   `emb_dim` (int): Output embedding dimension (default: 192).

### 3. RawNet2 (`RawNet2`)
A classic CNN-GRU architecture for ASV spoofing.
*   **Config Type**: `RawNet2`
*   **Arguments**:
    *   `filts` (list): Channels for each residual block.
    *   `gru_node` (int): GRU hidden size.
    *   `emb_dim` (int): Output dimension.

### 4. MLP (`MLP`)
A simple Multi-Layer Perceptron with configurable pooling. Good for SSL frontends (Wav2Vec2, WavLM) that already output high-level features.
*   **Config Type**: `MLP`
*   **Arguments**:
    *   `input_dim` (int): Dimension of input features.
    *   `projection` (list[int]): List of hidden layer sizes (e.g., `[128, 64]`).
    *   `pooling_type` (str): Pooling method (`mean`, `max`, `asp` (Attentive Statistics Pooling)).

### 5. Res2Net (`Nes2Net`)
A Res2Net-based convolutional architecture.
*   **Config Type**: `Nes2Net`
*   **Arguments**:
    *   `strides`: Stride settings for layers.
    *   `filts`: Channel counts for layers.

## Input/Output
*   **Input**: Features from Frontend `[B, T, C]`.
*   **Output**: Embedding vector `[B, Embedding_Dim]`.

---

> **Next Step**: [Loss Functions →](losses.md)
