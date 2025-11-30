# Backend Components

Backends take the features extracted by the Frontend and map them to a fixed-dimensional embedding vector.

## Available Backends

### 1. AASIST (`AASIST`)
A Graph Attention Network (GAT) based architecture designed for ASV spoofing.
*   **Config Type**: `AASIST`
*   **Arguments**:
    *   `filts`: Filter configuration (default: `[[1, 32], [32, 32], [32, 64], [64, 64]]`).
    *   `gat_dims`: Graph attention dimensions (default: `[64, 32]`).
    *   `pool_ratios`: Pooling ratios for graph pooling (default: `[0.5, 0.7, 0.5, 0.5]`).

### 2. MLP (`MLP`)
A simple Multi-Layer Perceptron with configurable pooling. Good for SSL frontends (Wav2Vec2, WavLM) that already output high-level features.
*   **Config Type**: `MLP`
*   **Arguments**:
    *   `input_dim` (int): Dimension of input features.
    *   `projection` (list[int]): List of hidden layer sizes (e.g., `[128, 64]`).
    *   `pooling_type` (str): Pooling method (`mean`, `max`, `asp` (Attentive Statistics Pooling)).

### 3. Res2Net (`Nes2Net` / `Res2Net`)
A Res2Net-based convolutional architecture.
*   **Config Type**: `Nes2Net`
*   **Arguments**:
    *   `strides`: Stride settings for layers.
    *   `filts`: Channel counts for layers.

## Input/Output
*   **Input**: Features from Frontend.
*   **Output**: Embedding vector `(Batch, Embedding_Dim)`.
