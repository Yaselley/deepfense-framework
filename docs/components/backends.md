# Backend Components

Backends take the features extracted by the Frontend and map them to a fixed-dimensional embedding vector.

## Available Backends

### 1. AASIST (`AASIST`)
A Graph Attention Network (GAT) based architecture designed for ASV spoofing.

**Configuration Signature:**
```yaml
backend:
  type: AASIST
  args:
    filts: list
    gat_dims: list
```

**Parameters:**

* **filts** - (*list*) Filter configuration.
* **gat_dims** - (*list*) Graph attention dimensions.

**Example:**
```yaml
backend:
  type: AASIST
  args:
    filts: [64, 128]
    gat_dims: [64, 32]
```

---

### 2. ECAPA-TDNN (`ECAPA_TDNN`)
**State-of-the-Art** backend for speaker verification, adapted for Deepfake Detection. Features channel attention (SE-Blocks) and multi-scale feature aggregation.

**Configuration Signature:**
```yaml
backend:
  type: ECAPA_TDNN
  args:
    channels: int
    emb_dim: int
```

**Parameters:**

* **channels** - (*int*) Number of channels in Res2Net blocks (default: 512).
* **emb_dim** - (*int*) Output embedding dimension (default: 192).

**Example:**
```yaml
backend:
  type: ECAPA_TDNN
  args:
    channels: 512
    emb_dim: 192
```

---

### 3. RawNet2 (`RawNet2`)
A classic CNN-GRU architecture for ASV spoofing.

**Configuration Signature:**
```yaml
backend:
  type: RawNet2
  args:
    filts: list
    gru_node: int
    emb_dim: int
```

**Parameters:**

* **filts** - (*list*) Channels for each residual block.
* **gru_node** - (*int*) GRU hidden size.
* **emb_dim** - (*int*) Output dimension.

**Example:**
```yaml
backend:
  type: RawNet2
  args:
    filts: [128, 256, 512]
    gru_node: 1024
    emb_dim: 1024
```

---

### 4. MLP (`MLP`)
A simple Multi-Layer Perceptron with configurable pooling. Good for SSL frontends (Wav2Vec2, WavLM) that already output high-level features.

**Configuration Signature:**
```yaml
backend:
  type: MLP
  args:
    input_dim: int
    projection: list[int]
    pooling_type: string
```

**Parameters:**

* **input_dim** - (*int*) Dimension of input features.
* **projection** - (*list[int]*) List of hidden layer sizes (e.g., `[128, 64]`).
* **pooling_type** - (*str*) Pooling method (`mean`, `max`, `asp` (Attentive Statistics Pooling)).

**Example:**
```yaml
backend:
  type: MLP
  args:
    input_dim: 768
    projection: [512, 128]
    pooling_type: asp
```

---

### 5. Res2Net (`Nes2Net`)
A Res2Net-based convolutional architecture.

**Configuration Signature:**
```yaml
backend:
  type: Nes2Net
  args:
    strides: list
    filts: list
```

**Parameters:**

* **strides** - (*list*) Stride settings for layers.
* **filts** - (*list*) Channel counts for layers.

**Example:**
```yaml
backend:
  type: Nes2Net
  args:
    strides: [1, 2, 2]
    filts: [64, 128, 256]
```

## Input/Output
*   **Input**: Features from Frontend `[B, T, C]`.
*   **Output**: Embedding vector `[B, Embedding_Dim]`.

---

> **Next Step**: [Loss Functions →](losses.md)
