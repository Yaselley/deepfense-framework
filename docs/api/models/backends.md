# Backends

Backends consume features and produce a fixed-size embedding.

## AASIST
`deepfense.models.backends.aasist.AASIST`

Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks. A complex Heterogeneous GAT architecture.

**Config Args**:
*   `filts` (List): Filter configuration for RawNet encoder.
*   `gat_dims` (List): Dimensions for Graph Attention layers.
*   `pool_ratios` (List): Pooling ratios.
*   `temperatures` (List): Softmax temperatures for attention.

## MLP
`deepfense.models.backends.mlp.MLPBackend`

Simple Multi-Layer Perceptron.

**Config Args**:
*   `input_dim` (int): Input feature dimension.
*   `hidden_dims` (List[int]): List of hidden layer sizes.
*   `output_dim` (int): Embedding size.

## Nes2Net
`deepfense.models.backends.nes2net.Nes2Net`

A specialized network architecture for spoofing detection.

## ResNet (RawNet/ECAPA)
*   `deepfense.models.backends.rawnet.RawNet`
*   `deepfense.models.backends.ecapa_tdnn.ECAPA_TDNN`

