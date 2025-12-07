# Loss Functions

Loss modules in DeepFense are unified: they handle **projection**, **loss calculation**, and **score generation**.

## Available Losses

### 1. AM-Softmax (`AMSoftmax`)
**Additive Margin Softmax**. Forces classes to be separated by a margin `m` on the hypersphere.
*   **Arguments**:
    *   `embedding_dim` (int): Input feature dimension.
    *   `n_classes` (int): Number of classes (usually 2).
    *   `m` (float): Margin value (e.g., 0.3).
    *   `s` (float): Scaling factor (e.g., 30.0).
    *   `weight` (float): Loss weight in multi-loss setups.

### 2. A-Softmax (`ASoftmax`)
**Angular Softmax** (SphereFace). Uses multiplicative angular margin.
*   **Arguments**:
    *   `embedding_dim` (int): Input dimension.
    *   `n_classes` (int): Number of classes.
    *   `m` (int): Integer margin (e.g., 4). Controls the angular margin size.
    *   `gamma` (float): Focusing parameter.
    *   `lambda_min`, `lambda_max`: Parameters for lambda annealing (stabilizes training).

### 3. OC-Softmax (`OCSoftmax`)
**One-Class Softmax**. Designed specifically for spoofing. It compacts the "Bonafide" class into a tight cluster and pushes "Spoof" samples away.
*   **Arguments**:
    *   `embedding_dim` (int): Input dimension.
    *   `m_real` (float): Margin for real speech (default: 0.5).
    *   `m_fake` (float): Margin for fake speech (default: 0.2).
    *   `alpha` (float): Scaling factor (default: 20.0).

### 4. Cross Entropy (`CrossEntropy`)
Standard classification loss with a linear projection layer.
*   **Arguments**:
    *   `embedding_dim` (int): Input dimension (required to build the linear layer).
    *   `n_classes` (int): Number of classes.

---

## Adding Custom Losses

See [Extending DeepFense](../tutorials/extending.md) for a guide on creating custom loss modules.
