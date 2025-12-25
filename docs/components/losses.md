# Loss Functions

Loss modules in DeepFense are unified: they handle **projection**, **loss calculation**, and **score generation**.

## Available Losses

### 1. AM-Softmax (`AMSoftmax`)
**Additive Margin Softmax**. Forces classes to be separated by a margin `m` on the hypersphere.
### 1. AM-Softmax (`AMSoftmax`)
**Additive Margin Softmax**. Forces classes to be separated by a margin `m` on the hypersphere.

**Parameters:**

* **embedding_dim** (*int*) - Input feature dimension.
* **n_classes** (*int*) - Number of classes (usually 2).
* **m** (*float*) - Margin value (e.g., 0.3).
* **s** (*float*) - Scaling factor (e.g., 30.0).
* **weight** (*float*) - Loss weight in multi-loss setups.

**Example:**
```yaml
loss:
  type: AMSoftmax
  args:
    embedding_dim: 192
    n_classes: 2
    m: 0.3
    s: 30.0
```

### 2. A-Softmax (`ASoftmax`)
**Angular Softmax** (SphereFace). Uses multiplicative angular margin.

**Parameters:**

* **embedding_dim** (*int*) - Input dimension.
* **n_classes** (*int*) - Number of classes.
* **m** (*int*) - Integer margin (e.g., 4). Controls the angular margin size.
* **gamma** (*float*) - Focusing parameter.
* **lambda_min/max** (*float*) - Parameters for lambda annealing (stabilizes training).

**Example:**
```yaml
loss:
  type: ASoftmax
  args:
    embedding_dim: 192
    m: 4
```

### 3. OC-Softmax (`OCSoftmax`)
**One-Class Softmax**. Designed specifically for spoofing. It compacts the "Bonafide" class into a tight cluster and pushes "Spoof" samples away.

**Parameters:**

* **embedding_dim** (*int*) - Input dimension.
* **m_real** (*float*) - Margin for real speech (default: 0.5).
* **m_fake** (*float*) - Margin for fake speech (default: 0.2).
* **alpha** (*float*) - Scaling factor (default: 20.0).

**Example:**
```yaml
loss:
  type: OCSoftmax
  args:
    embedding_dim: 192
    m_real: 0.5
    m_fake: 0.2
```

### 4. Cross Entropy (`CrossEntropy`)
Standard classification loss with a linear projection layer.

**Parameters:**

* **embedding_dim** (*int*) - Input dimension (required to build the linear layer).
* **n_classes** (*int*) - Number of classes.

**Example:**
```yaml
loss:
  type: CrossEntropy
  args:
    embedding_dim: 192
    n_classes: 2
```

---

## Adding Custom Losses

See [Extending DeepFense](../user_guide/extending.md) for a guide on creating custom loss modules.

---

> **Next Step**: [Augmentations →](augmentations.md)
