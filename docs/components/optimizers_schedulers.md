# Optimizers & Schedulers

This page details the supported optimizers and learning rate schedulers available in DeepFense.

## Optimizers

Optimizers are defined in the `optimizer` section of `training` in the config.

### 1. Adam (`adam`)
Standard Adam optimizer.
*   **Config Type**: `adam`

**Parameters:**

* **lr** (*float*) - Learning rate (default: `1e-6`).
* **weight_decay** (*float*) - L2 penalty (default: `1e-4`).
* **betas** (*tuple*) - Coefficients for running averages of gradient (default: `(0.9, 0.999)`).

**Example:**
```yaml
optimizer:
  type: adam
  args:
    lr: 0.0001
    weight_decay: 0.0001
```

### 2. AdamW (`adamw`)
Adam with decoupled weight decay. Generally recommended over Adam for transformer-based models (Wav2Vec2, etc.).
*   **Config Type**: `adamw`

**Parameters:**

* **lr** (*float*) - Learning rate (default: `1e-6`).
* **weight_decay** (*float*) - Weight decay coefficient (default: `1e-4`).
* **betas** (*tuple*) - (default: `(0.9, 0.999)`).

**Example:**
```yaml
optimizer:
  type: adamw
  args:
    lr: 0.000001
    weight_decay: 0.01
```

### 3. SGD (`sgd`)
Stochastic Gradient Descent.
*   **Config Type**: `sgd`

**Parameters:**

* **lr** (*float*) - Learning rate.
* **momentum** (*float*) - Momentum factor (default: `0.9`).
* **weight_decay** (*float*) - (default: `1e-4`).

**Example:**
```yaml
optimizer:
  type: sgd
  args:
    lr: 0.01
    momentum: 0.9
```

---

## Schedulers

Schedulers adjust the learning rate during training. They are defined in the `scheduler` section of `training`.

### 1. Step LR (`step_lr`)
Decays the learning rate by `gamma` every `step_size` epochs.
*   **Config Type**: `step_lr`

**Parameters:**

* **step_size** (*int*) - Period of learning rate decay (default: `10`).
* **gamma** (*float*) - Multiplicative factor of learning rate decay (default: `0.1`).

**Example:**
```yaml
scheduler:
  type: step_lr
  args:
    step_size: 15
    gamma: 0.1
```

### 2. Cosine Annealing (`cosine`)
Set the learning rate using a cosine annealing schedule.
*   **Config Type**: `cosine`

**Parameters:**

* **T_max** (*int*) - Max number of iterations (usually set to total epochs).
* **eta_min** (*float*) - Minimum learning rate (default: `0`).

**Example:**
```yaml
scheduler:
  type: cosine
  args:
    T_max: 100
    eta_min: 0.0000001
```

### 3. Exponential LR (`exponential`)
Decays the learning rate of each parameter group by `gamma` every epoch.
*   **Config Type**: `exponential`

**Parameters:**

* **gamma** (*float*) - Multiplicative factor of learning rate decay (default: `0.9`).

**Example:**
```yaml
scheduler:
  type: exponential
  args:
    gamma: 0.95
```
