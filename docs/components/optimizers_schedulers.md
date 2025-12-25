# Optimizers & Schedulers

This page details the supported optimizers and learning rate schedulers available in DeepFense.

## Optimizers

Optimizers are defined in the `optimizer` section of `training` in the config.

### 1. Adam (`adam`)
Standard Adam optimizer.
*   **Config Type**: `adam`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `lr` | float | Learning rate (default: `1e-6`). |
| `weight_decay` | float | L2 penalty (default: `1e-4`). |
| `betas` | tuple | Coefficients for running averages of gradient (default: `(0.9, 0.999)`). |

### 2. AdamW (`adamw`)
Adam with decoupled weight decay. Generally recommended over Adam for transformer-based models (Wav2Vec2, etc.).
*   **Config Type**: `adamw`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `lr` | float | Learning rate (default: `1e-6`). |
| `weight_decay` | float | Weight decay coefficient (default: `1e-4`). |
| `betas` | tuple | (default: `(0.9, 0.999)`). |

### 3. SGD (`sgd`)
Stochastic Gradient Descent.
*   **Config Type**: `sgd`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `lr` | float | Learning rate. |
| `momentum` | float | Momentum factor (default: `0.9`). |
| `weight_decay` | float | (default: `1e-4`). |

---

## Schedulers

Schedulers adjust the learning rate during training. They are defined in the `scheduler` section of `training`.

### 1. Step LR (`step_lr`)
Decays the learning rate by `gamma` every `step_size` epochs.
*   **Config Type**: `step_lr`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `step_size` | int | Period of learning rate decay (default: `10`). |
| `gamma` | float | Multiplicative factor of learning rate decay (default: `0.1`). |

### 2. Cosine Annealing (`cosine`)
Set the learning rate using a cosine annealing schedule, where `eta_min` is the minimum learning rate.
*   **Config Type**: `cosine`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `T_max` | int | Max number of iterations (usually set to total epochs). |
| `eta_min` | float | Minimum learning rate (default: `0`). |

### 3. Exponential LR (`exponential`)
Decays the learning rate of each parameter group by `gamma` every epoch.
*   **Config Type**: `exponential`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `gamma` | float | Multiplicative factor of learning rate decay (default: `0.9`). |
