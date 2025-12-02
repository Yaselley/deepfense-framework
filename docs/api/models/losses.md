# Loss Functions

Losses accept embeddings `[B, D]` and labels `[B]` and return a scalar loss.

## OCSoftmax
`deepfense.models.losses.oc_softmax.OCSoftmaxLoss`

One-Class Softmax Loss for generalized voice spoofing detection. Learns a compact center for the "bonafide" class and pushes "spoof" samples away.

**Config Args**:
*   `embedding_dim` (int): Input dimension.
*   `w_posi` (float): Weight for positive class.
*   `w_nega` (float): Weight for negative class.
*   `alpha` (float): Scaling factor.

## AMSoftmax
`deepfense.models.losses.am_softmax.AMSoftmaxLoss`

Additive Margin Softmax Loss.

**Config Args**:
*   `embedding_dim` (int).
*   `n_classes` (int): Number of classes (usually 2).
*   `s` (float): Scale factor.
*   `m` (float): Margin.

## CrossEntropy
`deepfense.models.losses.cross_entropy.CrossEntropyLoss`

Standard PyTorch Cross Entropy.

**Config Args**:
*   `class_weights` (List[float]): Optional class balancing weights.

