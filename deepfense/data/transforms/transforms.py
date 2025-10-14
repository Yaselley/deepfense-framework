import numpy as np
from deepfense.data.transforms.registry import register_transform

@register_transform.setdefault("pad")
def pad_combined(
    x: np.ndarray, 
    max_len: int = 64600, 
    random_pad: bool = False, 
    pad_type: str = "repeat"  # "repeat" or "zero"
):
    x_len = x.shape[0]

    # Truncate if longer than max_len
    if x_len > max_len:
        if random_pad:
            start = np.random.randint(0, x_len - max_len)
            return x[start:start + max_len]
        else:
            return x[:max_len]

    # Pad if shorter than max_len
    pad_len = max_len - x_len
    if pad_type == "repeat":
        repeats = int(np.ceil(max_len / x_len))
        padded = np.tile(x, repeats)[:max_len]
    elif pad_type == "zero":
        padded = np.zeros(max_len, dtype=x.dtype)
        padded[:x_len] = x
    else:
        raise ValueError(f"Unknown pad_type: {pad_type}. Use 'repeat' or 'zero'.")
    
    return padded
