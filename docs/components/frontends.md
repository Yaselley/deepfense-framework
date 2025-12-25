# Frontend Components

Frontends are responsible for converting raw audio waveforms into high-level features or representations.

## Available Frontends

### 1. Wav2Vec2 (`wav2vec2`)
Supports both Fairseq (original) and HuggingFace implementations.
*   **Config Type**: `wav2vec2`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `source` | str | `"fairseq"` (default) or `"huggingface"`. |
| `ckpt_path` | str | **Fairseq**: Path to `.pt` file.<br>**HuggingFace**: ID (e.g., `facebook/wav2vec2-base`) or path. |
| `freeze` | bool | Whether to freeze weights. |

### 2. WavLM (`wavlm`)
Supports both Microsoft (original) and HuggingFace implementations.
*   **Config Type**: `wavlm`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `source` | str | `"unil"` (default) or `"huggingface"`. |
| `ckpt_path` | str | **Microsoft**: Path to `.pt` file.<br>**HuggingFace**: ID (e.g., `microsoft/wavlm-base`) or path. |
| `freeze` | bool | Whether to freeze weights. |

### 3. HuBERT (`hubert`)
Supports both Fairseq (original) and HuggingFace implementations.
*   **Config Type**: `hubert`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `source` | str | `"fairseq"` (default) or `"huggingface"`. |
| `ckpt_path` | str | **Fairseq**: Path to `.pt` file.<br>**HuggingFace**: Model ID. |
| `freeze` | bool | Whether to freeze weights. |

### 4. MERT (`mert`)
Music Audio Pre-training model, specialized for music but useful for general audio.
*   **Config Type**: `mert`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `source` | str | `"huggingface"` (default). |
| `ckpt_path` | str | HF ID (e.g., `m-a-p/MERT-v1-95M`) or path. |
| `trust_remote_code` | bool | Needed for some MERT versions (default `True`). |
| `freeze` | bool | Whether to freeze weights. |

### 5. EAT (`eat`)
Efficient Audio Transformer. **Requires 16kHz input**. It performs internal Fbank extraction.
*   **Config Type**: `eat`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `source` | str | `"huggingface"` (default). |
| `ckpt_path` | str | HF ID (e.g. `worstchan/EAT-large_epoch20_pretrain`). |
| `trust_remote_code` | bool | (default `True`). |
| `freeze` | bool | Whether to freeze weights. |

### 6. Mel Spectrogram (`mel_spec`)
Standard Mel-spectrogram extraction using `torchaudio`.
*   **Config Type**: `mel_spec`

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `n_fft` | int | FFT window size (default: 1024). |
| `hop_length` | int | Hop length (default: 160). |
| `n_mels` | int | Number of mel bands (default: 80). |

## Input/Output
*   **Input**: Raw waveform Tensor of shape `(Batch, Time)`.
*   **Output**: Feature Tensor of shape `(Batch, Time, Dim)` (Transformers) or `(Batch, Channels, Freq, Time)` (Spectrograms).

---

> **Next Step**: [Backends →](backends.md)
