# Frontend Components

Frontends are responsible for converting raw audio waveforms into high-level features or representations.

## Available Frontends

### 1. Wav2Vec2 (`wav2vec2`)
Supports both Fairseq (original) and HuggingFace implementations.
*   **Config Type**: `wav2vec2`
*   **Arguments**:
    *   `source` (str): `"fairseq"` (default) or `"huggingface"`.
    *   `ckpt_path` (str):
        *   For **Fairseq**: Absolute path to `.pt` file.
        *   For **HuggingFace**: Model ID (e.g., `facebook/wav2vec2-base`) or local path.
    *   `freeze` (bool): Whether to freeze weights.

### 2. WavLM (`wavlm`)
Supports both Microsoft (original) and HuggingFace implementations.
*   **Config Type**: `wavlm`
*   **Arguments**:
    *   `source` (str): `"unil"` (default) or `"huggingface"`.
    *   `ckpt_path` (str):
        *   For **Microsoft**: Absolute path to `.pt` file.
        *   For **HuggingFace**: Model ID (e.g., `microsoft/wavlm-base`) or local path.
    *   `freeze` (bool): Whether to freeze weights.

### 3. HuBERT (`hubert`)
Supports both Fairseq (original) and HuggingFace implementations.
*   **Config Type**: `hubert`
*   **Arguments**:
    *   `source` (str): `"fairseq"` (default) or `"huggingface"`.
    *   `ckpt_path` (str):
        *   For **Fairseq**: Absolute path to `.pt` file.
        *   For **HuggingFace**: Model ID.
    *   `freeze` (bool): Whether to freeze weights.

### 4. MERT (`mert`)
Music Audio Pre-training model, specialized for music but useful for general audio.
*   **Config Type**: `mert`
*   **Arguments**:
    *   `source` (str): `"huggingface"` (default).
    *   `ckpt_path` (str): HF ID (e.g., `m-a-p/MERT-v1-95M`) or path.
    *   `trust_remote_code` (bool): Needed for some MERT versions (default `True`).
    *   `freeze` (bool).

### 5. EAT (`eat`)
Efficient Audio Transformer. **Requires 16kHz input**. It performs internal Fbank extraction.
*   **Config Type**: `eat`
*   **Arguments**:
    *   `source` (str): `"huggingface"` (default).
    *   `ckpt_path` (str): HF ID (e.g. `worstchan/EAT-large_epoch20_pretrain`).
    *   `trust_remote_code` (bool): (default `True`).
    *   `freeze` (bool).

### 6. Mel Spectrogram (`mel_spec`)
Standard Mel-spectrogram extraction using `torchaudio`.
*   **Config Type**: `mel_spec`
*   **Arguments**:
    *   `n_fft`: FFT window size (default: 1024).
    *   `hop_length`: Hop length (default: 160).
    *   `n_mels`: Number of mel bands (default: 80).

## Input/Output
*   **Input**: Raw waveform Tensor of shape `(Batch, Time)`.
*   **Output**: Feature Tensor of shape `(Batch, Time, Dim)` (Transformers) or `(Batch, Channels, Freq, Time)` (Spectrograms).

---

> **Next Step**: [Backends →](backends.md)
