# Frontend Components

Frontends are responsible for converting raw audio waveforms into high-level features or representations.

## Available Frontends

### 1. Wav2Vec2 (`wav2vec2`)
Supports both Fairseq (original) and HuggingFace implementations.

**Configuration Signature:**
```yaml
frontend:
  type: wav2vec2
  args:
    source: string
    ckpt_path: string
    freeze: bool
```

**Parameters:**

* **source** - (*str*) `"fairseq"` (default) or `"huggingface"`.
* **ckpt_path** - (*str*) **Fairseq**: Path to `.pt` file. **HuggingFace**: ID (e.g., `facebook/wav2vec2-base`) or path.
* **freeze** - (*bool*) Whether to freeze weights.

**Example:**
```yaml
frontend:
  type: wav2vec2
  args:
    source: huggingface
    ckpt_path: facebook/wav2vec2-base
    freeze: True
```

---

### 2. WavLM (`wavlm`)
Supports both Microsoft (original) and HuggingFace implementations.

**Configuration Signature:**
```yaml
frontend:
  type: wavlm
  args:
    source: string
    ckpt_path: string
    freeze: bool
```

**Parameters:**

* **source** - (*str*) `"unil"` (default) or `"huggingface"`.
* **ckpt_path** - (*str*) **Microsoft**: Path to `.pt` file. **HuggingFace**: ID (e.g., `microsoft/wavlm-base`) or path.
* **freeze** - (*bool*) Whether to freeze weights.

**Example:**
```yaml
frontend:
  type: wavlm
  args:
    source: huggingface
    ckpt_path: microsoft/wavlm-base
    freeze: True
```

---

### 3. HuBERT (`hubert`)
Supports both Fairseq (original) and HuggingFace implementations.

**Configuration Signature:**
```yaml
frontend:
  type: hubert
  args:
    source: string
    ckpt_path: string
    freeze: bool
```

**Parameters:**

* **source** - (*str*) `"fairseq"` (default) or `"huggingface"`.
* **ckpt_path** - (*str*) **Fairseq**: Path to `.pt` file. **HuggingFace**: Model ID.
* **freeze** - (*bool*) Whether to freeze weights.

**Example:**
```yaml
frontend:
  type: hubert
  args:
    source: fairseq
    ckpt_path: /path/to/hubert_large_ll60k.pt
    freeze: True
```

---

### 4. MERT (`mert`)
Music Audio Pre-training model, specialized for music but useful for general audio.

**Configuration Signature:**
```yaml
frontend:
  type: mert
  args:
    ckpt_path: string
    trust_remote_code: bool
    freeze: bool
```

**Parameters:**

* **source** - (*str*) `"huggingface"` (default).
* **ckpt_path** - (*str*) HF ID (e.g., `m-a-p/MERT-v1-95M`) or path.
* **trust_remote_code** - (*bool*) Needed for some MERT versions (default `True`).
* **freeze** - (*bool*) Whether to freeze weights.

**Example:**
```yaml
frontend:
  type: mert
  args:
    ckpt_path: m-a-p/MERT-v1-95M
    freeze: True
```

---

### 5. EAT (`eat`)
Efficient Audio Transformer. It performs internal Fbank extraction.

**Configuration Signature:**
```yaml
frontend:
  type: eat
  args:
    ckpt_path: string
    trust_remote_code: bool
    freeze: bool
```

**Parameters:**

* **source** - (*str*) `"huggingface"` (default).
* **ckpt_path** - (*str*) HF ID (e.g. `worstchan/EAT-large_epoch20_pretrain`).
* **trust_remote_code** - (*bool*) (default `True`).
* **freeze** - (*bool*) Whether to freeze weights.

**Example:**
```yaml
frontend:
  type: eat
  args:
    ckpt_path: worstchan/EAT-large_epoch20_pretrain
    freeze: True
```

---

### 6. Mel Spectrogram (`mel_spec`)
Standard Mel-spectrogram extraction using `torchaudio`.

**Configuration Signature:**
```yaml
frontend:
  type: mel_spec
  args:
    n_fft: int
    hop_length: int
    n_mels: int
```

**Parameters:**

* **n_fft** - (*int*) FFT window size (default: 1024).
* **hop_length** - (*int*) Hop length (default: 160).
* **n_mels** - (*int*) Number of mel bands (default: 80).

**Example:**
```yaml
frontend:
  type: mel_spec
  args:
    n_fft: 1024
    hop_length: 160
    n_mels: 80
```

## Input/Output
*   **Input**: Raw waveform Tensor of shape `(Batch, Time)`.
*   **Output**: Feature Tensor. Shape depends on the frontend:
    *   Transformers (Wav2Vec2, WavLM, etc.): `(Batch, Time, Dim)`
    *   Spectrograms (MelSpec): `(Batch, Channels, Freq, Time)` - *Note: Backends need adjustment to handle 4D input.*

---

> **Next Step**: [Backends →](backends.md)
