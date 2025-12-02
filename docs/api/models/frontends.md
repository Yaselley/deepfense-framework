# Frontends

Frontends convert raw waveform `[B, T]` into feature representations `[B, C, F, T]`.

## Wav2Vec2
`deepfense.models.frontends.wav2vec2.Wav2VecWrapper`

Wrapper for Fairseq or HuggingFace Wav2Vec2 models.

**Config Args**:
*   `source` (str): "fairseq" or "huggingface".
*   `ckpt_path` (str): Path to `.pt` file or HF model ID.
*   `freeze` (bool): Whether to freeze parameters.

## WavLM
`deepfense.models.frontends.wavlm.WavLMFrontend`

Wrapper for Microsoft's WavLM.

**Config Args**:
*   `ckpt_path` (str): Path to `WavLM-Large.pt`.

## EAT
`deepfense.models.frontends.eat.EATFrontend`

Frontend for "EAT: Enhanced Audio Transformer".
