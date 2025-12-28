import torch
import torch.nn as nn
import torchaudio
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
import logging

logger = logging.getLogger(__name__)

@register_frontend("eat")
class EATWrapper(BaseFrontend):
    """
    Wrapper for EAT (Efficient Audio Transformer).
    Incorporates Fbank feature extraction logic consistent with EAT training.
    """
    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "huggingface") 
        self.ckpt_path = config.get("ckpt_path", "worstchan/EAT-large_epoch20_pretrain")
        self.freeze = config.get("freeze", False)
        self.trust_remote_code = config.get("trust_remote_code", True)

        # EAT normalization constants
        self.norm_mean = -4.268
        self.norm_std = 4.569
        self.target_sr = 16000

        if self.source == "huggingface":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.ckpt_path, 
                trust_remote_code=self.trust_remote_code
            )
        else:
            raise ValueError(f"EAT only supports 'huggingface' source. Got {self.source}")

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def _wav_to_fbank(self, waveform: torch.Tensor):
        """
        Convert waveform to fbank (batch-wise) following EAT preprocessing.
        Input: [B, T]
        Output: [B, T_frames, F_bins]
        """
        # We assume input is already 16kHz as per framework design.
        # If not, resampling should happen in data loading/augmentation pipeline.
        
        # Normalize per sample
        # waveform: [B, T]
        # mean: [B, 1]
        waveform = waveform - waveform.mean(dim=1, keepdim=True)

        fbanks = []
        # Processing individually because Kaldi fbank is 2D (1 channel, T)
        device = waveform.device
        for w in waveform:
            mel = torchaudio.compliance.kaldi.fbank(
                w.unsqueeze(0),
                htk_compat=True,
                sample_frequency=self.target_sr,
                use_energy=False,
                window_type='hanning',
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10
            )
            fbanks.append(mel)

        fbanks = torch.stack(fbanks, dim=0).to(device)  # [B, T_frames, F_bins]
        return fbanks

    def forward(self, input_data, mask=None):
        """
        Args:
            input_data: [B, T] raw waveform
            mask: [B, T] (optional, ignored here as EAT usually handles fixed size or internal masking)
        """
        # 1. Extract Fbanks
        feats = self._wav_to_fbank(input_data) # [B, T, F]
        
        # 2. Global Normalization
        feats = (feats - self.norm_mean) / (self.norm_std * 2)
        
        # 3. Add Channel Dim: [B, 1, T, F]
        feats = feats.unsqueeze(1)
        
        # 4. Forward Pass
        # EAT returns last_hidden_state or extracted_features
        
        if hasattr(self.model, "extract_features"):
             # Direct call if model exposes it (custom HF model code)
             x = self.model.extract_features(feats)
        else:
             # Standard AutoModel call
             outputs = self.model(feats)
             if hasattr(outputs, "last_hidden_state"):
                 x = outputs.last_hidden_state
             else:
                 x = outputs[0]
        
        return x
