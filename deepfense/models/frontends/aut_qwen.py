import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

@register_frontend("aut_qwen")
class AuTWrapper(nn.Module):
    """
    Wrapper for the AuT (Audio Transformer) encoder from Qwen3-Omni.
    
    CRITICAL: This model uses "Packed Inference".
    It expects a single 2D tensor (Mels, Total_Time) containing all batch samples 
    concatenated together. It uses 'feature_lens' to split them internally.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.source = config.get("source", "huggingface")
        self.ckpt_path = config.get("ckpt_path", "Atotti/Qwen3-Omni-AudioTransformer")
        self.freeze = config.get("freeze", False)
        self.sampling_rate = config.get("sampling_rate", 16000)
        self.output_dim = config.get("output_dim", 2048)

        if self.source == "huggingface":
            self._load_from_huggingface()
            
        if self.freeze:
            for param in self.parameters():
                param.requires_grad = False
    
    def _load_from_huggingface(self):
        try:
            from transformers import WhisperFeatureExtractor
            from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
            
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.ckpt_path)
            self.model = Qwen3OmniMoeAudioEncoder.from_pretrained(
                self.ckpt_path, 
                torch_dtype=torch.bfloat16
            ).eval()
            
        except ImportError:
            logger.error("Required libraries not found. Please install transformers.")
            raise

    def preprocess_audio(self, waveform, sampling_rate=None):
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
            
        # Whisper extractor returns (Batch, Mels, Time)
        features = self.feature_extractor(
            waveform, 
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True 
        )
        return features["input_features"]
    
    def forward(self, input_data, feature_lens=None):
        """
        Forward pass that handles the required PACKING for Qwen3-Omni.
        
        Args:
            input_data: (Batch, Time) raw audio OR (Batch, Mels, Time) features
        """
        # 1. Preprocess / Normalize Input to (Batch, Mels, Time)
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
            
        if input_data.dim() == 2:
            input_features = self.preprocess_audio(input_data) # -> (Batch, 128, Time)
        elif input_data.dim() == 3:
            input_features = input_data
        else:
            raise ValueError(f"Invalid input shape: {input_data.shape}")
        
        # Move to model device/dtype
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(device=device, dtype=dtype)
        
        batch_size, mels, seq_len = input_features.shape

        if feature_lens is None:
            feature_lens = torch.full(
                (batch_size,), 
                seq_len, 
                dtype=torch.long, 
                device=device
            )

        input_permuted = input_features.permute(1, 0, 2) 
        packed_features = input_permuted.reshape(mels, -1)

        outputs = self.model(
            input_features=packed_features,
            feature_lens=feature_lens
        )
        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states.view(batch_size, -1, self.output_dim)

        return hidden_states

model = AuTWrapper({})
input_data = torch.randn(16, 64600)
output = model(input_data)
print(output.shape)
