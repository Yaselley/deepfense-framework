import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import numpy as np
from deepfense.models.frontends.registry import register_frontend

@register_frontend("wav2vec")
class Wav2VecWrapper(nn.Module):
    """
    Wrapper for HuggingFace Wav2Vec2 models to extract embeddings from raw audio,
    without relying on Wav2Vec2Processor.
    """
    def __init__(self, model_name="facebook/wav2vec2-base"):
        """
        Args:
            model_name (str): HuggingFace Wav2Vec2 model identifier.
        """
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)

    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor or np.ndarray): Audio array of shape (L,) or (B, L)
        
        Returns:
            embeddings (torch.Tensor): Wav2Vec embeddings of shape (B, T, hidden_size)
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        # Ensure waveform has batch dimension
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, L)
        elif waveform.ndim == 2 and waveform.shape[1] == 1:
            waveform = waveform.squeeze(1)  # (B, L)

        waveform = waveform.to(next(self.model.parameters()).device)

        # Normalize waveform
        waveform = (waveform - waveform.mean(dim=1, keepdim=True)) / (waveform.std(dim=1, keepdim=True) + 1e-7)

        outputs = self.model(waveform)
        return outputs.last_hidden_state


# if __name__ == "__main__":
#     # Example usage
#     dummy_audio = torch.randn(32, 16000).to("cuda")  # 1 sec random audio
#     w2v = Wav2VecWrapper(model_name="facebook/wav2vec2-base").to("cuda")
#     embeddings = w2v(dummy_audio)
#     print("Embeddings shape:", embeddings.shape)
