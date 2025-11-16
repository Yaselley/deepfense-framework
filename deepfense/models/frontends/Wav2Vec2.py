import torch
import torch.nn as nn
import numpy as np
import fairseq
import logging
from deepfense.models.frontends.registry import register_frontend

# Suppress all fairseq logs
def suppress_fairseq_logging():
    fairseq_logger = logging.getLogger("fairseq")
    fairseq_logger.setLevel(logging.CRITICAL)  # only critical errors
    for handler in fairseq_logger.handlers[:]:
        fairseq_logger.removeHandler(handler)
suppress_fairseq_logging()

@register_frontend("wav2vec2")
class Wav2VecWrapper(nn.Module):
    
    def __init__(self, config):

        super().__init__()
        self.ckpt_path = config.get("ckpt_path", None)
        self.model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.ckpt_path])
        self.model = self.model[0]
        return

    def forward(self, input_data):
        emb = self.model(input_data, mask=False, features_only=True)
        x, layerresult = emb['x'], emb['layer_results']
        layerresult = torch.stack([t[0].permute(1,0,2) if isinstance(t, tuple) else t for t in layerresult], dim=1)
        return x