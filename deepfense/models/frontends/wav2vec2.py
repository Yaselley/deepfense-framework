import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
import logging

logger = logging.getLogger(__name__)

@register_frontend("wav2vec2")
class Wav2VecWrapper(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)
        
        self.source = config.get("source", "fairseq")
        self.ckpt_path = config.get("ckpt_path", None)
        self.freeze = config.get("freeze", False)

        if self.source == "fairseq":
            import fairseq
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.ckpt_path]
            )
            self.model = models[0]
        
        elif self.source == "huggingface":
            from transformers import Wav2Vec2Model
            self.model = Wav2Vec2Model.from_pretrained(self.ckpt_path)
        
        else:
            raise ValueError(f"Unknown source: {self.source}")

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_data, mask=None):

        if self.source == "fairseq":
            emb = self.model(input_data, mask=False, features_only=True)
            x = emb["x"]
            return x

        elif self.source == "huggingface":
            attention_mask = None
            if mask is not None:
                attention_mask = mask.long()

            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state
