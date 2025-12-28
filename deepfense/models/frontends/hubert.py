import os
import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
import logging

logger = logging.getLogger(__name__)

@register_frontend("hubert")
class HubertWrapper(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "fairseq")
        self.ckpt_path = config.get("ckpt_path", None)
        self.freeze = config.get("freeze", False)

        if self.ckpt_path is None:
            raise ValueError("ckpt_path must be provided in config")

        if self.source == "fairseq":
            try:
                import fairseq
            except ImportError:
                raise ImportError(
                    "fairseq is required for 'fairseq' source. "
                    "Please install it following the README instructions."
                )
            
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {self.ckpt_path}. "
                    "Please verify the path is correct."
                )
            
            try:
                models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                    [self.ckpt_path]
                )
                self.model = models[0]
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load fairseq model from {self.ckpt_path}: {e}"
                )

        elif self.source == "huggingface":
            try:
                from transformers import HubertModel
                self.model = HubertModel.from_pretrained(self.ckpt_path)
            except ImportError:
                raise ImportError(
                    "transformers is required for 'huggingface' source. "
                    "Please install it: pip install transformers"
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load HuggingFace model '{self.ckpt_path}': {e}"
                )
        
        else:
            raise ValueError(f"Unknown source: {self.source}. Must be 'fairseq' or 'huggingface'")

        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input_data, mask=None):
        if self.source == "fairseq":
            emb = self.model(
                input_data,
                mask=False,
                features_only=True,
            )
            x = emb["x"]
            return x

        elif self.source == "huggingface":
            attention_mask = None
            if mask is not None:
                attention_mask = mask.long()

            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state
