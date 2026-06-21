import os
import torch
import torch.nn as nn
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend
from deepfense.models.frontends.freeze_policy import apply_ssl_freeze_policy
import logging

logger = logging.getLogger(__name__)


@register_frontend("wav2vec2")
class Wav2VecWrapper(BaseFrontend):
    # Wav2Vec2 / XLS-R conv stack [(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2
    # has stride 5*2*2*2*2*2*2 = 320 samples (= 20 ms at 16 kHz).
    frontend_hop: int = 320

    def __init__(self, config):
        super().__init__(config)

        self.source = config.get("source", "fairseq")
        self.ckpt_path = config.get("ckpt_path", None)
        self.freeze = config.get("freeze", False)

        if self.ckpt_path is None:
            raise ValueError("ckpt_path must be provided in config")

        if self.source == "fairseq":
            if not os.path.exists(self.ckpt_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {self.ckpt_path}. "
                    "Please verify the path is correct."
                )
            from deepfense.models.modules.fairseq_local import load_fairseq_model
            self.model = load_fairseq_model(self.ckpt_path)

        elif self.source == "huggingface":
            from transformers import Wav2Vec2Model
            self.model = Wav2Vec2Model.from_pretrained(self.ckpt_path)

        else:
            raise ValueError(f"Unknown source: {self.source}. Must be 'fairseq' or 'huggingface'")

        self.freeze_summary = apply_ssl_freeze_policy(self.model, self.source, config)

    def forward(self, input_data, mask=None):
        """
        ``mask`` (optional): ``(B, T_audio)`` float / bool with ``1 = valid sample,
        0 = padding`` (the convention emitted by the dataloader's collate).

        Both the local fairseq ``Wav2Vec2Model`` and HuggingFace ``Wav2Vec2Model``
        natively support a padding mask -- they're just expressed in opposite
        polarities (HF wants ``attention_mask`` = 1 for valid; fairseq wants
        ``padding_mask`` = True for padded). We honor whichever the underlying
        model expects so that batch-padding never leaks into features. Critical
        for variable-length partial-deepfake training.
        """
        if self.source == "fairseq":
            padding_mask = None
            if mask is not None:
                padding_mask = mask.eq(0)  # True where padded
            emb = self.model(
                input_data,
                padding_mask=padding_mask,
                mask=False,
                features_only=True,
            )
            return emb["x"]

        elif self.source == "huggingface":
            attention_mask = None
            if mask is not None:
                attention_mask = mask.long()
            outputs = self.model(input_data, attention_mask=attention_mask)
            return outputs.last_hidden_state
