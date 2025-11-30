import torch
import torch.nn as nn
import fairseq
from deepfense.utils.registry import register_frontend
from deepfense.models.base_model import BaseFrontend

@register_frontend("hubert")
class HubertWrapper(BaseFrontend):
    def __init__(self, config):
        super().__init__(config)

        self.ckpt_path = config.get("ckpt_path", None)
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [self.ckpt_path]
        )
        self.model = models[0]
        return

    def forward(self, input_data, mask=None):
        emb = self.model(
            input_data,
            mask=False,
            features_only=True,
        )
        x, layer_results = emb["x"], emb["features"]
        layer_results = torch.stack(
            [
                t[0].permute(1, 0, 2) if isinstance(t, tuple) else t
                for t in layer_results
            ],
            dim=1,
        )

        return x
