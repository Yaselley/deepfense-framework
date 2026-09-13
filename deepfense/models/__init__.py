from deepfense.models.detector import ModularDetector
from deepfense.models.frontends import (
    dummy,
    hubert,
    hubert_layer_sum,
    wav2vec2,
    wav2vec2_layer_sum,
    wav2vec2_custom_layers,
    wav2vec2_ssl_sf,
    wavlm,
    wavlm_layer_sum,
)
from deepfense.models.backends import (
    aasist,
    mlp,
    nes2net,
    tcm,
)
from deepfense.models.losses import (
    cross_entropy,
    a_softmax,
    am_softmax,
    oc_softmax,
)
