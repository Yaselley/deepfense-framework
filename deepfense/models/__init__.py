from deepfense.models.detector import ModularDetector
from deepfense.models.frontends import (
    hubert,
    wav2vec2,
    wavlm,
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
