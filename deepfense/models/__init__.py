from deepfense.models.detector import ModularDetector
from deepfense.models.frontends import (
    HuBERT,
    Wav2Vec2,
    WavLM,
)
from deepfense.models.backends import (
    AASIST,
    MLP,
    Nes2Net,
    TCM,
)
from deepfense.models.losses import (
    cross_entropy,
    a_softmax,
    am_softmax,
    oc_softmax,
)
