from deepfense.models.detector import ModularDetector
from deepfense.models import temporal_detector  # noqa: F401 — TemporalDetector
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
    frame_mlp,  # noqa: F401 — FrameMLP
    gmlp,  # noqa: F401 — GMLP
)
from deepfense.models.losses import (
    cross_entropy,
    a_softmax,
    am_softmax,
    oc_softmax,
    framewise_ce,  # noqa: F401 — FramewiseCrossEntropy
)
