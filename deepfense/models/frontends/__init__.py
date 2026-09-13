from .dummy import DummyFrontend
from .wav2vec2 import Wav2VecWrapper
from .wavlm import WavLMWrapper
from .hubert import HubertWrapper
from .mert import MERTWrapper
from .eat import EATWrapper
from .aut_qwen import AuTWrapper
from . import wav2vec2_layer_sum  # noqa: F401 — register wav2vec2_layer_sum
from . import wavlm_layer_sum  # noqa: F401 — register wavlm_layer_sum
from . import hubert_layer_sum  # noqa: F401 — register hubert_layer_sum
from . import wav2vec2_custom_layers  # noqa: F401 — register wav2vec2_custom_layers
from . import wav2vec2_ssl_sf  # noqa: F401 — register wav2vec2_ssl_sf

# from .mel_spec import MelSpectrogram # If implemented
