#!/usr/bin/env python
"""
infer.py — dead-simple "is this audio real or fake?" for clip-level DeepFense.

Why this exists
---------------
Running a pretrained DeepFense model on one audio file used to need two annoying
things that this script removes:

1.  A local fairseq / WavLM ``.pt`` base checkpoint.
    The shipped configs point ``frontend.args.ckpt_path`` at the author's machine
    (e.g. ``/netscratch/.../xlsr2_300m.pt``). The wav2vec2 / WavLM wrappers raise
    ``FileNotFoundError`` if that file is missing — *before* any weights load.
    But that base file is only used to (a) read the SSL architecture and (b) load
    base SSL weights, and BOTH are irrelevant for inference: ``best_model.pth``
    already contains the *fine-tuned* frontend weights.
    -> This script reconstructs the SSL architecture **directly from the shapes of
       the saved fine-tuned weights** and builds the skeleton with no base file
       and no network. (You can still pass ``--ssl-ckpt`` if you have the file.)

2.  The bonafide/spoof score is "hidden" inside the loss module.
    Each loss (CrossEntropy / OC-Softmax / AM-Softmax / A-Softmax) maps the
    backend embedding to a score differently (a logit, a cosine, a cosine margin).
    -> This script reads ``model(x)`` -> ``{"logits","scores"}`` (which DeepFense
       already routes through ``loss.get_logits`` / ``loss.get_score``) and turns
       it into a single REAL/FAKE verdict + confidence, with the right decision
       rule per loss family.

Usage
-----
    # From a local config + checkpoint:
    python infer.py audio.wav --config config.yaml --checkpoint best_model.pth

    # Straight from the HuggingFace hub (downloads config + best_model.pth):
    python infer.py audio.wav --model ASV19_WavLM_Nes2Net_NoAug_Seed42

    # Several files at once + machine-readable output:
    python infer.py a.flac b.wav c.mp3 --model <name> --json

This script is for the **clip-level** detectors (StandardDetector). For the
partial-deepfake / temporal models see the `deepfense-partial` branch.
"""

import os
import sys
import glob
import json
import argparse
import logging
import tempfile

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("deepfense.infer")

# Make `deepfense` importable when the script sits at the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Standard fairseq wav2vec2/HuBERT conv feature extractor (identical across
# base / large / xls-r). Used when synthesizing a skeleton config.
DEFAULT_CONV_LAYERS = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"

# Sources that need a local *base* SSL file at construction time.
LOCAL_FILE_SOURCES = {"fairseq", "unil"}
DEFAULT_SOURCE = {
    "wav2vec2": "fairseq", "hubert": "fairseq", "wavlm": "unil",
    "mert": "huggingface", "eat": "huggingface", "aut_qwen": "huggingface",
}


# --------------------------------------------------------------------------- #
# Config + checkpoint loading
# --------------------------------------------------------------------------- #
def load_config(path):
    """Load a YAML config to a plain dict (OmegaConf if available, else PyYAML)."""
    try:
        from omegaconf import OmegaConf
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)


def extract_state_dict(state):
    """Pull the model weights out of a checkpoint, tolerant to key naming."""
    if isinstance(state, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
        # Maybe it's already a raw state_dict (values are tensors).
        if state and all(hasattr(v, "shape") for v in state.values()):
            return state
    raise ValueError(
        "Could not find a model state_dict in the checkpoint. "
        "Expected a key like 'model_state'."
    )


def subdict(state_dict, prefix):
    """Return {key_without_prefix: value} for every key starting with prefix."""
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


# --------------------------------------------------------------------------- #
# Reconstruct the SSL architecture from the saved fine-tuned weights
# (no base .pt, no network)
# --------------------------------------------------------------------------- #
def _count_encoder_layers(fe):
    idx = set()
    for k in fe:
        if k.startswith("encoder.layers."):
            try:
                idx.add(int(k.split("encoder.layers.")[1].split(".")[0]))
            except (ValueError, IndexError):
                pass
    return (max(idx) + 1) if idx else 0


def _detect_extractor_mode(fe):
    # layer_norm mode => every conv block has a LayerNorm at .2.1
    # default mode    => only block 0 has a GroupNorm at .2
    for k in fe:
        if k.startswith("feature_extractor.conv_layers.") and k.endswith(".2.1.weight"):
            try:
                if int(k.split("conv_layers.")[1].split(".")[0]) >= 1:
                    return "layer_norm"
            except (ValueError, IndexError):
                pass
    return "default"


def infer_wav2vec2_arch(fe, expected_dim=None):
    """Infer wav2vec2 / HuBERT architecture kwargs from frontend weight shapes."""
    layers = _count_encoder_layers(fe)
    q = fe.get("encoder.layers.0.self_attn.q_proj.weight")
    fc1 = fe.get("encoder.layers.0.fc1.weight")
    if q is None or fc1 is None or layers == 0:
        raise ValueError(
            "Could not infer the SSL architecture from the checkpoint frontend "
            "weights (missing encoder layers). Pass --ssl-ckpt with the base "
            "model, or set the frontend source to 'huggingface' in the config."
        )
    embed = int(q.shape[0])
    if expected_dim and embed != expected_dim:
        log.warning("note: inferred encoder dim %d != backend input_dim %d",
                    embed, expected_dim)
    mode = _detect_extractor_mode(fe)
    arch = dict(
        conv_feature_layers=DEFAULT_CONV_LAYERS,
        extractor_mode=mode,
        conv_bias=("feature_extractor.conv_layers.0.0.bias" in fe),
        encoder_embed_dim=embed,
        encoder_layers=layers,
        encoder_ffn_embed_dim=int(fc1.shape[0]),
        encoder_attention_heads=max(1, embed // 64),
        layer_norm_first=(mode == "layer_norm"),
        activation_fn="gelu",
    )
    return arch


def infer_wavlm_arch(fe, expected_dim=None, max_distance=800):
    """Infer WavLM ('unil') config from frontend weight shapes.

    Note: WavLM's `max_distance` is a runtime-only hyperparameter (it shapes the
    relative-position bucketing but creates no weights), so it cannot be read
    from the checkpoint. We default to 800 (the value used by the official
    microsoft WavLM checkpoints); override with --wavlm-max-distance if needed.
    """
    base = infer_wav2vec2_arch(fe, expected_dim)  # shared encoder/conv inference
    rel_w = fe.get("encoder.layers.0.self_attn.relative_attention_bias.weight")
    cfg = dict(
        extractor_mode=base["extractor_mode"],
        encoder_layers=base["encoder_layers"],
        encoder_embed_dim=base["encoder_embed_dim"],
        encoder_ffn_embed_dim=base["encoder_ffn_embed_dim"],
        encoder_attention_heads=base["encoder_attention_heads"],
        layer_norm_first=base["layer_norm_first"],
        conv_feature_layers=DEFAULT_CONV_LAYERS,
        conv_bias=base["conv_bias"],
        relative_position_embedding=(rel_w is not None),
        num_buckets=int(rel_w.shape[0]) if rel_w is not None else 320,
        max_distance=max_distance,
        gru_rel_pos=("encoder.layers.0.self_attn.grep_linear.weight" in fe),
    )
    return cfg


def write_skeleton_w2v2(arch, tmpdir):
    """Write a tiny fairseq-style checkpoint (arch cfg + empty weights)."""
    import torch
    path = os.path.join(tmpdir, "ssl_skeleton.pt")
    torch.save({"cfg": {"model": arch, "task": {}}, "model": {}}, path)
    return path


def write_skeleton_wavlm(cfg, tmpdir):
    """Write a tiny WavLM-style checkpoint (cfg dict + empty weights)."""
    import torch
    path = os.path.join(tmpdir, "wavlm_skeleton.pt")
    torch.save({"cfg": cfg, "model": {}}, path)
    return path


def prepare_frontend(config, state_dict, tmpdir, ssl_ckpt=None, wavlm_max_distance=800):
    """Rewrite frontend config so construction needs no missing base file.

    Returns a short human-readable note describing what was done.
    """
    fe_cfg = config["model"]["frontend"]
    ftype = fe_cfg["type"]
    args = fe_cfg.setdefault("args", {})
    source = args.get("source", DEFAULT_SOURCE.get(ftype, "fairseq"))
    ckpt = args.get("ckpt_path")
    expected_dim = config["model"].get("backend", {}).get("args", {}).get("input_dim")

    # 1) User supplied a real base file -> just use it.
    if ssl_ckpt and os.path.exists(ssl_ckpt):
        args["ckpt_path"] = ssl_ckpt
        return f"frontend: using provided --ssl-ckpt ({ssl_ckpt})"

    # 2) HF-source frontends build themselves (transformers handles arch+weights).
    if source not in LOCAL_FILE_SOURCES:
        return f"frontend: source='{source}' (built by transformers)"

    # 3) Local-file source with the file present -> normal path.
    if ckpt and os.path.exists(ckpt):
        return f"frontend: using existing base file ({ckpt})"

    # 4) Local-file source, file missing -> reconstruct skeleton from weights.
    fe_weights = subdict(state_dict, "frontend.model.")
    if not fe_weights:
        raise ValueError(
            "Frontend base file is missing and the checkpoint has no "
            "'frontend.model.*' weights to reconstruct it from. "
            "Pass --ssl-ckpt or use a config with source: huggingface."
        )

    if ftype in ("wav2vec2", "hubert"):
        arch = infer_wav2vec2_arch(fe_weights, expected_dim)
        args["ckpt_path"] = write_skeleton_w2v2(arch, tmpdir)
        args["source"] = "fairseq"
        return ("frontend: rebuilt fairseq skeleton from checkpoint weights "
                f"(no base .pt) [dim={arch['encoder_embed_dim']}, "
                f"layers={arch['encoder_layers']}, mode={arch['extractor_mode']}]")

    if ftype == "wavlm":
        cfg = infer_wavlm_arch(fe_weights, expected_dim, max_distance=wavlm_max_distance)
        args["ckpt_path"] = write_skeleton_wavlm(cfg, tmpdir)
        args["source"] = "unil"
        log.warning("note: WavLM rebuilt from weights; assuming max_distance=%d "
                    "(override with --wavlm-max-distance if outputs look off).",
                    wavlm_max_distance)
        return ("frontend: rebuilt WavLM skeleton from checkpoint weights "
                f"(no base .pt) [dim={cfg['encoder_embed_dim']}, "
                f"layers={cfg['encoder_layers']}, buckets={cfg['num_buckets']}]")

    raise ValueError(
        f"Frontend '{ftype}' uses source '{source}' and its base file is "
        "missing. Auto-skeleton is supported for wav2vec2/hubert/wavlm. "
        "Pass --ssl-ckpt or switch the config to source: huggingface."
    )


# --------------------------------------------------------------------------- #
# Build the model
# --------------------------------------------------------------------------- #
def build_model(config, state_dict, device, ssl_ckpt=None, wavlm_max_distance=800,
                tmpdir=None):
    import torch
    from deepfense.utils.registry import build_detector
    import deepfense.models  # noqa: F401  (registers detectors/frontends/backends/losses)

    note = prepare_frontend(config, state_dict, tmpdir,
                            ssl_ckpt=ssl_ckpt, wavlm_max_distance=wavlm_max_distance)
    log.info(note)

    model = build_detector(config["model"]["type"], config["model"])
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    # Sanity check: did the frontend actually receive its weights?
    fe_missing = [k for k in missing if k.startswith("frontend.")]
    if fe_missing:
        log.warning("warning: %d frontend weights were NOT loaded — the rebuilt "
                    "architecture may not match the checkpoint.", len(fe_missing))

    model.to(device).eval()
    return model, note


# --------------------------------------------------------------------------- #
# Audio I/O (matches deepfense.data.transforms.load_audio + pad)
# --------------------------------------------------------------------------- #
def load_audio(path, target_sr=16000, mono=True):
    import soundfile as sf
    x, sr = sf.read(path, always_2d=False)
    if mono and x.ndim > 1:
        x = np.mean(x, axis=1)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr)
    return np.asarray(x, dtype=np.float32)


def pad_or_truncate(x, max_len, pad_type="repeat"):
    """Repeat-pad short clips / truncate long ones — same as eval's 'pad'."""
    n = x.shape[0]
    if n > max_len:
        return x[:max_len]
    reps = int(np.ceil(max_len / max(n, 1)))
    return np.tile(x, reps)[:max_len]


def resolve_max_len(config, override):
    """Figure out the eval crop length: CLI override, else config's pad transform."""
    if override is not None:
        return None if override <= 0 else override
    data = config.get("data", {})
    for split in ("test", "val", "train"):
        for t in (data.get(split, {}) or {}).get("base_transform", None) or []:
            if isinstance(t, dict) and t.get("type") == "pad" and t.get("max_len"):
                return int(t["max_len"])
    return None  # run on the full clip


# --------------------------------------------------------------------------- #
# Turn loss-coupled model outputs into a single verdict
# --------------------------------------------------------------------------- #
def decide(logits, scores, bonafide_label=1, threshold=None, oc_params=None):
    """Map model outputs -> per-sample verdict.

    DeepFense losses all follow "higher score = more bonafide", but the natural
    decision rule differs:
      * 2-class logits (CrossEntropy / AM-Softmax / A-Softmax): softmax over the
        two class outputs; decide on p(bonafide). --threshold is on that prob
        (default 0.5, i.e. argmax).
      * 1-class output (OC-Softmax): the score is cos(theta) to the bonafide
        center; decide score >= threshold (default = midpoint of the OC margins
        if known, else 0.0).
    """
    spoof_label = 1 - bonafide_label
    n = (logits.shape[0] if logits is not None else scores.shape[0])
    results = []

    two_class = logits is not None and logits.ndim == 2 and logits.shape[1] >= 2
    one_class = (logits is not None and logits.ndim == 2 and logits.shape[1] == 1) \
        or (logits is None and scores is not None)

    if two_class:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        probs = e / e.sum(axis=1, keepdims=True)
        thr = 0.5 if threshold is None else float(threshold)
        rule = f"softmax p(bonafide) >= {thr:g}"
        for i in range(n):
            p_bona = float(probs[i, bonafide_label])
            is_real = p_bona >= thr
            sc = float(scores[i]) if scores is not None else \
                float(logits[i, bonafide_label] - logits[i, spoof_label])
            results.append(_verdict(is_real, sc, p_bona, thr, rule))
        return results

    if one_class:
        s = logits[:, 0] if logits is not None else scores
        if threshold is not None:
            thr = float(threshold)
        elif oc_params and "w_posi" in oc_params and "w_nega" in oc_params:
            thr = 0.5 * (float(oc_params["w_posi"]) + float(oc_params["w_nega"]))
        else:
            thr = 0.0
        rule = f"cos(theta) >= {thr:g}"
        for i in range(n):
            sc = float(s[i])
            is_real = sc >= thr
            if oc_params and "w_posi" in oc_params and "w_nega" in oc_params:
                lo, hi = float(oc_params["w_nega"]), float(oc_params["w_posi"])
                p_bona = float(np.clip((sc - lo) / (hi - lo + 1e-8), 0.0, 1.0))
            else:
                p_bona = float(1.0 / (1.0 + np.exp(-4.0 * (sc - thr))))
            results.append(_verdict(is_real, sc, p_bona, thr, rule))
        return results

    raise ValueError("Unexpected model output shape; cannot derive a verdict.")


def _verdict(is_real, score, p_bona, thr, rule):
    return {
        "verdict": "REAL" if is_real else "FAKE",
        "label": "bonafide" if is_real else "spoof",
        "is_fake": (not is_real),
        "score": round(score, 6),           # model's bonafide score (higher=more real)
        "p_bonafide": round(p_bona, 6),      # probability/confidence of bonafide
        "confidence": round(p_bona if is_real else 1.0 - p_bona, 6),
        "threshold": thr,
        "rule": rule,
    }


def get_loss_meta(config):
    """Pull the main loss type, bonafide label, and OC margins from the config."""
    loss_cfg = config["model"].get("loss")
    if isinstance(loss_cfg, dict):
        loss_cfg = [loss_cfg]
    main = (loss_cfg or [{}])[0]
    oc = {k: main[k] for k in ("w_posi", "w_nega") if k in main} or None
    return {
        "type": main.get("type"),
        "bonafide_label": int(main.get("bonafide_label", 1)),
        "oc_params": oc,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def collect_audio_paths(args):
    paths = list(args.audio)
    if args.manifest:
        with open(args.manifest) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # take first comma field (supports plain list or csv with 'path')
                paths.append(line.split(",")[0].strip().strip('"'))
    # expand globs / dedupe while keeping order
    expanded, seen = [], set()
    for p in paths:
        for q in (glob.glob(p) or [p]):
            if q not in seen:
                seen.add(q)
                expanded.append(q)
    return expanded


def main():
    ap = argparse.ArgumentParser(
        description="Classify audio as REAL (bonafide) or FAKE (spoof) with a "
                    "clip-level DeepFense model.")
    ap.add_argument("audio", nargs="*", help="Audio file(s); globs allowed.")
    src = ap.add_argument_group("model source (choose one)")
    src.add_argument("--model", help="HuggingFace model name, e.g. "
                     "ASV19_WavLM_Nes2Net_NoAug_Seed42 (downloads config+weights).")
    src.add_argument("--config", help="Path to the model's config.yaml.")
    src.add_argument("--checkpoint", help="Path to best_model.pth.")
    ap.add_argument("--ssl-ckpt", help="Optional path to the base SSL .pt "
                    "(xls-r/WavLM). If omitted, the architecture is rebuilt from "
                    "the checkpoint weights — no base file needed.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold. For 2-class losses it is on "
                    "p(bonafide) (default 0.5); for OC-Softmax it is on cos(theta).")
    ap.add_argument("--max-len", type=int, default=None,
                    help="Pad/truncate length in samples. Default: the config's "
                    "eval crop. Use 0 to run on the full clip.")
    ap.add_argument("--wavlm-max-distance", type=int, default=800,
                    help="WavLM relative-position max_distance when rebuilding "
                    "from weights (default 800).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--manifest", help="Text/CSV file with one audio path per line.")
    ap.add_argument("--json", action="store_true", help="Emit JSON.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Show build details.")
    args = ap.parse_args()

    if args.verbose:
        log.setLevel(logging.INFO)

    # Resolve config + checkpoint -------------------------------------------------
    if args.model:
        from deepfense.hub import download_model
        files = download_model(args.model)
        config_path, ckpt_path = files["config"], files["checkpoint"]
    elif args.config and args.checkpoint:
        config_path, ckpt_path = args.config, args.checkpoint
    else:
        ap.error("provide either --model NAME or both --config and --checkpoint")

    audio_paths = collect_audio_paths(args)
    if not audio_paths:
        ap.error("no audio files given")

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    config = load_config(config_path)
    if config.get("model", {}).get("type") not in (None, "StandardDetector"):
        log.warning("note: this script targets StandardDetector (clip-level); "
                    "got type=%s.", config["model"].get("type"))

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(state)
    loss_meta = get_loss_meta(config)
    sr = int(config.get("data", {}).get("sampling_rate", 16000))
    max_len = resolve_max_len(config, args.max_len)

    with tempfile.TemporaryDirectory() as tmpdir:
        model, note = build_model(
            config, state_dict, device,
            ssl_ckpt=args.ssl_ckpt, wavlm_max_distance=args.wavlm_max_distance,
            tmpdir=tmpdir,
        )

        outputs = []
        for path in audio_paths:
            try:
                x = load_audio(path, target_sr=sr, mono=True)
                if max_len:
                    x = pad_or_truncate(x, max_len)
                xt = torch.from_numpy(x).unsqueeze(0).to(device)        # [1, T]
                mask = torch.ones_like(xt, dtype=torch.float32)
                with torch.no_grad():
                    out = model(xt, mask=mask)
                logits = out.get("logits")
                scores = out.get("scores")
                logits = logits.detach().cpu().numpy() if logits is not None else None
                scores = scores.detach().cpu().numpy() if scores is not None else None
                verdict = decide(logits, scores,
                                 bonafide_label=loss_meta["bonafide_label"],
                                 threshold=args.threshold,
                                 oc_params=loss_meta["oc_params"])[0]
            except Exception as e:  # keep going on a bad file
                verdict = {"error": f"{type(e).__name__}: {e}"}
            verdict["file"] = path
            outputs.append(verdict)

    # Report ---------------------------------------------------------------------
    if args.json:
        print(json.dumps(
            {"model": args.model or config_path, "loss": loss_meta["type"],
             "results": outputs}, indent=2))
        return

    print(f"\nmodel: {args.model or config_path}   loss: {loss_meta['type']}   "
          f"device: {device}")
    if max_len:
        print(f"crop:  {max_len} samples (~{max_len / sr:.1f}s); "
              f"use --max-len 0 for full clip")
    print("-" * 72)
    for r in outputs:
        name = os.path.basename(r["file"])
        if "error" in r:
            print(f"  {name:<32} ERROR  {r['error']}")
            continue
        mark = "🟢 REAL" if r["verdict"] == "REAL" else "🔴 FAKE"
        print(f"  {name:<32} {mark}   confidence {r['confidence']*100:5.1f}%   "
              f"(score {r['score']:+.3f}, p_bonafide {r['p_bonafide']:.3f})")
    print("-" * 72)
    print(f"rule: {outputs[0].get('rule', 'n/a')}   "
          f"(REAL = bonafide, FAKE = spoof). "
          f"Tune --threshold on a labeled dev set for best accuracy.\n")


if __name__ == "__main__":
    main()