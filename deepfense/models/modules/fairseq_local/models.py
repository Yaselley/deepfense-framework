"""
Minimal, self-contained reimplementation of the fairseq Wav2Vec2 / HuBERT
architecture for **inference only** (features_only=True path).

No fairseq imports are needed.  Parameter names are kept identical to the
original so that official fairseq checkpoints can be loaded with
``model.load_state_dict(state, strict=False)``.

Based on:
  https://github.com/facebookresearch/fairseq  (MIT licence)
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Small helper modules (match fairseq parameter names exactly)
# ---------------------------------------------------------------------------

def get_activation_fn(name: str):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "tanh":
        return torch.tanh
    raise RuntimeError(f"Unknown activation: {name}")


class TransposeLast(nn.Module):
    def forward(self, x):
        return x.transpose(-1, -2)


class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        out = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.type_as(x)


class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, x):
        out = F.group_norm(
            x.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return out.type_as(x)


LayerNorm = Fp32LayerNorm


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.new(x)

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


class SamePad(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.remove = kernel_size % 2 == 0

    def forward(self, x):
        if self.remove:
            x = x[:, :, :-1]
        return x


# ---------------------------------------------------------------------------
#  Multi-head attention  (fairseq-compatible parameter names:
#    k_proj, v_proj, q_proj, out_proj)
# ---------------------------------------------------------------------------

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=False,
    ):
        tgt_len, bsz, _ = query.size()
        src_len = key.size(0)

        q = self.q_proj(query).view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(key).view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(value).view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool), float("-inf")
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights = self.dropout_module(attn_weights)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)
        return attn, None


# ---------------------------------------------------------------------------
#  Transformer encoder layer (matches fairseq key names exactly)
# ---------------------------------------------------------------------------

class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.activation_fn = get_activation_fn(activation_fn)
        self.layer_norm_first = layer_norm_first

        self.self_attn = MultiheadAttention(
            embedding_dim, num_attention_heads, dropout=attention_dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.self_attn_layer_norm = LayerNorm(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, embedding_dim)
        self.final_layer_norm = LayerNorm(embedding_dim)

    def forward(self, x, self_attn_padding_mask=None, need_weights=False, **kwargs):
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            layer_result = x
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x, key=x, value=x, key_padding_mask=self_attn_padding_mask
            )
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            layer_result = x
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, (attn, layer_result)


# ---------------------------------------------------------------------------
#  Convolutional feature extractor
# ---------------------------------------------------------------------------

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()
        assert mode in {"default", "layer_norm"}

        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(TransposeLast(), Fp32LayerNorm(n_out, elementwise_affine=True), TransposeLast()),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(n_out, n_out, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3
            dim, k, stride = cl
            self.conv_layers.append(
                block(
                    in_d, dim, k, stride,
                    is_layer_norm=(mode == "layer_norm"),
                    is_group_norm=(mode == "default" and i == 0),
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)  # BxT -> BxCxT
        for conv in self.conv_layers:
            x = conv(x)
        return x


# ---------------------------------------------------------------------------
#  Positional convolution helpers
# ---------------------------------------------------------------------------

def _make_conv_pos(embed_dim, kernel_size, groups, is_batch_norm=False):
    pos_conv = nn.Conv1d(
        embed_dim, embed_dim, kernel_size=kernel_size,
        padding=kernel_size // 2, groups=groups,
    )
    std = math.sqrt(4.0 / (kernel_size * embed_dim))
    nn.init.normal_(pos_conv.weight, mean=0, std=std)
    nn.init.constant_(pos_conv.bias, 0)

    if not is_batch_norm:
        pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
        pos_conv = nn.Sequential(pos_conv, SamePad(kernel_size), nn.GELU())
    else:
        batch_norm = nn.BatchNorm1d(embed_dim)
        pos_conv = nn.Sequential(batch_norm, pos_conv, SamePad(kernel_size), nn.GELU())
    return pos_conv


def _make_conv_pos_deep(embed_dim, kernel_size, groups, depth):
    k = max(3, kernel_size // depth)
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k // 2, groups=groups),
                SamePad(k),
                TransposeLast(),
                LayerNorm(embed_dim, elementwise_affine=False),
                TransposeLast(),
                nn.GELU(),
            )
            for _ in range(depth)
        ]
    )


def _pad_to_multiple(x, multiple, dim=-2, value=0):
    seqlen = x.size(dim)
    m = seqlen % multiple
    if m == 0:
        return x, 0
    remainder = multiple - m
    pad_offset = [0] * (-1 - dim) * 2 + [0, remainder]
    return F.pad(x, pad_offset, value=value), remainder


# ---------------------------------------------------------------------------
#  Transformer encoder
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
        layerdrop: float = 0.0,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        conv_pos_batch_norm: bool = False,
        pos_conv_depth: int = 1,
        required_seq_len_multiple: int = 2,
    ):
        super().__init__()
        self.dropout = dropout
        self.embedding_dim = embed_dim
        self.required_seq_len_multiple = required_seq_len_multiple
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop

        if pos_conv_depth > 1:
            self.pos_conv = _make_conv_pos_deep(embed_dim, conv_pos, conv_pos_groups, pos_conv_depth)
        else:
            self.pos_conv = _make_conv_pos(embed_dim, conv_pos, conv_pos_groups, is_batch_norm=conv_pos_batch_norm)

        self.layers = nn.ModuleList([
            TransformerSentenceEncoderLayer(
                embedding_dim=embed_dim,
                ffn_embedding_dim=ffn_dim,
                num_attention_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                layer_norm_first=layer_norm_first,
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x, padding_mask=None, layer=None):
        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x, pad_length = _pad_to_multiple(x, self.required_seq_len_multiple, dim=-2, value=0)
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        elif pad_length > 0 and padding_mask is not None:
            padding_mask, _ = _pad_to_multiple(padding_mask, self.required_seq_len_multiple, dim=-1, value=True)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, trf_layer in enumerate(self.layers):
            dropout_prob = np.random.random() if self.layerdrop > 0 else 1
            if not self.training or dropout_prob > self.layerdrop:
                x, (z, lr) = trf_layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append((x, z, lr))
            if layer is not None and i == layer:
                break

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        if pad_length > 0:
            x = x[:, :-pad_length]

        return x, layer_results


# ---------------------------------------------------------------------------
#  Unified Wav2Vec2 / HuBERT model  (inference-only, features_only path)
# ---------------------------------------------------------------------------

class Wav2Vec2Model(nn.Module):
    """
    Loads and runs inference for fairseq wav2vec2 / HuBERT / XLS-R
    checkpoints.  Only the ``features_only=True`` code path is implemented;
    pre-training heads (quantizer, label embeddings, etc.) are intentionally
    omitted.
    """

    def __init__(
        self,
        conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        extractor_mode: str = "default",
        conv_bias: bool = False,
        encoder_embed_dim: int = 768,
        encoder_layers: int = 12,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        feature_grad_mult: float = 1.0,
        encoder_layerdrop: float = 0.0,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        conv_pos_batch_norm: bool = False,
        pos_conv_depth: int = 1,
        required_seq_len_multiple: int = 2,
        crop_seq_to_multiple: int = 1,
    ):
        super().__init__()
        feature_enc_layers = eval(conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]
        self.feature_grad_mult = feature_grad_mult
        self.crop_seq_to_multiple = crop_seq_to_multiple

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=extractor_mode,
            conv_bias=conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, encoder_embed_dim)
            if self.embed != encoder_embed_dim
            else None
        )

        self.layer_norm = LayerNorm(self.embed)
        self.dropout_input = nn.Dropout(dropout_input)
        self.dropout_features = nn.Dropout(dropout_features)

        # kept so the checkpoint key ``mask_emb`` can be loaded without error
        self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_embed_dim).uniform_())

        self.encoder = TransformerEncoder(
            embed_dim=encoder_embed_dim,
            ffn_dim=encoder_ffn_embed_dim,
            num_heads=encoder_attention_heads,
            num_layers=encoder_layers,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            layer_norm_first=layer_norm_first,
            layerdrop=encoder_layerdrop,
            conv_pos=conv_pos,
            conv_pos_groups=conv_pos_groups,
            conv_pos_batch_norm=conv_pos_batch_norm,
            pos_conv_depth=pos_conv_depth,
            required_seq_len_multiple=required_seq_len_multiple,
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        conv_cfg_list = eval(
            getattr(self, "_conv_feature_layers_str", "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2")
        )
        for _, kernel_size, stride in conv_cfg_list:
            input_lengths = torch.floor((input_lengths - kernel_size).float() / stride + 1).long()
        return input_lengths

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        features_only: bool = True,
        layer: Optional[int] = None,
        **kwargs,
    ):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1, 2)  # BxCxT -> BxTxC
        features = self.layer_norm(features)

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)
            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )
            padding_mask[
                torch.arange(padding_mask.shape[0], device=padding_mask.device),
                output_lengths - 1,
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        tsz_drop = features.size(1) % self.crop_seq_to_multiple
        if tsz_drop != 0:
            features = features[:, :-tsz_drop]
            if padding_mask is not None:
                padding_mask = padding_mask[:, :-tsz_drop]

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        x = features  # no masking at inference time
        x, layer_results = self.encoder(x, padding_mask=padding_mask, layer=layer)

        return {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}

    def extract_features(self, source, padding_mask=None, mask=False, ret_conv=False, output_layer=None):
        res = self.forward(
            source, padding_mask=padding_mask, mask=False, features_only=True,
            layer=None if output_layer is None else output_layer - 1,
        )
        return (res["features"] if ret_conv else res["x"]), res["padding_mask"]
