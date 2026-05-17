"""Apply quantization to a Qwen3 model via module surgery.

Swap the 7 projection nn.Linear layers per decoder layer (q/k/v/o_proj,
gate/up/down_proj) for a quant linear. Norms (incl. q_norm/k_norm), embeddings,
lm_head, rotary stay full precision.

  method='sherry' -> Arenas    (1.25-bit, 3:4-sparse ternary)
  method='seq'    -> SEQLinear (2-bit, stretched elastic, learnable step)
"""
import torch
import torch.nn as nn

from quant import Arenas, SEQLinear, QuantLinear

ATTN_PROJ = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_PROJ = ["gate_proj", "up_proj", "down_proj"]


def _make_quant(linear, method, w_bits, granularity, group_size, N, M):
    if method == "sherry":
        q = Arenas(linear.in_features, linear.out_features,
                   bias=linear.bias is not None,
                   granularity=granularity, group_size=group_size, N=N, M=M)
    elif method == "seq":
        q = SEQLinear(linear.in_features, linear.out_features,
                      bias=linear.bias is not None, w_bits=w_bits)
    else:
        raise ValueError(f"unknown method {method}")
    with torch.no_grad():
        q.weight.copy_(linear.weight)
        if linear.bias is not None:
            q.bias.copy_(linear.bias)
    q = q.to(linear.weight.device, linear.weight.dtype)
    if method == "seq":
        q.init_clip_val()
    return q


def _iter_proj(model):
    for layer in model.model.layers:
        for holder, names in ((layer.self_attn, ATTN_PROJ), (layer.mlp, MLP_PROJ)):
            for name in names:
                yield holder, name


def quantize_qwen3(model, method="sherry", w_bits=2, granularity="per_group",
                   group_size=128, N=3, M=4):
    """Swap projection Linears for quant layers. Returns (model, num_quantized)."""
    n = 0
    for holder, name in _iter_proj(model):
        setattr(holder, name, _make_quant(getattr(holder, name), method, w_bits,
                                          granularity, group_size, N, M))
        n += 1
    return model, n


def bake_quantized(model):
    """Replace every quant linear with a plain nn.Linear holding its dequantized
    weight. The result is a standard Qwen3 checkpoint -> directly vLLM-loadable,
    and `F.linear(x, dequant(W))` is bit-identical to the fake-quant forward."""
    for holder, name in _iter_proj(model):
        q = getattr(holder, name)
        if not isinstance(q, QuantLinear):
            continue
        lin = nn.Linear(q.in_features, q.out_features, bias=q.bias is not None)
        lin = lin.to(q.weight.device, q.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(q.quantized_weight().to(q.weight.dtype))
            if q.bias is not None:
                lin.bias.copy_(q.bias)
        setattr(holder, name, lin)
    return model


def quant_stats(model):
    """Sparsity / level stats over quant layers."""
    layers, zeros, params, uniq = 0, 0, 0, 0
    for m in model.modules():
        if isinstance(m, QuantLinear):
            qw = m.quantized_weight()
            layers += 1
            zeros += (qw == 0).sum().item()
            params += qw.numel()
            if layers == 1:
                uniq = qw.unique().numel()
    return {"quant_layers": layers, "zero_frac": zeros / max(1, params),
            "quant_params": params, "unique_vals_layer0": uniq}
