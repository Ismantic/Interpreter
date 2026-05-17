"""Quantization layers for Qwen3 QAT -- two methods.

- Sherry 1.25-bit (method='sherry'): NMQuant + Arenas. 3:4-sparse ternary
  ({-1,0,+1}, top-3 of every 4 kept), per-group absmean scale. From the Sherry
  paper (AngelSlim origin/sherry, modified from ParetoQ).
- SEQ 2-bit (method='seq'): StretchedElasticQuant + SEQLinear. Stretched
  Elastic Quantization -> 4 levels with a learnable per-channel step size.
  This is the algorithm behind HY-MT1.5-1.8B-2bit. Code ported verbatim from
  the same utils_quant.py.

Both quant linears subclass `QuantLinear`: a learnable FP master weight, a
fake-quant forward (STE backward), and a convex annealing residual
`out = (1-eps)*quant(x) + eps*fp(x)` with eps annealed 1->0 by the trainer, so
QAT starts at exact FP and eases into the quantized path -- suited to
preserving an already-strong model. (The Sherry paper's literal Arenas is
additive `quant + eps*fp`; the convex form avoids the eps~1 output blow-up
while still injecting W into the backward path.)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantLinear(nn.Linear):
    """nn.Linear + annealing-residual coefficient `eps` (set by the trainer)."""

    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.eps = 0.0

    def _blend(self, x, q_out):
        if self.training and self.eps > 0.0:
            q_out = (1.0 - self.eps) * q_out + self.eps * F.linear(x, self.weight.to(x.dtype))
        if self.bias is not None:
            q_out = q_out + self.bias
        return q_out


# ===================== Sherry 1.25-bit: 3:4-sparse ternary =====================

class NMQuant(torch.autograd.Function):
    """N:M-sparse ternary fake-quant. Straight-through estimator on backward."""

    @staticmethod
    def forward(ctx, weight, granularity, group_size, N, M):
        shape = weight.shape
        assert len(shape) == 2 and shape[1] % M == 0
        w = weight.reshape(shape[0], shape[1] // M, M)
        _, topk = torch.topk(w.abs(), N, dim=-1)
        mask = torch.zeros_like(w, dtype=torch.bool).scatter_(-1, topk, True)
        sparse = (w * mask).reshape(shape)
        if granularity == "per_tensor":
            x = sparse.reshape(1, -1)
        elif granularity == "per_channel":
            x = sparse.reshape(shape[0], -1)
        elif granularity == "per_group":
            assert sparse.numel() % group_size == 0
            x = sparse.reshape(-1, group_size)
        else:
            raise NotImplementedError(granularity)
        scale = x.abs().sum(-1, keepdim=True) / (x.shape[-1] / M * N)
        return (torch.sign(x) * scale).reshape(shape)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class Arenas(QuantLinear):
    def __init__(self, in_features, out_features, bias=False,
                 granularity="per_group", group_size=128, N=3, M=4):
        super().__init__(in_features, out_features, bias=bias)
        self.granularity = granularity
        self.group_size = group_size
        self.N = N
        self.M = M

    def forward(self, x):
        qw = NMQuant.apply(self.weight, self.granularity, self.group_size,
                           self.N, self.M).to(x.dtype)
        return self._blend(x, F.linear(x, qw))

    @torch.no_grad()
    def quantized_weight(self):
        return NMQuant.apply(self.weight, self.granularity, self.group_size,
                             self.N, self.M)


# ===================== SEQ 2-bit: stretched elastic quant =====================

class StretchedElasticQuant(torch.autograd.Function):
    """Stretched Elastic Quantization (ported verbatim from Sherry utils_quant.py,
    itself modified from Learned Step-size Quantization). num_bits=2 -> 4 levels."""

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        ctx.num_bits = num_bits
        if num_bits >= 16:
            return input
        eps = torch.tensor(0.00001, device=input.device).float()
        alpha = torch.where(alpha > eps, alpha, eps)
        clip_val = 1 - 1e-2
        if num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (num_bits - 1)
            shift = 0.5
        Qp = (n_levels - shift) / n_levels
        Qn = -Qp
        grad_scale = 1.0 / math.sqrt(input.numel() * Qp) if Qp else 1.0 / math.sqrt(input.numel())
        ctx.save_for_backward(input, alpha)
        ctx.other = grad_scale, Qn, Qp, layerwise
        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (torch.round(torch.clamp(input / alpha, -clip_val, clip_val) * n_levels - shift)
                   + shift) / n_levels
        return q_w * alpha

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None
        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp, layerwise = ctx.other
        q_w = input_ / alpha
        clip_val = 1 - 1e-2
        if ctx.num_bits == 0:
            n_levels = 1.5
            shift = 0
        else:
            n_levels = 2 ** (ctx.num_bits - 1)
            shift = 0.5
        indicate_small = (q_w < -clip_val).float()
        indicate_big = (q_w > clip_val).float()
        indicate_middle = 1.0 - indicate_small - indicate_big
        if ctx.num_bits == 1:
            grad_alpha = ((input_.sign()) * grad_output * grad_scale)
            grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        else:
            grad_alpha = ((indicate_small * Qn + indicate_big * Qp
                           + indicate_middle * (-q_w + (torch.round(
                               torch.clamp(q_w, -clip_val, clip_val) * n_levels - shift)
                               + shift) / n_levels))
                          * grad_output * grad_scale)
            grad_alpha = torch.sum(grad_alpha, dim=-1, keepdim=True)
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class SEQLinear(QuantLinear):
    """2-bit SEQ linear with a learnable per-output-channel step size."""

    def __init__(self, in_features, out_features, bias=False, w_bits=2):
        super().__init__(in_features, out_features, bias=bias)
        self.w_bits = w_bits
        self.weight_clip_val = nn.Parameter(torch.ones(out_features, 1))

    @torch.no_grad()
    def init_clip_val(self):
        # LSQ-style step-size init: alpha = 2*mean|W| / sqrt(Qp), per output channel,
        # so the weights spread across all 2^w_bits levels (max|W| would collapse
        # almost everything onto the inner levels). alpha is learned thereafter.
        n_levels = 2 ** (self.w_bits - 1)
        Qp = (n_levels - 0.5) / n_levels
        a = 2.0 * self.weight.abs().mean(dim=1, keepdim=True) / (Qp ** 0.5)
        self.weight_clip_val.copy_(a.clamp_min(1e-5).to(self.weight_clip_val.dtype))

    def forward(self, x):
        qw = StretchedElasticQuant.apply(self.weight, self.weight_clip_val,
                                         self.w_bits, False).to(x.dtype)
        return self._blend(x, F.linear(x, qw))

    @torch.no_grad()
    def quantized_weight(self):
        return StretchedElasticQuant.apply(self.weight, self.weight_clip_val,
                                           self.w_bits, False)


# ===================== shared trainer hooks =====================

def set_arenas_eps(model, eps):
    """Set the annealing residual coefficient on all quant layers."""
    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.eps = float(eps)


def anneal_eps(step, total_steps, fp_warmup=0.05, decay_end=0.75):
    """Convex-blend eps schedule, 1 -> 0:
      [0, fp_warmup)         eps = 1   (pure FP warmup)
      [fp_warmup, decay_end) eps cosine 1 -> 0
      [decay_end, end]       eps = 0   (pure quant convergence)
    """
    w = fp_warmup * total_steps
    d = decay_end * total_steps
    if step < w:
        return 1.0
    if step >= d:
        return 0.0
    progress = (step - w) / max(1.0, d - w)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
