"""The fused operations the publisher's models were trained with, each evaluated as one operation
(docs/maintainer/lossless-route.md §2, §2a), and the parameter setter both shadow runners share."""

from __future__ import annotations

from fractions import Fraction
import math

import mpmath
import torch

from . import exact
from .tensor import F64, Shadow, shadow, value


# ------------------------------------------------------------------------------------------------ attention

def exact_attention(module, query, key, value_, attention_mask, scaling, dropout=0.0, **kwargs):
    """Attention as one correctly rounded operation per output: causal, or the keys a boolean (True: visible)
    or additive (0: visible) [batch, 1, queries, keys] mask leaves visible (QSA's causal and selected keys)."""
    q, k, v = (value(t) for t in (query, key, value_))
    batch, heads, length, d = q.shape
    keys = k.shape[2]
    k = k.repeat_interleave(heads // k.shape[1], dim=1)
    v = v.repeat_interleave(heads // v.shape[1], dim=1)
    if math.frexp(scaling)[0] != 0.5:
        raise NotImplementedError("shadow attention: the softmax scale is not a power of two")
    causal = torch.triu(torch.ones(length, keys, dtype=torch.bool), diagonal=1 + keys - length)
    out = torch.empty(batch, heads, length, d, dtype=F64)
    for b in range(batch):
        if attention_mask is None:
            hidden = causal
        else:
            m = value(attention_mask)[b, 0]
            hidden = ~m if m.dtype == torch.bool else m != 0
        s = torch.matmul(q[b], k[b].transpose(-1, -2)) * scaling
        err_s = torch.matmul(q[b].abs(), k[b].abs().transpose(-1, -2)) * scaling * ((d + 4) * exact.U * 1.01)
        s = s.masked_fill(hidden, -math.inf)
        err_s = err_s.masked_fill(hidden, 0.0)
        top = s.max(dim=-1, keepdim=True).values
        err_top = err_s.max(dim=-1, keepdim=True).values
        arg = s - top
        err_arg = err_s + err_top + arg.abs() * exact.U
        arg32 = exact.round_to(arg, torch.float32)
        ok = exact.certain(arg, err_arg, torch.float32) | hidden
        if not bool(ok.all()):
            for h, t, j in torch.nonzero(~ok).tolist():
                visible = torch.nonzero(~hidden[t]).flatten().tolist()
                arg32[h, t, j] = _exact_argument(q[b, h, t], k[b, h], s[h, t], err_s[h, t], scaling, visible, j)
        p = exact.transcendental("exp", arg32, torch.float32)
        num = torch.matmul(p, v[b])
        err_num = torch.matmul(p, v[b].abs()) * ((keys + 4) * exact.U * 1.01)
        den = p.sum(dim=-1, keepdim=True)
        err_den = den * ((keys + 4) * exact.U * 1.01)
        o = num / den
        err_o = (err_num + o.abs() * err_den) / den + o.abs() * exact.U
        rounded = exact.round_to(o, query.dtype)
        ok = exact.certain(o, err_o, query.dtype)
        if not bool(ok.all()):
            for h, t, c in torch.nonzero(~ok).tolist():
                visible = torch.nonzero(~hidden[t]).flatten().tolist()
                pj = [Fraction(float(p[h, t, j])) for j in visible]
                vj = [Fraction(float(v[b, h, j, c])) for j in visible]
                total = sum(pj, Fraction(0))
                rounded[h, t, c] = exact.round_fraction(sum((x * y for x, y in zip(pj, vj)), Fraction(0)) / total,
                                                        query.dtype)
        out[b] = rounded
    return Shadow(out.transpose(1, 2).contiguous(), query.dtype), None


def _exact_argument(qv, keys, s_row, err_row, scaling, visible, j) -> float:
    """fp32(s_j - max s) from exact scores (the maximum among the visible scores within error of the binary64 one)."""
    top = max(float(s_row[i]) for i in visible)
    slack = max(float(err_row[i]) for i in visible)
    candidates = [i for i in visible if float(s_row[i]) + float(err_row[i]) >= top - slack]
    qs = qv.tolist()
    score = lambda i: sum((Fraction(a) * Fraction(b) for a, b in zip(qs, keys[i].tolist())), Fraction(0)) * Fraction(scaling)
    m = max(score(i) for i in candidates)
    return exact.round_fraction(score(j) - m, torch.float32)


# -------------------------------------------------------------------------------------------- convolution

def exact_causal_conv(x, weight, bias=None, activation=None, seq_idx=None, **kwargs):
    """causal_conv1d_fn: the kernel accumulates in binary32 and applies SiLU to that: bf16(silu(fp32(sum over taps
    of w x))) per channel and token, the sum exact before its one rounding to binary32 (the engine's causal_conv
    kernel, ops/lossless/lossless_ops.cu: a SiLU of the exact sum rounded once to BF16 would differ where the
    SiLU lands within a binary32 step of a BF16 midpoint)."""
    if bias is not None or activation not in ("silu", "swish"):
        raise NotImplementedError("shadow convolution: only the bias-free SiLU form")
    xv, wv = value(x), value(weight)
    taps, length = wv.shape[1], xv.shape[-1]
    padded = torch.nn.functional.pad(xv, (taps - 1, 0))
    terms = [wv[None, :, i, None] * padded[:, :, i:i + length] for i in range(taps)]
    s = sum(terms)
    err_s = sum(t.abs() for t in terms) * ((taps + 2) * exact.U)
    s32 = exact.round_to(s, torch.float32)
    sure = exact.certain(s, err_s, torch.float32)
    for b, c, t in torch.nonzero(~sure).tolist():
        total = sum((Fraction(float(wv[c, i])) * Fraction(float(padded[b, c, t + i])) for i in range(taps)),
                    Fraction(0))
        s32[b, c, t] = exact.round_fraction(total, torch.float32)
    y = torch.nn.functional.silu(s32)  # of the binary32 value, itself exact in binary64
    err_y = y.abs() * (32 * exact.U) + 2.0 ** -1070
    out = exact.round_to(y, x.dtype)
    ok = exact.certain(y, err_y, x.dtype)
    if not bool(ok.all()):
        for b, c, t in torch.nonzero(~ok).tolist():
            sm = mpmath.mpf(float(s32[b, c, t]))
            out[b, c, t] = exact.round_mpf(sm / (1 + mpmath.exp(-sm)), x.dtype)
    return Shadow(out, x.dtype)


# --------------------------------------------------------------------------------------- gated delta rule

def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def fla_gated_delta_rule(query, key, value_, g, beta, initial_state=None, output_final_state=False,
                         use_qk_l2norm_in_kernel=False, **kwargs):
    """FLA's gated delta rule as the lossless route defines it, on Shadow tensors (each operation rounded)."""
    if initial_state is not None:
        raise NotImplementedError("shadow: the gated delta rule starts from a zero state")
    initial_dtype = query.dtype
    q, k, v, b, decay = (t.transpose(1, 2).to(torch.float32) for t in (query, key, value_, beta, g))
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm(q), _l2norm(k)
    scale = exact.round_mpf(1 / mpmath.sqrt(q.shape[-1]), torch.float32)
    q = q * scale
    batch, heads, length, dk = k.shape
    dv = v.shape[-1]
    state = shadow(torch.zeros(batch, heads, dk, dv, dtype=F64), torch.float32)
    out = shadow(torch.zeros(batch, heads, length, dv, dtype=F64), torch.float32)
    for t in range(length):
        state = state * decay[:, :, t].exp()[..., None, None]
        k_t = k[:, :, t]
        memory = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v[:, :, t] - memory) * b[:, :, t].unsqueeze(-1)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, t] = (state * q[:, :, t].unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).contiguous().to(initial_dtype), None


# ------------------------------------------------------------------------------------------------ helpers

def set_parameter(module: torch.nn.Module, dotted: str, tensor: torch.Tensor) -> None:
    *path, leaf = dotted.split(".")
    for part in path:
        module = getattr(module, part)
    if leaf in module._parameters:
        module._parameters[leaf] = tensor
    elif leaf in module._buffers:
        module._buffers[leaf] = tensor
    else:
        raise KeyError(dotted)
