"""The shadow tensor: a binary64 value exactly representable in a nominal dtype.

A Shadow behaves like a tensor of its nominal dtype (BF16 or FP32), so the publisher's model code runs
unchanged, but every ATen operation on it computes the exact real result of that operation on its inputs
and rounds it once to the dtype PyTorch gives the result (found by running the operation on meta
tensors of the nominal dtypes). Python scalars meet BF16 and FP32 tensors as binary32 values, as in
PyTorch's opmath. Views alias the binary64 storage; in-place operations write the rounded result.
"""

from __future__ import annotations

import torch
from torch.utils._pytree import tree_flatten, tree_map

from . import exact

F64 = torch.float64
_REDUCED = (torch.bfloat16, torch.float16, torch.float32)
_TRANSCENDENTAL = {"exp", "sigmoid", "silu", "rsqrt", "cos", "sin", "tanh", "log", "log1p", "expm1", "sqrt"}
_FORBIDDEN = {"_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
              "_scaled_dot_product_flash_attention_for_cpu", "scaled_dot_product_attention",
              "_scaled_dot_product_attention_math", "baddbmm", "addbmm", "addmv", "_log_softmax", "logsumexp",
              "cumsum", "cumprod", "var", "std", "norm", "linalg_vector_norm", "prod", "erf", "gelu"}


class Shadow(torch.Tensor):
    inner: torch.Tensor

    @staticmethod
    def __new__(cls, inner: torch.Tensor, dtype: torch.dtype):
        out = torch.Tensor._make_wrapper_subclass(cls, inner.size(), strides=inner.stride(),
                                                  storage_offset=inner.storage_offset(), dtype=dtype,
                                                  device=inner.device, requires_grad=False)
        out.inner = inner
        return out

    def __repr__(self) -> str:
        return f"Shadow({self.dtype}, {tuple(self.shape)})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return dispatch(func, args, kwargs or {})


def shadow(values: torch.Tensor, dtype: torch.dtype) -> Shadow:
    """values (any floating dtype) as a Shadow of dtype, rounded once."""
    return Shadow(exact.round_to(values.to(F64), dtype), dtype)


def value(x) -> torch.Tensor:
    return x.inner if isinstance(x, Shadow) else x


def _meta(x):
    if isinstance(x, torch.device):
        return torch.device("meta")
    if isinstance(x, Shadow):
        return torch.empty_strided(x.inner.size(), x.inner.stride(), dtype=x.dtype, device="meta")
    if isinstance(x, torch.Tensor):
        return torch.empty_strided(x.size(), x.stride(), dtype=x.dtype, device="meta")
    return x


def _is_view(func) -> bool:
    return any(r.alias_info is not None and not r.alias_info.is_write for r in func._schema.returns)


def _is_inplace(func) -> bool:
    return any(a.alias_info is not None and a.alias_info.is_write for a in func._schema.arguments)


def _inner(x, opmath: bool):
    """An argument as binary64 (Shadow values; plain floating tensors widened exactly; weak scalars as binary32)."""
    if isinstance(x, Shadow):
        return x.inner
    if isinstance(x, torch.Tensor):
        if x.dtype.is_floating_point:
            if opmath and x.dim() == 0 and x.dtype == F64:
                return exact.round_to(x, torch.float32)
            return x.to(F64)
        return x
    if isinstance(x, float) and opmath:
        return float(torch.tensor(x, dtype=torch.float32))
    if isinstance(x, torch.dtype) and x.is_floating_point:
        return F64
    return x


def _wrap(out, meta, view: bool):
    if isinstance(out, torch.Tensor) and isinstance(meta, torch.Tensor) and meta.dtype.is_floating_point:
        return Shadow(out if view else exact.round_to(out, meta.dtype), meta.dtype)
    if isinstance(out, (list, tuple)):
        wrapped = [_wrap(o, m, view) for o, m in zip(out, meta)]
        return type(out)(wrapped) if isinstance(out, tuple) else wrapped
    return out


def _floating_dtype(args) -> torch.dtype:
    dtypes = [a.dtype for a in tree_flatten(args)[0] if isinstance(a, torch.Tensor) and a.dtype.is_floating_point]
    order = [torch.bfloat16, torch.float16, torch.float32, F64]
    return max(dtypes, key=order.index) if dtypes else torch.float32


def dispatch(func, args, kwargs):
    name = func.overloadpacket.__name__.rstrip("_")
    if name in _FORBIDDEN:
        raise NotImplementedError(f"shadow: {func} has no exact form here (override the module that calls it)")
    try:
        meta = func(*tree_map(_meta, args), **tree_map(_meta, kwargs))
    except Exception:  # data-dependent shapes: the result takes the widest floating input dtype
        meta = None
    out_meta = meta[0] if isinstance(meta, (list, tuple)) and meta else meta
    out_dtype = out_meta.dtype if isinstance(out_meta, torch.Tensor) else _floating_dtype((args, kwargs))
    opmath = out_dtype in _REDUCED
    inplace = _is_inplace(func)
    if inplace and not isinstance(args[0], Shadow):
        raise NotImplementedError(f"shadow: {func} writes a plain tensor (make the destination a Shadow)")
    handler = _HANDLERS.get(name)
    if handler is not None:
        result = handler(func, args, kwargs, args[0].dtype if inplace else out_dtype)
        if inplace:
            args[0].inner.copy_(value(result if not isinstance(result, tuple) else result[0]))
            return args[0]
        return result
    inner_args = tree_map(lambda x: _inner(x, opmath), args)
    inner_kwargs = tree_map(lambda x: _inner(x, opmath), kwargs)
    result = func(*inner_args, **inner_kwargs)
    if inplace:
        target = args[0]
        target.inner.copy_(exact.round_to(target.inner, target.dtype))
        return target
    if meta is None:
        if isinstance(result, torch.Tensor) and result.dtype.is_floating_point:
            return Shadow(exact.round_to(result, out_dtype), out_dtype)
        return result
    return _wrap(result, meta, _is_view(func))


# -------------------------------------------------------------------------------------------------- handlers

def _matmul(func, args, kwargs, dtype):
    name = func.overloadpacket.__name__
    if name == "addmm":
        bias, a, b = (value(x) for x in args[:3])
        if kwargs.get("beta", 1) != 1 or kwargs.get("alpha", 1) != 1:
            raise NotImplementedError("shadow: addmm with beta or alpha")
        return Shadow(exact.matmul(_f64(a), _f64(b), dtype, _f64(bias)), dtype)
    a, b = (_f64(value(x)) for x in args[:2])
    if name == "mv":
        return Shadow(exact.matmul(a, b[:, None], dtype)[:, 0], dtype)
    if name == "dot":
        return Shadow(exact.matmul(a[None, :], b[:, None], dtype)[0, 0], dtype)
    return Shadow(exact.matmul(a, b, dtype), dtype)


def _f64(x):
    return x.to(F64) if isinstance(x, torch.Tensor) and x.dtype != F64 and x.dtype.is_floating_point else x


def _dims(args, kwargs):
    dims = kwargs.get("dim", args[1] if len(args) > 1 else None)
    if dims is None:
        return ()
    return (dims,) if isinstance(dims, int) else tuple(dims)


def _sum(func, args, kwargs, dtype):
    x = _f64(value(args[0]))
    dims = _dims(args, kwargs)
    keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 and isinstance(args[2], bool) else False)
    return Shadow(exact.reduce_sum(x, dims, keepdim, dtype), dtype)


def _mean(func, args, kwargs, dtype):
    x = _f64(value(args[0]))
    dims = _dims(args, kwargs)
    keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 and isinstance(args[2], bool) else False)
    n = 1
    for d in (dims or range(x.dim())):
        n *= x.shape[d]
    return Shadow(exact.reduce_sum(x, dims, keepdim, dtype, divisor=n), dtype)


def _unary(func, args, kwargs, dtype):
    name = func.overloadpacket.__name__.rstrip("_")
    return Shadow(exact.transcendental(name, _f64(value(args[0])), dtype), dtype)


def _softplus(func, args, kwargs, dtype):
    x = _f64(value(args[0]))
    beta = args[1] if len(args) > 1 else kwargs.get("beta", 1)
    threshold = args[2] if len(args) > 2 else kwargs.get("threshold", 20)
    return Shadow(exact.softplus(x, float(beta), float(threshold), dtype), dtype)


def _pow(func, args, kwargs, dtype):
    base, exponent = args[0], args[1]
    if isinstance(base, (int, float)):
        return Shadow(exact.power_scalar_base(float(base), _f64(value(exponent)), dtype), dtype)
    x = _f64(value(base))
    if isinstance(exponent, (int, float)):
        if exponent == 2:
            return Shadow(exact.round_to(x * x, dtype), dtype)
        if exponent == 1:
            return Shadow(exact.round_to(x, dtype), dtype)
        if exponent == 0.5:
            return Shadow(exact.transcendental("sqrt", x, dtype), dtype)
        if exponent == -0.5:
            return Shadow(exact.transcendental("rsqrt", x, dtype), dtype)
    raise NotImplementedError(f"shadow: pow with exponent {exponent!r}")


def _softmax(func, args, kwargs, dtype):
    x, dim = _f64(value(args[0])), args[1]
    return Shadow(exact.softmax(x, dim, dtype), dtype)


def _topk(func, args, kwargs, dtype):
    x = _f64(value(args[0]))
    k = args[1]
    dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
    largest = args[3] if len(args) > 3 else kwargs.get("largest", True)
    if not largest:
        raise NotImplementedError("shadow: topk smallest")
    # Equal values: the lower index first (a stable sort of the negated values).
    order = torch.sort(-x, dim=dim, stable=True).indices.narrow(dim, 0, k)
    return torch.return_types.topk((Shadow(torch.gather(x, dim, order), dtype), order))


def _convolution(func, args, kwargs, dtype):
    """A depthwise 1-D convolution without bias or padding (Flash-Next PLE's dilated one): every output the
    exact sum of its taps' products, rounded once."""
    x, w, bias, stride, padding, dilation, transposed, _, groups = args[:9]
    x, w = _f64(value(x)), _f64(value(w))
    if (bias is not None or transposed or x.dim() != 3 or groups != x.shape[1] or w.shape[1] != 1
            or list(stride) != [1] or list(padding) != [0]):
        raise NotImplementedError("shadow: only bias-free unpadded depthwise 1-D convolutions")
    taps, step = w.shape[-1], dilation[0]
    length = x.shape[-1] - (taps - 1) * step
    terms = [w[None, :, 0, i, None] * x[:, :, i * step:i * step + length] for i in range(taps)]
    total = sum(terms)
    error = sum(t.abs() for t in terms) * ((taps + 2) * exact.U) + 2.0 ** -1070
    out = exact.round_to(total, dtype)
    ok = exact.certain(total, error, dtype)
    if not bool(ok.all()):
        from fractions import Fraction
        for b, c, t in torch.nonzero(~ok).tolist():
            q = sum((Fraction(float(w[c, 0, i])) * Fraction(float(x[b, c, t + i * step])) for i in range(taps)),
                    Fraction(0))
            out[b, c, t] = exact.round_fraction(q, dtype)
    return Shadow(out, dtype)


_HANDLERS = {
    "convolution": _convolution,
    "mm": _matmul, "bmm": _matmul, "addmm": _matmul, "mv": _matmul, "dot": _matmul,
    "sum": _sum, "mean": _mean, "softplus": _softplus, "pow": _pow, "_softmax": _softmax, "topk": _topk,
    **{name: _unary for name in _TRANSCENDENTAL},
}
