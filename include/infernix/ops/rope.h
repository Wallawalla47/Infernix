#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <cstdint>
namespace infernix::ops {

/**
 * Applies split-half NeoX RoPE in place. For pair i in [0,rotary_dim/2), angle phi(i,t), and
 * each head:
 *
 *   ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *   ideal[i+rotary_dim/2] = x[i+R/2] * cos(phi) + x[i] * sin(phi).
 *
 * Dimensions [rotary_dim,head_dim) are unchanged. Supported modes are:
 *
 * - Text 1-D: positions I32 [T], either head_dim=256 with even 0<rotary_dim<=256, or the
 *   DFlash full-head domain head_dim=rotary_dim=128; phi=positions[t]*theta^(-2*i/rotary_dim).
 * - Text MRoPE: positions I32 [T,3], head_dim=256, rotary_dim=64; pair i uses axis i%3 with
 *   the same frequency as Text 1-D.
 * - Vision 2-D: positions I32 [T,2], head_dim=rotary_dim=72; pairs 0..17 use axis 0 and pairs
 *   18..35 use axis 1, each with local frequency theta^(-2*(i%18)/36).
 *
 * positions is contiguous and theta is positive and finite. Q/K tensors are BF16
 * [head_dim,heads,T] with positive head counts, contiguous head features and heads, and an optional
 * padded token stride. The registered optimized domains are D256/R64 Text Q/K head geometries
 * 24/4 and 16/2, D128/R128 1-D Text geometry 32/8, plus Vision geometry 16/16. q and k must not
 * overlap one another or positions. The Op mutates only dimensions [0,rotary_dim) of the supplied
 * Q/K tensor storage. The oracle evaluates the rotated dimensions naively in FP64 from the
 * represented inputs. The updated BF16 values are promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Unrotated
 * dimensions remain bit-exact. Private kernel arithmetic is implementation-defined. The Op uses no
 * workspace or persistent state. `execution` supplies the stream and the selected device's
 * positive physical SM count; the count selects launch geometry, not the transformation.
 */
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
          DeviceExecutionView execution);

/**
 * Optional text YaRN scaling. factor must be finite in [1,4], original_max_positions positive.
 * factor=1 delegates to the unscaled overload exactly (including Vision). factor>1 requires
 * Text 1-D or MRoPE and theta>1. Vision scaling is not supported.
 * For R=rotary_dim, define d(beta)=R*ln(original_max_positions/(2*pi*beta))/(2*ln(theta)),
 * low=max(floor(d(32)),0), high=min(ceil(d(1)),R-1). If low==high, replace high by high+0.001.
 * For pair i, ramp=clamp((i-low)/(high-low),0,1), and replace its inverse frequency by
 * theta^(-2*i/R)*((1-ramp)+ramp/factor). Multiply both cosine and sine by
 * 1+0.1*ln(factor). Axis selection, unrotated dimensions, BF16 output boundary and aliasing
 * are unchanged. No allocation, workspace, mutable global state or persistent state is used.
 */
struct RopeScaling {
    float factor                         = 1.0F;
    std::uint32_t original_max_positions = 262144;
};

// Prepare once, store as immutable model data, and pass by const reference at execution.
// Produced only by prepare_rope; callers must not modify fields after preparation. The plain
// value contains no pointers/device allocations and is copied into CUDA graph launch arguments.
struct PreparedRope {
    double inverse[128]{};
    float attention_scale = 1.0F;
    int rotary_dim        = 0;
    float theta           = 0.0F;
    float factor          = 1.0F;
};

PreparedRope prepare_rope(int rotary_dim, float theta, const RopeScaling& scaling = {});

void rope(const Tensor& positions, const PreparedRope& prepared, Tensor& q, Tensor& k,
          DeviceExecutionView execution);
void rope(const Tensor& positions, const PreparedRope& prepared, Tensor& x,
          DeviceExecutionView execution);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope(const Tensor& positions, int rotary_dim, float theta, Tensor& x,
          DeviceExecutionView execution);

} // namespace infernix::ops
