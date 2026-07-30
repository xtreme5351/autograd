//
// Created by Pranav C on 29/07/2026.
//
// Metal compute kernels for the 2D tensor backend.
//
// Deliberately written to mirror src/gpu/cuda/kernels.cu line for line: the
// bounds-guard idiom (`if (gid >= n) return;`) is used instead of Metal's
// non-uniform dispatchThreads so that both files keep the same structure and
// stay easy to diff when one is changed.
//
// Compiled offline by CMake with -fno-fast-math. Fast math would reassociate
// and contract these operations, which would break the bitwise CPU/GPU
// agreement the tests rely on.

#include <metal_stdlib>

using namespace metal;

// Must match MatMulParams in metal_backend.cpp. Kept to 32-bit fields so the
// host and device layouts agree without padding surprises.
struct MatMulParams {
  uint m, n, k;
  uint trans_a, trans_b;
  float beta;
};

// Threads per threadgroup for the 1D kernels. The reduction's scratch array is
// sized to this, and the tree below assumes it is a power of two.
constant uint kThreadgroupSize = 256;

/* ---- elementwise binary ---- */

kernel void ew_add(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant uint& n [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = a[gid] + b[gid];
}

kernel void ew_sub(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant uint& n [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = a[gid] - b[gid];
}

kernel void ew_mul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* out [[buffer(2)]],
                   constant uint& n [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = a[gid] * b[gid];
}

// out = alpha * a + beta. Covers every tensor-scalar form the operator
// overloads need. Written as a plain multiply-add, not fma, to mirror
// cpu::affine's expression exactly.
kernel void ew_affine(device const float* a [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant float& alpha [[buffer(2)]],
                      constant float& beta [[buffer(3)]],
                      constant uint& n [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  out[gid] = alpha * a[gid] + beta;
}

/* ---- accumulate-in-place ---- */

// y += alpha * x
kernel void ew_axpy(device const float* x [[buffer(0)]],
                    device float* y [[buffer(1)]],
                    constant float& alpha [[buffer(2)]],
                    constant uint& n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  y[gid] += alpha * x[gid];
}

// y += g * v
kernel void ew_mul_add(device const float* g [[buffer(0)]],
                       device const float* v [[buffer(1)]],
                       device float* y [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  y[gid] += g[gid] * v[gid];
}

// y += alpha  (broadcast of a scalar upstream gradient)
kernel void ew_add_bias(device float* y [[buffer(0)]],
                        constant float& alpha [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  y[gid] += alpha;
}

kernel void ew_fill(device float* y [[buffer(0)]],
                    constant float& v [[buffer(1)]],
                    constant uint& n [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  y[gid] = v;
}

/* ---- reduction ---- */

// One partial per threadgroup; the host sums the partials sequentially.
//
// Deliberately NOT using atomic_fetch_add_explicit on floats: atomics leave the
// summation order up to the scheduler, so the same input would produce
// different bits run to run and no stable test could be written. A fixed grid
// with a fixed tree order is bit-exactly repeatable.
kernel void reduce_sum(device const float* a [[buffer(0)]],
                       device float* partials [[buffer(1)]],
                       constant uint& n [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint tg_id [[threadgroup_position_in_grid]]) {
  threadgroup float scratch[kThreadgroupSize];

  scratch[tid] = (gid < n) ? a[gid] : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint stride = kThreadgroupSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) partials[tg_id] = scratch[0];
}

/* ---- matrix product ---- */

// C(m,n) = op(A)(m,k) * op(B)(k,n) + beta * C(m,n)
//
// The transpose flags are uniform across every thread, so they cost an index
// computation rather than divergence. They exist so matmul backward
// (dA = dC*B^T, dB = A^T*dC) needs no transpose op and no scratch matrices.
kernel void matmul(device const float* a [[buffer(0)]],
                   device const float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   constant MatMulParams& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
  const uint i = gid.y;
  const uint j = gid.x;
  if (i >= p.m || j >= p.n) return;

  float acc = 0.0f;
  for (uint kk = 0; kk < p.k; ++kk) {
    const float av = p.trans_a ? a[kk * p.m + i] : a[i * p.k + kk];
    const float bv = p.trans_b ? b[j * p.k + kk] : b[kk * p.n + j];
    acc += av * bv;
  }

  const uint idx = i * p.n + j;
  // Must not read c[idx] when beta is 0: a pooled buffer holds garbage, and
  // 0 * NaN is NaN rather than 0.
  c[idx] = (p.beta == 0.0f) ? acc : fma(p.beta, c[idx], acc);
}
