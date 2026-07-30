//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_CONSTS_H
#define AUTOGRAD_CONSTS_H

namespace autograd {
enum NodeType { ADD, SUB, MUL, DIV, CONST, MATMUL, SUM, MEAN };
enum TensorClass { TENSOR_ZERO, TENSOR_ONE, TENSOR_TWO };

// Which backend a tape's ops run on. Selected per tape, not globally, so a
// single process can run both and cross-check them against each other.
enum class Device { CPU, GPU };

// Scalar type for 2D tensors. float rather than double because Metal Shading
// Language has no float64 so the CPU reference has to agree with the GPU.
using Scalar = float;

// The Tape constructor's default argument
inline Device default_device = Device::CPU;

// ---- size-aware dispatch thresholds ----
//
// A Device::GPU request only reaches the GPU once the problem is big enough to
// pay for the host<->device round trip the backends do per op (see the comment
// in backend.h). Below these, ops:: silently runs on the CPU, which is faster.
//
// Elementwise work is bandwidth-bound with arithmetic intensity ~1 flop per
// element, so it has to be very large before a transfer plus a launch beats an
// L1-resident loop. Matmul does m*n*k flops on m*k + k*n elements, so its
// intensity grows with k and it crosses over far earlier -- which is why the
// two are counted in different units: elements for elementwise, flops (m*n*k)
// for matmul.
//
// IMPORTANT: these defaults are order-of-magnitude placeholders, NOT measured.
// Run benchmarks/bench.cpp (target: autograd_bench) on the target machine and
// replace them -- the crossover differs sharply between Metal (unified memory,
// a memcpy) and discrete CUDA (a PCIe transfer). Set either to 0 to force the
// GPU path regardless of size; the cross-check tests do exactly that.
inline size_t gpu_min_elementwise = 1u << 20;  // 1M elements
inline size_t gpu_min_matmul_flops = 1u << 18;  // e.g. 64x64x64
}  // namespace autograd

#endif  // AUTOGRAD_CONSTS_H
