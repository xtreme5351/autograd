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
}  // namespace autograd

#endif  // AUTOGRAD_CONSTS_H
