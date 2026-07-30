//
// Created by Pranav C on 30/07/2026.
//
// Test-only helper for the size-aware dispatch heuristic.

#ifndef AUTOGRAD_TESTS_DEVICE_GUARD_H
#define AUTOGRAD_TESTS_DEVICE_GUARD_H

#include "../../include/consts.h"

namespace autograd_tests {

// Disables the size thresholds in consts.h for the duration of a scope.
//
// Tests use deliberately small tensors so their expected values can be written
// out by hand, but those sizes sit far below the thresholds -- so a
// Device::GPU tensor would quietly run on the CPU and any CPU-vs-GPU
// comparison would pass while exercising no kernel at all. Any test that
// compares devices through ops:: or the tape must hold one of these.
//
// Tests that call gpu:: directly bypass the dispatcher and do not need it.
struct ForceGpu {
  size_t saved_elementwise = autograd::gpu_min_elementwise;
  size_t saved_matmul = autograd::gpu_min_matmul_flops;

  ForceGpu() {
    autograd::gpu_min_elementwise = 0;
    autograd::gpu_min_matmul_flops = 0;
  }
  ~ForceGpu() {
    autograd::gpu_min_elementwise = saved_elementwise;
    autograd::gpu_min_matmul_flops = saved_matmul;
  }

  ForceGpu(const ForceGpu&) = delete;
  ForceGpu& operator=(const ForceGpu&) = delete;
};

}  // namespace autograd_tests

#endif  // AUTOGRAD_TESTS_DEVICE_GUARD_H
