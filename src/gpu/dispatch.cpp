//
// Created by Pranav C on 29/07/2026.
//
// Runtime device dispatch. Kept in one place so the "GPU requested but not
// available" fallback isn't duplicated across every tape call site.

#include "backend.h"

namespace autograd::ops {

// Both backends stay linked at once, which is what lets a single test binary
// run the same tape program on CPU and GPU and compare the two.
static bool use_gpu(const Device d) {
  return d == Device::GPU && gpu::available();
}

void add(const Device d, const Scalar* a, const Scalar* b, Scalar* out,
         const size_t n) {
  if (use_gpu(d)) return gpu::add(a, b, out, n);
  return cpu::add(a, b, out, n);
}

void sub(const Device d, const Scalar* a, const Scalar* b, Scalar* out,
         const size_t n) {
  if (use_gpu(d)) return gpu::sub(a, b, out, n);
  return cpu::sub(a, b, out, n);
}

void mul(const Device d, const Scalar* a, const Scalar* b, Scalar* out,
         const size_t n) {
  if (use_gpu(d)) return gpu::mul(a, b, out, n);
  return cpu::mul(a, b, out, n);
}

void axpy(const Device d, const Scalar alpha, const Scalar* x, Scalar* y,
          const size_t n) {
  if (use_gpu(d)) return gpu::axpy(alpha, x, y, n);
  return cpu::axpy(alpha, x, y, n);
}

void mul_add(const Device d, const Scalar* g, const Scalar* v, Scalar* y,
             const size_t n) {
  if (use_gpu(d)) return gpu::mul_add(g, v, y, n);
  return cpu::mul_add(g, v, y, n);
}

void add_bias(const Device d, const Scalar alpha, Scalar* y, const size_t n) {
  if (use_gpu(d)) return gpu::add_bias(alpha, y, n);
  return cpu::add_bias(alpha, y, n);
}

void fill(const Device d, const Scalar v, Scalar* y, const size_t n) {
  if (use_gpu(d)) return gpu::fill(v, y, n);
  return cpu::fill(v, y, n);
}

Scalar sum(const Device d, const Scalar* a, const size_t n) {
  if (use_gpu(d)) return gpu::sum(a, n);
  return cpu::sum(a, n);
}

void matmul(const Device d, const Scalar* a, const Scalar* b, Scalar* c,
            const MatMulSpec& s) {
  if (use_gpu(d)) return gpu::matmul(a, b, c, s);
  return cpu::matmul(a, b, c, s);
}

}  // namespace autograd::ops
