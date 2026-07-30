//
// Created by Pranav C on 29/07/2026.
//
// Linked when CMake selects no GPU backend. available() reports false, so
// ops:: never routes here -- the forwards below exist only to keep the gpu::
// symbol table complete for anything that links against it directly.

#include "backend.h"

namespace autograd::gpu {

bool available() { return false; }

const char* name() { return "none"; }

void add(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  cpu::add(a, b, out, n);
}

void sub(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  cpu::sub(a, b, out, n);
}

void mul(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  cpu::mul(a, b, out, n);
}

void affine(const Scalar alpha, const Scalar* a, const Scalar beta, Scalar* out,
            const size_t n) {
  cpu::affine(alpha, a, beta, out, n);
}

void axpy(const Scalar alpha, const Scalar* x, Scalar* y, const size_t n) {
  cpu::axpy(alpha, x, y, n);
}

void mul_add(const Scalar* g, const Scalar* v, Scalar* y, const size_t n) {
  cpu::mul_add(g, v, y, n);
}

void add_bias(const Scalar alpha, Scalar* y, const size_t n) {
  cpu::add_bias(alpha, y, n);
}

void fill(const Scalar v, Scalar* y, const size_t n) { cpu::fill(v, y, n); }

Scalar sum(const Scalar* a, const size_t n) { return cpu::sum(a, n); }

void matmul(const Scalar* a, const Scalar* b, Scalar* c, const MatMulSpec& s) {
  cpu::matmul(a, b, c, s);
}

}  // namespace autograd::gpu
