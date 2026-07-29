//
// Created by Pranav C on 29/07/2026.
//
// Reference CPU implementation of the backend primitives.
//
// Accumulation stays in Scalar (float) rather than widening to double. A more
// accurate CPU path would disagree with the GPU by more than the GPU's own
// reassociation error, which would make the cross-check test measure precision
// instead of backend agreement. Absolute accuracy is the job of the
// double-precision reference in the finite-difference test.

#include "backend.h"

namespace autograd::cpu {

bool available() { return true; }

const char* name() { return "cpu"; }

void add(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

void sub(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] - b[i];
}

void mul(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
}

void axpy(const Scalar alpha, const Scalar* x, Scalar* y, const size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

void mul_add(const Scalar* g, const Scalar* v, Scalar* y, const size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] += g[i] * v[i];
}

void add_bias(const Scalar alpha, Scalar* y, const size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] += alpha;
}

void fill(const Scalar v, Scalar* y, const size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] = v;
}

Scalar sum(const Scalar* a, const size_t n) {
  Scalar acc = 0;
  for (size_t i = 0; i < n; ++i) acc += a[i];
  return acc;
}

void matmul(const Scalar* a, const Scalar* b, Scalar* c, const MatMulSpec& s) {
  for (size_t i = 0; i < s.m; ++i) {
    for (size_t j = 0; j < s.n; ++j) {
      Scalar acc = 0;
      for (size_t kk = 0; kk < s.k; ++kk) {
        const Scalar av = s.trans_a ? a[kk * s.m + i] : a[i * s.k + kk];
        const Scalar bv = s.trans_b ? b[j * s.k + kk] : b[kk * s.n + j];
        acc += av * bv;
      }
      // Must not read c[] when beta is 0: freshly allocated GPU memory holds
      // garbage, and 0 * NaN is NaN, not 0.
      Scalar* dst = &c[i * s.n + j];
      *dst = (s.beta == 0) ? acc : s.beta * *dst + acc;
    }
  }
}

}  // namespace autograd::cpu
