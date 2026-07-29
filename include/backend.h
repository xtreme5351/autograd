//
// Created by Pranav C on 29/07/2026.
//
// Compute backends for 2D tensor work.
//
// Three namespaces expose the identical set of nine primitives:
//   cpu::  the reference implementation, always compiled
//   gpu::  Metal or CUDA, whichever CMake selected (exactly one)
//   ops::  the dispatcher; takes a Device and forwards to one of the above
//
// Everything moves through host pointers rather than device handles. That is
// what lets Metal (memcpy into a shared buffer) and CUDA (cudaMemcpy) sit
// behind one signature, and it keeps tape storage as plain std::vector so
// value()/grad() need no sync step.
//
// IMPORTANT: nvcc compiles this header, so it must stay C++17-clean. All C++20
// lives in tape.h / tensor_two.h, which the CUDA build never sees.

#ifndef AUTOGRAD_BACKEND_H
#define AUTOGRAD_BACKEND_H

#include <cstddef>

#include "consts.h"

namespace autograd {

// C(m,n) = op(A)(m,k) * op(B)(k,n) + beta * C(m,n)
//
// trans_a: A is stored (k,m) row-major instead of (m,k)
// trans_b: B is stored (n,k) row-major instead of (k,n)
//
// The transpose flags exist so matmul backward (dA = dC*B^T, dB = A^T*dC) needs
// no transpose op and allocates no intermediate matrices. beta plays the BLAS
// gemm role: 0 overwrites C for a forward pass, 1 accumulates into it for a
// backward pass.
struct MatMulSpec {
  size_t m, n, k;
  bool trans_a, trans_b;
  Scalar beta;
};

namespace cpu {
bool available();
const char* name();

void add(const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void sub(const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void mul(const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void axpy(Scalar alpha, const Scalar* x, Scalar* y, size_t n);     // y += a*x
void mul_add(const Scalar* g, const Scalar* v, Scalar* y, size_t n);  // y += g*v
void add_bias(Scalar alpha, Scalar* y, size_t n);                  // y += alpha
void fill(Scalar v, Scalar* y, size_t n);
Scalar sum(const Scalar* a, size_t n);
void matmul(const Scalar* a, const Scalar* b, Scalar* c, const MatMulSpec& s);
}  // namespace cpu

// Same surface as cpu::. Exactly one implementation is linked -- Metal, CUDA,
// or the stub that reports unavailable.
namespace gpu {
bool available();
const char* name();

void add(const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void sub(const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void mul(const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void axpy(Scalar alpha, const Scalar* x, Scalar* y, size_t n);
void mul_add(const Scalar* g, const Scalar* v, Scalar* y, size_t n);
void add_bias(Scalar alpha, Scalar* y, size_t n);
void fill(Scalar v, Scalar* y, size_t n);
Scalar sum(const Scalar* a, size_t n);
void matmul(const Scalar* a, const Scalar* b, Scalar* c, const MatMulSpec& s);
}  // namespace gpu

// The only entry points the tape calls. A Device::GPU request falls back to
// the CPU when no GPU backend is available, so asking for a GPU on a machine
// without one degrades instead of crashing.
namespace ops {
void add(Device d, const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void sub(Device d, const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void mul(Device d, const Scalar* a, const Scalar* b, Scalar* out, size_t n);
void axpy(Device d, Scalar alpha, const Scalar* x, Scalar* y, size_t n);
void mul_add(Device d, const Scalar* g, const Scalar* v, Scalar* y, size_t n);
void add_bias(Device d, Scalar alpha, Scalar* y, size_t n);
void fill(Device d, Scalar v, Scalar* y, size_t n);
Scalar sum(Device d, const Scalar* a, size_t n);
void matmul(Device d, const Scalar* a, const Scalar* b, Scalar* c,
            const MatMulSpec& s);
}  // namespace ops

}  // namespace autograd

#endif  // AUTOGRAD_BACKEND_H
