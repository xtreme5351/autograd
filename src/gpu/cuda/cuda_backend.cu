//
// Created by Pranav C on 29/07/2026.
//
// CUDA implementation of the gpu:: backend.
//
// A deliberate line-for-line port of src/gpu/metal/kernels.metal and
// metal_backend.cpp: same kernels, same bounds-guard idiom, same per-op host
// staging, same buffer pool. Keeping the two structurally identical is what
// makes it practical to change one and mirror it into the other.
//
// Compiled at C++17 (CMAKE_CUDA_STANDARD 17) because nvcc's C++20 support is
// partial. backend.h is the only project header this sees and is kept
// C++17-clean for exactly that reason.
//
// Do not build this with -use_fast_math: it would reassociate and contract the
// arithmetic, breaking the bitwise CPU/GPU agreement the tests assert.
// --fmad=false is worth adding if the strict comparisons ever drift.
//
// Not thread-safe, matching the Metal backend and the tape's single-threaded,
// synchronous use.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"

namespace autograd::gpu {
namespace {

// Matches MatMulParams in kernels.metal.
struct MatMulParams {
  unsigned m, n, k;
  unsigned trans_a, trans_b;
  float beta;
};

constexpr unsigned kBlockSize = 256;  // must match kThreadgroupSize
constexpr unsigned kTileSize = 16;    // 16x16 = 256 threads for matmul

/* ---- kernels (mirror kernels.metal) ---- */

__global__ void ew_add(const float* a, const float* b, float* out,
                       const unsigned n) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  out[gid] = a[gid] + b[gid];
}

__global__ void ew_sub(const float* a, const float* b, float* out,
                       const unsigned n) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  out[gid] = a[gid] - b[gid];
}

__global__ void ew_mul(const float* a, const float* b, float* out,
                       const unsigned n) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  out[gid] = a[gid] * b[gid];
}

__global__ void ew_axpy(const float* x, float* y, const float alpha,
                        const unsigned n) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  y[gid] += alpha * x[gid];
}

__global__ void ew_mul_add(const float* g, const float* v, float* y,
                           const unsigned n) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  y[gid] += g[gid] * v[gid];
}

__global__ void ew_add_bias(float* y, const float alpha, const unsigned n) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  y[gid] += alpha;
}

__global__ void ew_fill(float* y, const float v, const unsigned n) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  y[gid] = v;
}

// One partial per block; the host folds the partials sequentially.
//
// Deliberately NOT atomicAdd on floats: atomics leave the summation order to
// the scheduler, so identical input would give different bits run to run and
// GpuBackend.SumIsBitwiseRepeatable could not hold.
__global__ void reduce_sum(const float* a, float* partials,
                           const unsigned n) {
  __shared__ float scratch[kBlockSize];
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned tid = threadIdx.x;

  scratch[tid] = (gid < n) ? a[gid] : 0.0f;
  __syncthreads();

  for (unsigned stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }

  if (tid == 0) partials[blockIdx.x] = scratch[0];
}

// C(m,n) = op(A)(m,k) * op(B)(k,n) + beta * C(m,n)
__global__ void matmul_kernel(const float* a, const float* b, float* c,
                              const MatMulParams p) {
  const unsigned i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.m || j >= p.n) return;

  float acc = 0.0f;
  for (unsigned kk = 0; kk < p.k; ++kk) {
    const float av = p.trans_a ? a[kk * p.m + i] : a[i * p.k + kk];
    const float bv = p.trans_b ? b[j * p.k + kk] : b[kk * p.n + j];
    acc += av * bv;
  }

  const unsigned idx = i * p.n + j;
  // Must not read c[idx] when beta is 0: a pooled allocation holds garbage,
  // and 0 * NaN is NaN rather than 0.
  c[idx] = (p.beta == 0.0f) ? acc : fmaf(p.beta, c[idx], acc);
}

/* ---- host side ---- */

// Reuses device allocations across ops. cudaMalloc is expensive and the tape
// issues several per node; at 50k nodes the allocation churn dominates.
class BufferPool {
 public:
  float* acquire(const size_t bytes) {
    const size_t bucket = round_up(bytes);
    auto& bin = free_[bucket];
    if (!bin.empty()) {
      float* p = bin.back();
      bin.pop_back();
      return p;
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bucket) != cudaSuccess) return nullptr;
    sizes_[p] = bucket;
    return static_cast<float*>(p);
  }

  void release(float* p) {
    if (p == nullptr) return;
    free_[sizes_[p]].push_back(p);
  }

 private:
  static size_t round_up(const size_t bytes) {
    size_t v = 1;
    while (v < bytes) v <<= 1;
    return v;
  }

  std::unordered_map<size_t, std::vector<float*>> free_;
  std::unordered_map<void*, size_t> sizes_;
};

struct CudaState {
  bool initialised = false;
  bool ok = false;
  std::string device_name = "none";
  BufferPool pool;
};

CudaState& state() {
  static CudaState s;
  if (s.initialised) return s;
  s.initialised = true;

  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
    return s;  // ok stays false -> everything falls back to the CPU
  }
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return s;

  s.device_name = prop.name;
  s.ok = true;
  return s;
}

// Owns a device allocation for the duration of one op, mirroring the Metal
// backend's Dispatch. Host staging in, kernel, host staging out.
class DeviceBuffer {
 public:
  DeviceBuffer(const size_t n, const Scalar* upload) : n_(n) {
    ptr_ = state().pool.acquire(n * sizeof(float));
    if (ptr_ != nullptr && upload != nullptr) {
      cudaMemcpy(ptr_, upload, n * sizeof(float), cudaMemcpyHostToDevice);
    }
  }

  ~DeviceBuffer() { state().pool.release(ptr_); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void download(Scalar* dst) const {
    cudaMemcpy(dst, ptr_, n_ * sizeof(float), cudaMemcpyDeviceToHost);
  }

  float* get() const { return ptr_; }
  bool valid() const { return ptr_ != nullptr; }

 private:
  float* ptr_ = nullptr;
  size_t n_ = 0;
};

unsigned grid_1d(const size_t n) {
  return static_cast<unsigned>((n + kBlockSize - 1) / kBlockSize);
}

}  // namespace

bool available() { return state().ok; }

const char* name() { return state().device_name.c_str(); }

void add(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  if (n == 0) return;
  const DeviceBuffer da(n, a), db(n, b), dout(n, nullptr);
  ew_add<<<grid_1d(n), kBlockSize>>>(da.get(), db.get(), dout.get(),
                                     static_cast<unsigned>(n));
  cudaDeviceSynchronize();
  dout.download(out);
}

void sub(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  if (n == 0) return;
  const DeviceBuffer da(n, a), db(n, b), dout(n, nullptr);
  ew_sub<<<grid_1d(n), kBlockSize>>>(da.get(), db.get(), dout.get(),
                                     static_cast<unsigned>(n));
  cudaDeviceSynchronize();
  dout.download(out);
}

void mul(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  if (n == 0) return;
  const DeviceBuffer da(n, a), db(n, b), dout(n, nullptr);
  ew_mul<<<grid_1d(n), kBlockSize>>>(da.get(), db.get(), dout.get(),
                                     static_cast<unsigned>(n));
  cudaDeviceSynchronize();
  dout.download(out);
}

void axpy(const Scalar alpha, const Scalar* x, Scalar* y, const size_t n) {
  if (n == 0) return;
  const DeviceBuffer dx(n, x), dy(n, y);  // dy is seeded: the kernel reads it
  ew_axpy<<<grid_1d(n), kBlockSize>>>(dx.get(), dy.get(), alpha,
                                      static_cast<unsigned>(n));
  cudaDeviceSynchronize();
  dy.download(y);
}

void mul_add(const Scalar* g, const Scalar* v, Scalar* y, const size_t n) {
  if (n == 0) return;
  const DeviceBuffer dg(n, g), dv(n, v), dy(n, y);
  ew_mul_add<<<grid_1d(n), kBlockSize>>>(dg.get(), dv.get(), dy.get(),
                                         static_cast<unsigned>(n));
  cudaDeviceSynchronize();
  dy.download(y);
}

void add_bias(const Scalar alpha, Scalar* y, const size_t n) {
  if (n == 0) return;
  const DeviceBuffer dy(n, y);
  ew_add_bias<<<grid_1d(n), kBlockSize>>>(dy.get(), alpha,
                                          static_cast<unsigned>(n));
  cudaDeviceSynchronize();
  dy.download(y);
}

void fill(const Scalar v, Scalar* y, const size_t n) {
  if (n == 0) return;
  const DeviceBuffer dy(n, nullptr);  // every element is written unconditionally
  ew_fill<<<grid_1d(n), kBlockSize>>>(dy.get(), v, static_cast<unsigned>(n));
  cudaDeviceSynchronize();
  dy.download(y);
}

Scalar sum(const Scalar* a, const size_t n) {
  if (n == 0) return 0;
  const unsigned blocks = grid_1d(n);

  std::vector<Scalar> partials(blocks);
  {
    const DeviceBuffer da(n, a), dp(blocks, nullptr);
    reduce_sum<<<blocks, kBlockSize>>>(da.get(), dp.get(),
                                       static_cast<unsigned>(n));
    cudaDeviceSynchronize();
    dp.download(partials.data());
  }

  // Fixed grid -> fixed partition -> fixed tree order on device, and a
  // sequential fold here. The whole reduction is bit-exactly repeatable.
  Scalar total = 0;
  for (const Scalar p : partials) total += p;
  return total;
}

void matmul(const Scalar* a, const Scalar* b, Scalar* c, const MatMulSpec& s) {
  const size_t c_elems = s.m * s.n;
  if (c_elems == 0) return;

  const DeviceBuffer da(s.m * s.k, a), db(s.k * s.n, b);
  // beta != 0 means the kernel reads C, so it has to be seeded from the host.
  const DeviceBuffer dc(c_elems, (s.beta == 0) ? nullptr : c);

  const MatMulParams p{
      static_cast<unsigned>(s.m),       static_cast<unsigned>(s.n),
      static_cast<unsigned>(s.k),       static_cast<unsigned>(s.trans_a),
      static_cast<unsigned>(s.trans_b), s.beta};

  const dim3 block(kTileSize, kTileSize);
  const dim3 grid(static_cast<unsigned>((s.n + kTileSize - 1) / kTileSize),
                  static_cast<unsigned>((s.m + kTileSize - 1) / kTileSize));
  matmul_kernel<<<grid, block>>>(da.get(), db.get(), dc.get(), p);
  cudaDeviceSynchronize();
  dc.download(c);
}

}  // namespace autograd::gpu
