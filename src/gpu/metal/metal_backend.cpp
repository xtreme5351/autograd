//
// Created by Pranav C on 29/07/2026.
//
// Metal implementation of the gpu:: backend, via metal-cpp.
//
// Two things drive the shape of this file:
//
//   1. metal-cpp objects are NOT ARC-managed. Anything returned by a method
//      beginning alloc/new/copy/Create is owned and has to be released, so
//      every long-lived handle is an NS::SharedPtr. Command buffers and
//      encoders are autoreleased, which in a C++ program means they leak
//      unless something drains a pool -- hence the NS::AutoreleasePool in
//      Dispatch's constructor. All nine entry points funnel through it so the
//      pool cannot be forgotten per op.
//
//   2. No Metal type may appear in a header. All state lives here behind a
//      function-local static, and backend.h stays plain C++.
//
// Not thread-safe: the lazy init and the buffer pool both assume a single
// caller, which matches the tape's synchronous, single-threaded use.

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"

#ifndef AUTOGRAD_METALLIB_PATH
#error "AUTOGRAD_METALLIB_PATH must be defined by the build"
#endif

namespace autograd::gpu {
namespace {

// Matches MatMulParams in kernels.metal. 32-bit fields so host and device
// layouts agree without padding surprises.
struct MatMulParams {
  uint32_t m, n, k;
  uint32_t trans_a, trans_b;
  float beta;
};

constexpr uint32_t kThreadgroupSize = 256;  // must match kernels.metal
constexpr uint32_t kTileSize = 16;          // 16x16 = 256 threads for matmul

const char* const kKernelNames[] = {
    "ew_add",      "ew_sub",  "ew_mul",     "ew_affine", "ew_axpy",
    "ew_mul_add",  "ew_fill", "ew_add_bias", "reduce_sum", "matmul"};

// Reuses buffers across ops. newBuffer is a real allocation, and the tape
// issues a handful per node -- at 50k nodes the allocation churn dominates.
class BufferPool {
 public:
  NS::SharedPtr<MTL::Buffer> acquire(MTL::Device* device, const size_t bytes) {
    const size_t bucket = round_up(bytes);
    auto& bin = free_[bucket];
    if (!bin.empty()) {
      NS::SharedPtr<MTL::Buffer> buf = bin.back();
      bin.pop_back();
      return buf;
    }
    return NS::TransferPtr(
        device->newBuffer(bucket, MTL::ResourceStorageModeShared));
  }

  void release(NS::SharedPtr<MTL::Buffer> buf) {
    if (!buf) return;
    free_[buf->length()].push_back(std::move(buf));
  }

 private:
  // Power-of-two buckets keep the map small; a 1-byte floor avoids a zero-size
  // allocation for empty tensors.
  static size_t round_up(const size_t bytes) {
    size_t v = 1;
    while (v < bytes) v <<= 1;
    return v;
  }

  std::unordered_map<size_t, std::vector<NS::SharedPtr<MTL::Buffer>>> free_;
};

struct MetalState {
  bool initialised = false;
  bool ok = false;
  std::string device_name = "none";

  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::CommandQueue> queue;
  NS::SharedPtr<MTL::Library> library;
  std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>>
      pipelines;
  BufferPool pool;
};

MetalState& state() {
  static MetalState s;
  if (s.initialised) return s;
  s.initialised = true;

  // Init itself creates autoreleased objects (NS::String, NS::URL, NS::Error).
  const NS::SharedPtr<NS::AutoreleasePool> pool =
      NS::TransferPtr(NS::AutoreleasePool::alloc()->init());

  s.device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!s.device) return s;  // ok stays false -> everything falls back to CPU

  s.device_name = s.device->name()->utf8String();

  s.queue = NS::TransferPtr(s.device->newCommandQueue());
  if (!s.queue) return s;

  // The metallib is built offline by CMake, so a kernel typo is a build error
  // rather than a failure here. AUTOGRAD_METALLIB_PATH is absolute, so the
  // working directory does not matter.
  NS::Error* error = nullptr;
  const NS::String* path =
      NS::String::string(AUTOGRAD_METALLIB_PATH, NS::UTF8StringEncoding);
  s.library = NS::TransferPtr(s.device->newLibrary(NS::URL::fileURLWithPath(path), &error));
  if (!s.library) return s;

  for (const char* name : kKernelNames) {
    const NS::SharedPtr<MTL::Function> fn = NS::TransferPtr(
        s.library->newFunction(NS::String::string(name, NS::UTF8StringEncoding)));
    if (!fn) return s;
    NS::SharedPtr<MTL::ComputePipelineState> pso =
        NS::TransferPtr(s.device->newComputePipelineState(fn.get(), &error));
    if (!pso) return s;
    s.pipelines.emplace(name, std::move(pso));
  }

  s.ok = true;
  return s;
}

// One encode/commit/wait cycle, with the autorelease pool that keeps the
// command buffer and encoder from leaking. Synchronous, matching the tape's
// eager semantics where values_two[id] is read immediately after the op.
class Dispatch {
 public:
  Dispatch(MetalState& s, const char* kernel)
      : pool_(NS::TransferPtr(NS::AutoreleasePool::alloc()->init())), state_(s) {
    cmd_ = s.queue->commandBuffer();
    enc_ = cmd_->computeCommandEncoder();
    enc_->setComputePipelineState(s.pipelines.at(kernel).get());
  }

  ~Dispatch() {
    for (auto& b : owned_) state_.pool.release(std::move(b));
  }

  Dispatch(const Dispatch&) = delete;
  Dispatch& operator=(const Dispatch&) = delete;

  // Uploads n floats and binds them. On M1 the shared storage mode means this
  // memcpy is the whole "upload" -- one physical allocation, no PCIe transfer.
  void in(const Scalar* src, const size_t n, const uint32_t index) {
    NS::SharedPtr<MTL::Buffer> buf = state_.pool.acquire(
        state_.device.get(), n * sizeof(Scalar));
    std::memcpy(buf->contents(), src, n * sizeof(Scalar));
    enc_->setBuffer(buf.get(), 0, index);
    owned_.push_back(std::move(buf));
  }

  // Bind a buffer that is read back afterwards. Seeded with the current host
  // contents because the accumulate kernels (axpy, mul_add, add_bias, and
  // matmul with beta=1) read their destination.
  size_t inout(const Scalar* src, const size_t n, const uint32_t index) {
    NS::SharedPtr<MTL::Buffer> buf = state_.pool.acquire(
        state_.device.get(), n * sizeof(Scalar));
    std::memcpy(buf->contents(), src, n * sizeof(Scalar));
    enc_->setBuffer(buf.get(), 0, index);
    owned_.push_back(std::move(buf));
    return owned_.size() - 1;
  }

  // Bind an output-only buffer. Left uninitialised: every kernel using this
  // writes each element unconditionally.
  size_t out(const size_t n, const uint32_t index) {
    NS::SharedPtr<MTL::Buffer> buf = state_.pool.acquire(
        state_.device.get(), n * sizeof(Scalar));
    enc_->setBuffer(buf.get(), 0, index);
    owned_.push_back(std::move(buf));
    return owned_.size() - 1;
  }

  template <typename T>
  void bytes(const T& value, const uint32_t index) {
    enc_->setBytes(&value, sizeof(T), index);
  }

  void run_1d(const size_t n) {
    const size_t groups = (n + kThreadgroupSize - 1) / kThreadgroupSize;
    enc_->dispatchThreadgroups(MTL::Size::Make(groups, 1, 1),
                               MTL::Size::Make(kThreadgroupSize, 1, 1));
    finish();
  }

  void run_2d(const size_t rows, const size_t cols) {
    const size_t gx = (cols + kTileSize - 1) / kTileSize;
    const size_t gy = (rows + kTileSize - 1) / kTileSize;
    enc_->dispatchThreadgroups(MTL::Size::Make(gx, gy, 1),
                               MTL::Size::Make(kTileSize, kTileSize, 1));
    finish();
  }

  void read_back(const size_t handle, Scalar* dst, const size_t n) const {
    std::memcpy(dst, owned_[handle]->contents(), n * sizeof(Scalar));
  }

 private:
  void finish() {
    enc_->endEncoding();
    cmd_->commit();
    cmd_->waitUntilCompleted();
  }

  NS::SharedPtr<NS::AutoreleasePool> pool_;
  MetalState& state_;
  MTL::CommandBuffer* cmd_ = nullptr;          // autoreleased
  MTL::ComputeCommandEncoder* enc_ = nullptr;  // autoreleased
  std::vector<NS::SharedPtr<MTL::Buffer>> owned_;
};

// Shared body for add/sub/mul, which differ only by kernel name.
void binary_op(const char* kernel, const Scalar* a, const Scalar* b,
               Scalar* out, const size_t n) {
  if (n == 0) return;
  Dispatch d(state(), kernel);
  d.in(a, n, 0);
  d.in(b, n, 1);
  const size_t o = d.out(n, 2);
  d.bytes(static_cast<uint32_t>(n), 3);
  d.run_1d(n);
  d.read_back(o, out, n);
}

}  // namespace

bool available() { return state().ok; }

const char* name() { return state().device_name.c_str(); }

void add(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  binary_op("ew_add", a, b, out, n);
}

void sub(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  binary_op("ew_sub", a, b, out, n);
}

void mul(const Scalar* a, const Scalar* b, Scalar* out, const size_t n) {
  binary_op("ew_mul", a, b, out, n);
}

void affine(const Scalar alpha, const Scalar* a, const Scalar beta, Scalar* out,
            const size_t n) {
  if (n == 0) return;
  // d.in copies a into its own buffer and d.out allocates a separate one, so
  // out == a is safe here.
  Dispatch d(state(), "ew_affine");
  d.in(a, n, 0);
  const size_t o = d.out(n, 1);
  d.bytes(alpha, 2);
  d.bytes(beta, 3);
  d.bytes(static_cast<uint32_t>(n), 4);
  d.run_1d(n);
  d.read_back(o, out, n);
}

void axpy(const Scalar alpha, const Scalar* x, Scalar* y, const size_t n) {
  if (n == 0) return;
  Dispatch d(state(), "ew_axpy");
  d.in(x, n, 0);
  const size_t o = d.inout(y, n, 1);
  d.bytes(alpha, 2);
  d.bytes(static_cast<uint32_t>(n), 3);
  d.run_1d(n);
  d.read_back(o, y, n);
}

void mul_add(const Scalar* g, const Scalar* v, Scalar* y, const size_t n) {
  if (n == 0) return;
  Dispatch d(state(), "ew_mul_add");
  d.in(g, n, 0);
  d.in(v, n, 1);
  const size_t o = d.inout(y, n, 2);
  d.bytes(static_cast<uint32_t>(n), 3);
  d.run_1d(n);
  d.read_back(o, y, n);
}

void add_bias(const Scalar alpha, Scalar* y, const size_t n) {
  if (n == 0) return;
  Dispatch d(state(), "ew_add_bias");
  const size_t o = d.inout(y, n, 0);
  d.bytes(alpha, 1);
  d.bytes(static_cast<uint32_t>(n), 2);
  d.run_1d(n);
  d.read_back(o, y, n);
}

void fill(const Scalar v, Scalar* y, const size_t n) {
  if (n == 0) return;
  Dispatch d(state(), "ew_fill");
  const size_t o = d.out(n, 0);
  d.bytes(v, 1);
  d.bytes(static_cast<uint32_t>(n), 2);
  d.run_1d(n);
  d.read_back(o, y, n);
}

Scalar sum(const Scalar* a, const size_t n) {
  if (n == 0) return 0;
  const size_t groups = (n + kThreadgroupSize - 1) / kThreadgroupSize;

  std::vector<Scalar> partials(groups);
  {
    Dispatch d(state(), "reduce_sum");
    d.in(a, n, 0);
    const size_t o = d.out(groups, 1);
    d.bytes(static_cast<uint32_t>(n), 2);
    d.run_1d(n);
    d.read_back(o, partials.data(), groups);
  }

  // Fixed grid -> fixed partition -> fixed tree order on device, and a
  // sequential fold here. The whole reduction is bit-exactly repeatable.
  Scalar total = 0;
  for (const Scalar p : partials) total += p;
  return total;
}

void matmul(const Scalar* a, const Scalar* b, Scalar* c, const MatMulSpec& s) {
  const size_t a_elems = s.m * s.k;
  const size_t b_elems = s.k * s.n;
  const size_t c_elems = s.m * s.n;
  if (c_elems == 0) return;

  Dispatch d(state(), "matmul");
  d.in(a, a_elems, 0);
  d.in(b, b_elems, 1);
  // beta != 0 means the kernel reads C, so it has to be seeded from the host.
  const size_t o = (s.beta == 0) ? d.out(c_elems, 2) : d.inout(c, c_elems, 2);
  const MatMulParams p{
      static_cast<uint32_t>(s.m),       static_cast<uint32_t>(s.n),
      static_cast<uint32_t>(s.k),       static_cast<uint32_t>(s.trans_a),
      static_cast<uint32_t>(s.trans_b), s.beta};
  d.bytes(p, 3);
  d.run_2d(s.m, s.n);
  d.read_back(o, c, c_elems);
}

}  // namespace autograd::gpu
