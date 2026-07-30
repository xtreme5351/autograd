//
// Created by Pranav C on 30/07/2026.
//
// CPU-vs-GPU crossover benchmark.
//
// The thresholds in include/consts.h ship as order-of-magnitude placeholders.
// This binary is how you replace them with measured numbers: it reports, per
// op and per size, the point at which routing to the GPU actually wins.
//
// Run it on the target machine -- the answer differs sharply between Metal
// (unified memory, so an "upload" is a memcpy) and discrete CUDA (a real PCIe
// transfer), and between fp32-heavy and fp64-heavy hardware.
//
//   cmake --build build --target autograd_bench && ./build/autograd_bench
//
// Deliberately dependency-free (std::chrono, no gbenchmark) so it builds
// anywhere the library does.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "backend.h"
#include "operations.h"
#include "tape.h"
#include "tensor_two.h"

using namespace autograd;
using Clock = std::chrono::steady_clock;

namespace {

std::vector<Scalar> random_data(const size_t n, const uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<Scalar> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

// Median of `reps` timed runs, after a warmup that pays for lazy device and
// pipeline initialisation. Median rather than mean: a single scheduler hiccup
// should not move the number.
template <typename F>
double median_ms(F&& f, const int reps = 7) {
  f();  // warmup
  std::vector<double> samples;
  samples.reserve(reps);
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
    f();
    const auto t1 = Clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void row(const std::string& label, const size_t size, const double cpu_ms,
         const double gpu_ms) {
  const double speedup = (gpu_ms > 0.0) ? cpu_ms / gpu_ms : 0.0;
  std::printf("%-14s %10zu %12.4f %12.4f %9.2fx  %s\n", label.c_str(), size,
              cpu_ms, gpu_ms, speedup, speedup > 1.0 ? "GPU" : "cpu");
}

void header(const char* title) {
  std::printf("\n%s\n", title);
  std::printf("%-14s %10s %12s %12s %10s  %s\n", "op", "size", "cpu ms",
              "gpu ms", "speedup", "winner");
  std::printf("%s\n", std::string(72, '-').c_str());
}

// Calls gpu:: / cpu:: directly rather than going through ops::, so the size
// heuristic under measurement cannot suppress the very path being measured.
void bench_elementwise() {
  header("elementwise (add) -- bandwidth bound, ~1 flop per element");
  for (const size_t n : {1u << 10, 1u << 14, 1u << 16, 1u << 18, 1u << 20,
                         1u << 22, 1u << 24}) {
    const std::vector<Scalar> a = random_data(n, 1), b = random_data(n, 2);
    std::vector<Scalar> out(n);
    const double cpu_ms =
        median_ms([&] { cpu::add(a.data(), b.data(), out.data(), n); });
    const double gpu_ms =
        gpu::available()
            ? median_ms([&] { gpu::add(a.data(), b.data(), out.data(), n); })
            : 0.0;
    row("add", n, cpu_ms, gpu_ms);
  }
}

void bench_mul() {
  header("elementwise (mul)");
  for (const size_t n : {1u << 16, 1u << 20, 1u << 24}) {
    const std::vector<Scalar> a = random_data(n, 3), b = random_data(n, 4);
    std::vector<Scalar> out(n);
    const double cpu_ms =
        median_ms([&] { cpu::mul(a.data(), b.data(), out.data(), n); });
    const double gpu_ms =
        gpu::available()
            ? median_ms([&] { gpu::mul(a.data(), b.data(), out.data(), n); })
            : 0.0;
    row("mul", n, cpu_ms, gpu_ms);
  }
}

void bench_affine() {
  header("elementwise (affine) -- the tensor-scalar path");
  for (const size_t n : {1u << 16, 1u << 20, 1u << 24}) {
    const std::vector<Scalar> a = random_data(n, 5);
    std::vector<Scalar> out(n);
    const double cpu_ms =
        median_ms([&] { cpu::affine(2.0f, a.data(), 1.0f, out.data(), n); });
    const double gpu_ms =
        gpu::available() ? median_ms([&] {
          gpu::affine(2.0f, a.data(), 1.0f, out.data(), n);
        })
                         : 0.0;
    row("affine", n, cpu_ms, gpu_ms);
  }
}

// Square matmul. Reported against m*n*k because that is the unit
// gpu_min_matmul_flops is expressed in.
void bench_matmul() {
  header("matmul (square, n^3 flops on n^2 data) -- report vs m*n*k");
  for (const size_t d : {16u, 32u, 64u, 128u, 256u, 512u}) {
    const std::vector<Scalar> a = random_data(d * d, 6), b = random_data(d * d, 7);
    std::vector<Scalar> c(d * d);
    const MatMulSpec spec{d, d, d, false, false, 0};
    const double cpu_ms =
        median_ms([&] { cpu::matmul(a.data(), b.data(), c.data(), spec); }, 3);
    const double gpu_ms =
        gpu::available() ? median_ms(
                               [&] {
                                 gpu::matmul(a.data(), b.data(), c.data(), spec);
                               },
                               3)
                         : 0.0;
    row("matmul " + std::to_string(d), d * d * d, cpu_ms, gpu_ms);
  }
}

// The point of this one: it measures the per-node host<->device round trip that
// backend.h's host-pointer design forces. A tape with many small nodes is the
// worst case, and it is the case a training loop actually generates.
void bench_tape_program() {
  header("tape program (fwd + bwd), 32 chained ops -- per-node round trip");
  for (const size_t d : {16u, 64u, 128u}) {
    auto run = [&](const Device dev) {
      Tape tape(TensorClass::TENSOR_TWO, 1.0, dev);
      const TensorTwo a(random_data(d * d, 8), Shape2{d, d}, true, &tape);
      const TensorTwo b(random_data(d * d, 9), Shape2{d, d}, true, &tape);
      TensorTwo acc = a * b;
      for (int i = 0; i < 32; ++i) acc = acc + b;
      const TensorTwo loss = mean(acc);
      loss.backward();
    };
    // Force the GPU on: at these sizes the heuristic would otherwise route the
    // "gpu" column to the CPU and the comparison would be meaningless.
    const size_t saved_ew = gpu_min_elementwise;
    const size_t saved_mm = gpu_min_matmul_flops;
    const double cpu_ms = median_ms([&] { run(Device::CPU); }, 3);
    gpu_min_elementwise = 0;
    gpu_min_matmul_flops = 0;
    const double gpu_ms =
        gpu::available() ? median_ms([&] { run(Device::GPU); }, 3) : 0.0;
    gpu_min_elementwise = saved_ew;
    gpu_min_matmul_flops = saved_mm;
    row("tape " + std::to_string(d), d * d, cpu_ms, gpu_ms);
  }
}

}  // namespace

int main() {
  std::printf("cpu backend: %s\n", cpu::name());
  std::printf("gpu backend: %s (available: %s)\n", gpu::name(),
              gpu::available() ? "yes" : "no");
  if (!gpu::available()) {
    std::printf(
        "\nNOTE: no GPU backend compiled in or no device present. The gpu\n"
        "columns below are all zero -- this run measures the CPU only and\n"
        "cannot be used to calibrate the thresholds in include/consts.h.\n");
  }

  bench_elementwise();
  bench_mul();
  bench_affine();
  bench_matmul();
  bench_tape_program();

  std::printf(
      "\nTo calibrate: set gpu_min_elementwise to the element count where the\n"
      "elementwise rows first favour the GPU, and gpu_min_matmul_flops to the\n"
      "m*n*k where the matmul rows do. Both live in include/consts.h.\n");
  return 0;
}
