//
// Created by Pranav C on 29/07/2026.
//
// CPU-vs-GPU agreement tests.
//
// Both backends stay linked simultaneously (see src/gpu/dispatch.cpp), which
// is what lets one binary run the same tape program twice and compare. When no
// GPU backend is compiled in, these skip rather than fail.
//
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>

#include "../../include/backend.h"
#include "../../include/operations.h"
#include "../../include/tape.h"
#include "../../include/tensor_two.h"

using namespace autograd;

namespace autograd_tests {
namespace {

constexpr size_t kBig = 4096;  // spans many threadgroups

bool gpu_ready() { return gpu::available(); }

// Deterministic pseudorandom data, so a failure is reproducible.
std::vector<Scalar> random_data(const size_t n, const uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<Scalar> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
  return v;
}

// Mixed relative/absolute tolerance. The +1 floor matters: gradients after
// cancellation land near zero, where a pure relative bound is unmeetable.
void expect_close(const Scalar got, const Scalar want) {
  EXPECT_NEAR(got, want, 1e-5f * (std::abs(want) + 1.0f));
}

bool bitwise_equal(const Scalar a, const Scalar b) {
  return std::memcmp(&a, &b, sizeof(Scalar)) == 0;
}

}  // namespace

TEST(GpuBackend, ReportsAvailability) {
  // Not an assertion about which backend is present -- just makes the selected
  // device visible in the test log.
  GTEST_LOG_(INFO) << "cpu backend: " << cpu::name()
                   << ", gpu backend: " << gpu::name()
                   << ", gpu available: " << (gpu::available() ? "yes" : "no");
  SUCCEED();
}

// With fast math disabled, each output element of an elementwise op is a single
// IEEE operation, so the GPU must match the CPU exactly -- not just closely.
TEST(GpuBackend, ElementwiseIsBitwiseIdenticalToCpu) {
  if (!gpu_ready()) GTEST_SKIP() << "no GPU backend available";

  const std::vector<Scalar> a = random_data(kBig, 1);
  const std::vector<Scalar> b = random_data(kBig, 2);
  std::vector<Scalar> want(kBig), got(kBig);

  cpu::add(a.data(), b.data(), want.data(), kBig);
  gpu::add(a.data(), b.data(), got.data(), kBig);
  for (size_t i = 0; i < kBig; ++i) EXPECT_TRUE(bitwise_equal(got[i], want[i]));

  cpu::sub(a.data(), b.data(), want.data(), kBig);
  gpu::sub(a.data(), b.data(), got.data(), kBig);
  for (size_t i = 0; i < kBig; ++i) EXPECT_TRUE(bitwise_equal(got[i], want[i]));

  cpu::mul(a.data(), b.data(), want.data(), kBig);
  gpu::mul(a.data(), b.data(), got.data(), kBig);
  for (size_t i = 0; i < kBig; ++i) EXPECT_TRUE(bitwise_equal(got[i], want[i]));
}

TEST(GpuBackend, AccumulatorsMatchCpu) {
  if (!gpu_ready()) GTEST_SKIP() << "no GPU backend available";

  const std::vector<Scalar> x = random_data(kBig, 3);
  const std::vector<Scalar> v = random_data(kBig, 4);
  const std::vector<Scalar> y0 = random_data(kBig, 5);

  std::vector<Scalar> want = y0, got = y0;
  cpu::axpy(-1.0f, x.data(), want.data(), kBig);
  gpu::axpy(-1.0f, x.data(), got.data(), kBig);
  for (size_t i = 0; i < kBig; ++i) EXPECT_TRUE(bitwise_equal(got[i], want[i]));

  want = y0;
  got = y0;
  cpu::mul_add(x.data(), v.data(), want.data(), kBig);
  gpu::mul_add(x.data(), v.data(), got.data(), kBig);
  for (size_t i = 0; i < kBig; ++i) EXPECT_TRUE(bitwise_equal(got[i], want[i]));

  want = y0;
  got = y0;
  cpu::add_bias(0.5f, want.data(), kBig);
  gpu::add_bias(0.5f, got.data(), kBig);
  for (size_t i = 0; i < kBig; ++i) EXPECT_TRUE(bitwise_equal(got[i], want[i]));

  cpu::fill(7.5f, want.data(), kBig);
  gpu::fill(7.5f, got.data(), kBig);
  for (size_t i = 0; i < kBig; ++i) EXPECT_TRUE(bitwise_equal(got[i], want[i]));
}

// The GPU reduces as a tree while the CPU folds sequentially, so the two will
// not agree bitwise on arbitrary data. Integer-valued inputs are exactly
// representable at every partial sum, which makes every summation order give
// the identical answer -- so this can assert exact equality.
TEST(GpuBackend, SumMatchesCpuOnExactlyRepresentableData) {
  if (!gpu_ready()) GTEST_SKIP() << "no GPU backend available";

  std::vector<Scalar> a(kBig);
  for (size_t i = 0; i < kBig; ++i) {
    a[i] = static_cast<Scalar>(i % 17) - 8.0f;  // small integers
  }

  EXPECT_FLOAT_EQ(gpu::sum(a.data(), kBig), cpu::sum(a.data(), kBig));
}

// Guards the "no float atomics in the reduction" rule. Atomics would leave the
// summation order to the scheduler, so identical input would give different
// bits between runs and this test would start flaking.
TEST(GpuBackend, SumIsBitwiseRepeatable) {
  if (!gpu_ready()) GTEST_SKIP() << "no GPU backend available";

  const std::vector<Scalar> a = random_data(kBig, 6);
  const Scalar first = gpu::sum(a.data(), kBig);
  for (int run = 0; run < 5; ++run) {
    EXPECT_TRUE(bitwise_equal(gpu::sum(a.data(), kBig), first))
        << "GPU reduction is not deterministic (run " << run << ")";
  }
}

// Exercises every trans_a/trans_b/beta combination the backward pass uses.
TEST(GpuBackend, MatmulMatchesCpuAcrossAllSpecs) {
  if (!gpu_ready()) GTEST_SKIP() << "no GPU backend available";

  constexpr size_t m = 37, n = 23, k = 19;  // deliberately not tile multiples
  const std::vector<Scalar> a = random_data(m * k, 7);
  const std::vector<Scalar> b = random_data(k * n, 8);
  const std::vector<Scalar> c0 = random_data(m * n, 9);

  for (const bool ta : {false, true}) {
    for (const bool tb : {false, true}) {
      for (const Scalar beta : {0.0f, 1.0f}) {
        const MatMulSpec spec{m, n, k, ta, tb, beta};
        std::vector<Scalar> want = c0, got = c0;
        cpu::matmul(a.data(), b.data(), want.data(), spec);
        gpu::matmul(a.data(), b.data(), got.data(), spec);
        for (size_t i = 0; i < m * n; ++i) {
          expect_close(got[i], want[i]);
        }
      }
    }
  }
}

// The real end-to-end check: build the identical tape program on a CPU tape and
// a GPU tape, and compare both the forward values and every gradient.
TEST(GpuBackend, TapeProgramAgreesAcrossDevices) {
  if (!gpu_ready()) GTEST_SKIP() << "no GPU backend available";

  constexpr size_t m = 24, k = 16, n = 12;
  const std::vector<Scalar> a_val = random_data(m * k, 10);
  const std::vector<Scalar> b_val = random_data(k * n, 11);
  const std::vector<Scalar> c_val = random_data(m * n, 12);

  struct Result {
    std::vector<Scalar> loss, grad_a, grad_b, grad_c;
  };

  auto run = [&](const Device device) {
    Tape tape(TensorClass::TENSOR_TWO, 1.0, device);
    const TensorTwo a(a_val, Shape2{m, k}, true, &tape);
    const TensorTwo b(b_val, Shape2{k, n}, true, &tape);
    const TensorTwo c(c_val, Shape2{m, n}, true, &tape);

    const TensorTwo p = matmul(a, b);   // (m,n)
    const TensorTwo q = p * c;          // elementwise
    const TensorTwo r = q - c;          //
    const TensorTwo loss = mean(r);     // (1,1)
    loss.backward();

    return Result{loss.value(), tape.grad_two[a.node_id],
                  tape.grad_two[b.node_id], tape.grad_two[c.node_id]};
  };

  const Result on_cpu = run(Device::CPU);
  const Result on_gpu = run(Device::GPU);

  ASSERT_EQ(on_gpu.loss.size(), on_cpu.loss.size());
  expect_close(on_gpu.loss[0], on_cpu.loss[0]);

  ASSERT_EQ(on_gpu.grad_a.size(), on_cpu.grad_a.size());
  for (size_t i = 0; i < on_cpu.grad_a.size(); ++i) {
    expect_close(on_gpu.grad_a[i], on_cpu.grad_a[i]);
  }
  ASSERT_EQ(on_gpu.grad_b.size(), on_cpu.grad_b.size());
  for (size_t i = 0; i < on_cpu.grad_b.size(); ++i) {
    expect_close(on_gpu.grad_b[i], on_cpu.grad_b[i]);
  }
  ASSERT_EQ(on_gpu.grad_c.size(), on_cpu.grad_c.size());
  for (size_t i = 0; i < on_cpu.grad_c.size(); ++i) {
    expect_close(on_gpu.grad_c[i], on_cpu.grad_c[i]);
  }
}

// Asking for a GPU on a machine without one must fall back to the CPU, not
// crash -- that is what makes Device::GPU safe to hardcode in a source file.
TEST(GpuBackend, GpuDeviceFallsBackWhenUnavailable) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0, Device::GPU);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4}, Shape2{2, 2}, true, &tape);
  const TensorTwo b(std::vector<Scalar>{5, 6, 7, 8}, Shape2{2, 2}, true, &tape);

  const TensorTwo c = a + b;
  c.backward();

  EXPECT_FLOAT_EQ(c.data[0], 6.0f);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(tape.grad_two[a.node_id][i], 1.0f);
  }
}

}  // namespace autograd_tests
