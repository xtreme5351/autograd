//
// Created by Pranav C on 30/07/2026.
//
// Untaped ("eager") 2D operator coverage.
//
// Before the device was added to TensorTwo, every tape == nullptr branch ran a
// hand-written scalar loop and could not reach a backend at all. These tests
// pin down the two properties that replacement has to preserve:
//
//   1. the arithmetic is unchanged, and
//   2. a Device::GPU tensor produces bit-identical results to a Device::CPU one
//      (on a machine with no GPU this still runs, because ops:: falls back --
//      it just compares the CPU path against itself, which is why the numeric
//      assertions below are written out explicitly rather than only as a
//      CPU-vs-GPU diff).

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "../../include/backend.h"
#include "../../include/operations.h"
#include "../../include/tensor_two.h"
#include "../support/device_guard.h"

using namespace autograd;

namespace autograd_tests {
namespace {

TensorTwo make(const std::vector<Scalar>& v, const Device d) {
  return TensorTwo(v, Shape2{2, 2}, false, nullptr, d);
}

const std::vector<Scalar> kA{1.0f, 2.0f, 3.0f, 4.0f};
const std::vector<Scalar> kB{0.5f, -1.0f, 2.0f, 0.25f};

// Runs `op` on a CPU tensor and a GPU tensor and requires the two to agree
// bitwise, then hands back the CPU result for a value assertion.
//
// The ForceGpu guard matters: these tensors are tiny so their expected values
// can be written out by hand, which puts them far below the dispatch
// thresholds. Without it the "GPU" run would fall back to the CPU even on a
// machine with a working backend, and this would compare the CPU against
// itself.
template <typename Op>
std::vector<Scalar> both_devices(Op op) {
  const std::vector<Scalar> on_cpu = op(Device::CPU);
  const ForceGpu force;
  const std::vector<Scalar> on_gpu = op(Device::GPU);
  EXPECT_EQ(on_gpu.size(), on_cpu.size());
  for (size_t i = 0; i < on_cpu.size(); ++i) {
    EXPECT_EQ(std::memcmp(&on_gpu[i], &on_cpu[i], sizeof(Scalar)), 0)
        << "CPU/GPU eager mismatch at " << i;
  }
  return on_cpu;
}

}  // namespace

TEST(Eager2DTest, DefaultsToCpuDevice) {
  const TensorTwo a(kA, Shape2{2, 2});
  EXPECT_EQ(a.device, Device::CPU);
}

TEST(Eager2DTest, TapedTensorTakesDeviceFromTape) {
  // The tape is the single source of truth: the constructor argument is ignored
  // when a tape is present, so a tensor can never disagree with its own node.
  Tape tape(TensorClass::TENSOR_TWO, 1.0, Device::GPU);
  const TensorTwo a(kA, Shape2{2, 2}, true, &tape, Device::CPU);
  EXPECT_EQ(a.device, Device::GPU);

  const TensorTwo b(kB, Shape2{2, 2}, true, &tape);
  EXPECT_EQ((a + b).device, Device::GPU);  // results too
}

TEST(Eager2DTest, ResultCarriesOperandDevice) {
  const TensorTwo a = make(kA, Device::GPU);
  const TensorTwo b = make(kB, Device::GPU);
  EXPECT_EQ((a + b).device, Device::GPU);
  EXPECT_EQ((a * 2.0f).device, Device::GPU);
  EXPECT_EQ(matmul(a, b).device, Device::GPU);
  EXPECT_EQ(sum(a).device, Device::GPU);
}

TEST(Eager2DTest, MixedDevicesThrow) {
  const TensorTwo cpu_t = make(kA, Device::CPU);
  const TensorTwo gpu_t = make(kB, Device::GPU);
  EXPECT_THROW(cpu_t + gpu_t, std::invalid_argument);
  EXPECT_THROW(cpu_t - gpu_t, std::invalid_argument);
  EXPECT_THROW(cpu_t * gpu_t, std::invalid_argument);
  EXPECT_THROW(matmul(cpu_t, gpu_t), std::invalid_argument);
}

TEST(Eager2DTest, TensorTensorOps) {
  const auto add = both_devices([](const Device d) {
    return (make(kA, d) + make(kB, d)).value();
  });
  EXPECT_FLOAT_EQ(add[0], 1.5f);
  EXPECT_FLOAT_EQ(add[1], 1.0f);
  EXPECT_FLOAT_EQ(add[2], 5.0f);
  EXPECT_FLOAT_EQ(add[3], 4.25f);

  const auto sub = both_devices([](const Device d) {
    return (make(kA, d) - make(kB, d)).value();
  });
  EXPECT_FLOAT_EQ(sub[0], 0.5f);
  EXPECT_FLOAT_EQ(sub[3], 3.75f);

  const auto mul = both_devices([](const Device d) {
    return (make(kA, d) * make(kB, d)).value();
  });
  EXPECT_FLOAT_EQ(mul[0], 0.5f);
  EXPECT_FLOAT_EQ(mul[1], -2.0f);
  EXPECT_FLOAT_EQ(mul[3], 1.0f);
}

// Every tensor-scalar form goes through ops::affine, so these pin down the
// (alpha, beta) mapping -- in particular that `k - a` negates rather than
// subtracting the wrong way round.
TEST(Eager2DTest, TensorScalarOps) {
  const auto plus = both_devices(
      [](const Device d) { return (make(kA, d) + 10.0f).value(); });
  EXPECT_FLOAT_EQ(plus[0], 11.0f);
  EXPECT_FLOAT_EQ(plus[3], 14.0f);

  const auto rplus = both_devices(
      [](const Device d) { return (10.0f + make(kA, d)).value(); });
  EXPECT_FLOAT_EQ(rplus[0], 11.0f);

  const auto minus = both_devices(
      [](const Device d) { return (make(kA, d) - 1.0f).value(); });
  EXPECT_FLOAT_EQ(minus[0], 0.0f);
  EXPECT_FLOAT_EQ(minus[3], 3.0f);

  const auto rminus = both_devices(
      [](const Device d) { return (10.0f - make(kA, d)).value(); });
  EXPECT_FLOAT_EQ(rminus[0], 9.0f);
  EXPECT_FLOAT_EQ(rminus[3], 6.0f);

  const auto times = both_devices(
      [](const Device d) { return (make(kA, d) * 3.0f).value(); });
  EXPECT_FLOAT_EQ(times[0], 3.0f);
  EXPECT_FLOAT_EQ(times[3], 12.0f);

  const auto rtimes = both_devices(
      [](const Device d) { return (3.0f * make(kA, d)).value(); });
  EXPECT_FLOAT_EQ(rtimes[3], 12.0f);
}

// The in-place forms alias source and destination through the backend.
TEST(Eager2DTest, CompoundAssignment) {
  const auto pe = both_devices([](const Device d) {
    TensorTwo a = make(kA, d);
    a += make(kB, d);
    return a.value();
  });
  EXPECT_FLOAT_EQ(pe[0], 1.5f);
  EXPECT_FLOAT_EQ(pe[3], 4.25f);

  const auto me = both_devices([](const Device d) {
    TensorTwo a = make(kA, d);
    a -= make(kB, d);
    return a.value();
  });
  EXPECT_FLOAT_EQ(me[0], 0.5f);

  const auto te = both_devices([](const Device d) {
    TensorTwo a = make(kA, d);
    a *= make(kB, d);
    return a.value();
  });
  EXPECT_FLOAT_EQ(te[1], -2.0f);

  const auto pek = both_devices([](const Device d) {
    TensorTwo a = make(kA, d);
    a += 5.0f;
    return a.value();
  });
  EXPECT_FLOAT_EQ(pek[0], 6.0f);

  const auto mek = both_devices([](const Device d) {
    TensorTwo a = make(kA, d);
    a -= 5.0f;
    return a.value();
  });
  EXPECT_FLOAT_EQ(mek[0], -4.0f);

  const auto tek = both_devices([](const Device d) {
    TensorTwo a = make(kA, d);
    a *= 2.0f;
    return a.value();
  });
  EXPECT_FLOAT_EQ(tek[0], 2.0f);
  EXPECT_FLOAT_EQ(tek[3], 8.0f);
}

TEST(Eager2DTest, MatmulAndReductions) {
  // Non-square, so a row/column mix-up in the untaped path cannot hide.
  const auto mm = both_devices([](const Device d) {
    const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4, 5, 6}, Shape2{2, 3}, false,
                      nullptr, d);
    const TensorTwo b(std::vector<Scalar>{7, 8, 9, 10, 11, 12}, Shape2{3, 2},
                      false, nullptr, d);
    return matmul(a, b).value();
  });
  ASSERT_EQ(mm.size(), 4u);
  EXPECT_FLOAT_EQ(mm[0], 58.0f);   // 1*7 + 2*9 + 3*11
  EXPECT_FLOAT_EQ(mm[1], 64.0f);   // 1*8 + 2*10 + 3*12
  EXPECT_FLOAT_EQ(mm[2], 139.0f);  // 4*7 + 5*9 + 6*11
  EXPECT_FLOAT_EQ(mm[3], 154.0f);  // 4*8 + 5*10 + 6*12

  const auto s = both_devices(
      [](const Device d) { return sum(make(kA, d)).value(); });
  EXPECT_FLOAT_EQ(s[0], 10.0f);

  const auto mn = both_devices(
      [](const Device d) { return mean(make(kA, d)).value(); });
  EXPECT_FLOAT_EQ(mn[0], 2.5f);
}

}  // namespace autograd_tests
