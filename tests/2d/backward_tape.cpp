//
// Created by Pranav C on 29/07/2026.
//
// Reverse-mode tests for the 2D tape: elementwise rules, multi-path
// accumulation, and the reduction/matmul shape changes.
//
#include <gtest/gtest.h>

#include "../../include/operations.h"
#include "../../include/tape.h"
#include "../../include/tensor_two.h"

using namespace autograd;

namespace autograd_tests {

TEST(Backward2DTest, AddSubGradients) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4}, Shape2{2, 2}, true, &tape);
  const TensorTwo b(std::vector<Scalar>{5, 6, 7, 8}, Shape2{2, 2}, true, &tape);

  const TensorTwo c = a - b;
  tape.backward_two(c.node_id);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(tape.grad_two[a.node_id][i], 1.0f);
    EXPECT_FLOAT_EQ(tape.grad_two[b.node_id][i], -1.0f);
  }
}

TEST(Backward2DTest, MulGradientIsTheOtherOperand) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4}, Shape2{2, 2}, true, &tape);
  const TensorTwo b(std::vector<Scalar>{5, 6, 7, 8}, Shape2{2, 2}, true, &tape);

  const TensorTwo c = a * b;
  tape.backward_two(c.node_id);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(tape.grad_two[a.node_id][i], b.data[i]);
    EXPECT_FLOAT_EQ(tape.grad_two[b.node_id][i], a.data[i]);
  }
}

// b = a+a, c = a*a, d = b+c: a reaches the output through two branches that
// recombine, so dd/da = 2 + 2a by the sum-over-paths rule.
TEST(Backward2DTest, DiamondMultiPathAccumulation) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4}, Shape2{2, 2}, true, &tape);

  const TensorTwo b = a + a;
  const TensorTwo c = a * a;
  const TensorTwo d = b + c;

  tape.backward_two(d.node_id);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(tape.grad_two[a.node_id][i], 2.0f + 2.0f * a.data[i]);
  }
}

TEST(Backward2DTest, SumGradientIsOnesBroadcast) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4, 5, 6}, Shape2{2, 3}, true,
                    &tape);

  const TensorTwo s = sum(a);
  EXPECT_EQ(s.dims().rows, 1);
  EXPECT_EQ(s.dims().cols, 1);
  EXPECT_FLOAT_EQ(s.data[0], 21.0f);

  s.backward();
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(tape.grad_two[a.node_id][i], 1.0f);
  }
}

TEST(Backward2DTest, MeanGradientIsOneOverN) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  const TensorTwo a(std::vector<Scalar>{2, 4, 6, 8}, Shape2{2, 2}, true, &tape);

  const TensorTwo m = mean(a);
  EXPECT_FLOAT_EQ(m.data[0], 5.0f);

  m.backward();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(tape.grad_two[a.node_id][i], 0.25f);
  }
}

TEST(Backward2DTest, MatmulForwardShapeAndValues) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  // A is (2,3), B is (3,2) -> C is (2,2)
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4, 5, 6}, Shape2{2, 3}, true,
                    &tape);
  const TensorTwo b(std::vector<Scalar>{7, 8, 9, 10, 11, 12}, Shape2{3, 2},
                    true, &tape);

  const TensorTwo c = matmul(a, b);

  EXPECT_EQ(c.dims().rows, 2);
  EXPECT_EQ(c.dims().cols, 2);
  // [1 2 3] . [7 9 11]^T = 7 + 18 + 33 = 58
  EXPECT_FLOAT_EQ(c.at(0, 0), 58.0f);
  EXPECT_FLOAT_EQ(c.at(0, 1), 64.0f);   // 8 + 20 + 36
  EXPECT_FLOAT_EQ(c.at(1, 0), 139.0f);  // 28 + 45 + 66
  EXPECT_FLOAT_EQ(c.at(1, 1), 154.0f);  // 32 + 50 + 72
}

// With a seed of all ones, dA = ones(M,N) * B^T, i.e. every entry of row i of
// dA is the row sum of B; dB entries are the column sums of A.
TEST(Backward2DTest, MatmulGradients) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4, 5, 6}, Shape2{2, 3}, true,
                    &tape);
  const TensorTwo b(std::vector<Scalar>{7, 8, 9, 10, 11, 12}, Shape2{3, 2},
                    true, &tape);

  const TensorTwo c = matmul(a, b);
  tape.backward_two(c.node_id);

  const std::vector<Scalar>& ga = tape.grad_two[a.node_id];
  ASSERT_EQ(ga.size(), 6);
  // Row sums of B: [7+8, 9+10, 11+12] = [15, 19, 23], same for both rows of A.
  const Scalar expect_a[6] = {15, 19, 23, 15, 19, 23};
  for (size_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(ga[i], expect_a[i]);

  const std::vector<Scalar>& gb = tape.grad_two[b.node_id];
  ASSERT_EQ(gb.size(), 6);
  // Column sums of A: [1+4, 2+5, 3+6] = [5, 7, 9], repeated across B's cols.
  const Scalar expect_b[6] = {5, 5, 7, 7, 9, 9};
  for (size_t i = 0; i < 6; ++i) EXPECT_FLOAT_EQ(gb[i], expect_b[i]);
}

// A non-square matmul feeding a scalar loss -- the shape bookkeeping has to
// hold across (2,3)x(3,4) -> (2,4) -> (1,1) and all the way back.
TEST(Backward2DTest, MatmulThenSumEndToEnd) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4, 5, 6}, Shape2{2, 3}, true,
                    &tape);
  const TensorTwo b(std::vector<Scalar>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                    Shape2{3, 4}, true, &tape);

  const TensorTwo c = matmul(a, b);
  ASSERT_EQ(c.dims().rows, 2);
  ASSERT_EQ(c.dims().cols, 4);

  const TensorTwo loss = sum(c);
  loss.backward();

  // dL/dA[i][k] = sum over j of B[k][j] -- the row sums of B.
  const Scalar row_sums_b[3] = {1 + 2 + 3 + 4, 5 + 6 + 7 + 8, 9 + 10 + 11 + 12};
  const std::vector<Scalar>& ga = tape.grad_two[a.node_id];
  for (size_t i = 0; i < 2; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      EXPECT_FLOAT_EQ(ga[i * 3 + k], row_sums_b[k]);
    }
  }

  // dL/dB[k][j] = sum over i of A[i][k] -- the column sums of A.
  const Scalar col_sums_a[3] = {1 + 4, 2 + 5, 3 + 6};
  const std::vector<Scalar>& gb = tape.grad_two[b.node_id];
  for (size_t k = 0; k < 3; ++k) {
    for (size_t j = 0; j < 4; ++j) {
      EXPECT_FLOAT_EQ(gb[k * 4 + j], col_sums_a[k]);
    }
  }
}

// Central finite differences against a double-precision reference forward pass.
// eps must be far larger than the 1e-6 the 1D suite uses: float's machine
// epsilon puts sqrt(eps_m) near 3.4e-4, so at 1e-6 the quotient is pure noise.
TEST(Backward2DTest, NumericalGradCheck) {
  const std::vector<Scalar> a_val = {1.5f, -2.0f, 0.5f, 3.0f};
  const std::vector<Scalar> b_val = {2.0f, 3.0f, -1.0f, 0.25f};

  // x1 = a + b; x2 = x1 * a; x3 = x2 - b; loss = sum(x3)
  auto forward_double = [&](const std::vector<Scalar>& av,
                            const std::vector<Scalar>& bv) {
    double acc = 0.0;
    for (size_t i = 0; i < av.size(); ++i) {
      const double a = av[i], b = bv[i];
      acc += (a + b) * a - b;
    }
    return acc;
  };

  Tape tape(TensorClass::TENSOR_TWO, 1.0);
  const TensorTwo a(a_val, Shape2{2, 2}, true, &tape);
  const TensorTwo b(b_val, Shape2{2, 2}, true, &tape);
  const TensorTwo x1 = a + b;
  const TensorTwo x2 = x1 * a;
  const TensorTwo x3 = x2 - b;
  const TensorTwo loss = sum(x3);
  loss.backward();

  const std::vector<Scalar> analytic = tape.grad_two[a.node_id];
  constexpr double eps = 1e-3;
  for (size_t k = 0; k < a_val.size(); ++k) {
    std::vector<Scalar> plus = a_val, minus = a_val;
    plus[k] += static_cast<Scalar>(eps);
    minus[k] -= static_cast<Scalar>(eps);
    const double numeric =
        (forward_double(plus, b_val) - forward_double(minus, b_val)) /
        (2 * eps);
    EXPECT_NEAR(analytic[k], numeric, 1e-2);
  }
}

// backward_two zeroes grad_two[0..node_id] on every call, same as the 1D path,
// so reusing one tape for two computations clobbers the earlier result.
// Documenting the behaviour rather than assuming PyTorch semantics.
TEST(Backward2DTest, SecondBackwardCallClobbersEarlierResults) {
  Tape tape(TensorClass::TENSOR_TWO, 1.0);

  const TensorTwo p1(std::vector<Scalar>{2, 3}, Shape2{1, 2}, true, &tape);
  const TensorTwo p2(std::vector<Scalar>{4, 5}, Shape2{1, 2}, true, &tape);
  const TensorTwo out1 = p1 * p2;
  tape.backward_two(out1.node_id);
  EXPECT_FLOAT_EQ(tape.grad_two[p1.node_id][0], 4.0f);

  const TensorTwo q1(std::vector<Scalar>{10, 10}, Shape2{1, 2}, true, &tape);
  const TensorTwo q2(std::vector<Scalar>{1, 1}, Shape2{1, 2}, true, &tape);
  const TensorTwo out2 = q1 + q2;
  tape.backward_two(out2.node_id);
  EXPECT_FLOAT_EQ(tape.grad_two[q1.node_id][0], 1.0f);

  EXPECT_FLOAT_EQ(tape.grad_two[p1.node_id][0], 0.0f);
}

}  // namespace autograd_tests
