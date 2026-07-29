//
// Created by Pranav C on 29/07/2026.
//
// Forward-pass and graph-construction tests for the 2D tensor.
//
// All test data is integer-valued and small (magnitudes well under 2^12), so
// every intermediate is exactly representable in float and EXPECT_FLOAT_EQ is
// an assertion about the algorithm rather than about rounding.
//
#include <gtest/gtest.h>

#include "../../include/operations.h"
#include "../../include/tape.h"
#include "../../include/tensor_two.h"

using namespace autograd;

namespace autograd_tests {

TEST(Basic2DTest, ConstructionAndShape) {
  Tape tape(TensorClass::TENSOR_TWO);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4, 5, 6}, Shape2{2, 3}, true,
                    &tape);

  EXPECT_EQ(a.dims().rows, 2);
  EXPECT_EQ(a.dims().cols, 3);
  EXPECT_EQ(a.size(), 6);
  EXPECT_FLOAT_EQ(a.at(0, 2), 3.0f);
  EXPECT_FLOAT_EQ(a.at(1, 0), 4.0f);
  EXPECT_EQ(tape.nodes.size(), 1);
}

TEST(Basic2DTest, FillConstructor) {
  Tape tape(TensorClass::TENSOR_TWO);
  const TensorTwo a(Shape2{3, 2}, 7.0f, false, &tape);

  EXPECT_EQ(a.size(), 6);
  for (const Scalar v : a.value()) EXPECT_FLOAT_EQ(v, 7.0f);
}

TEST(Basic2DTest, ElementwiseChain) {
  Tape tape(TensorClass::TENSOR_TWO);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4}, Shape2{2, 2}, true, &tape);
  const TensorTwo b(std::vector<Scalar>{5, 6, 7, 8}, Shape2{2, 2}, true, &tape);

  const TensorTwo c = a + b;
  const TensorTwo d = c * a;
  const TensorTwo e = d - b;

  for (size_t i = 0; i < 4; ++i) {
    const Scalar av = a.data[i], bv = b.data[i];
    EXPECT_FLOAT_EQ(e.data[i], (av + bv) * av - bv);
  }

  // 2 constants + add + mul + sub = 5 nodes
  EXPECT_EQ(tape.nodes.size(), 5);
}

TEST(Basic2DTest, TapelessEagerOps) {
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4}, Shape2{2, 2});
  const TensorTwo b(std::vector<Scalar>{10, 20, 30, 40}, Shape2{2, 2});

  const TensorTwo c = a + b;
  EXPECT_FLOAT_EQ(c.data[0], 11.0f);
  EXPECT_FLOAT_EQ(c.data[3], 44.0f);
  EXPECT_EQ(c.tape, nullptr);
}

TEST(Basic2DTest, ScalarOverloads) {
  Tape tape(TensorClass::TENSOR_TWO);
  const TensorTwo a(std::vector<Scalar>{1, 2, 3, 4}, Shape2{2, 2}, false,
                    &tape);

  const TensorTwo b = a + 10.0f;
  const TensorTwo c = 10.0f - a;
  const TensorTwo d = a * 3.0f;

  EXPECT_FLOAT_EQ(b.data[0], 11.0f);
  EXPECT_FLOAT_EQ(c.data[0], 9.0f);   // 10 - 1, not 1 - 10
  EXPECT_FLOAT_EQ(c.data[3], 6.0f);   // 10 - 4
  EXPECT_FLOAT_EQ(d.data[3], 12.0f);  // 4 * 3
}

TEST(Basic2DTest, CompoundAssignRebindsNode) {
  Tape tape(TensorClass::TENSOR_TWO);
  TensorTwo a(std::vector<Scalar>{1, 2, 3, 4}, Shape2{2, 2}, false, &tape);
  const size_t original_id = a.node_id;

  a += 5.0f;

  EXPECT_NE(a.node_id, original_id);
  EXPECT_FLOAT_EQ(a.data[0], 6.0f);
  EXPECT_FLOAT_EQ(a.data[3], 9.0f);
  EXPECT_EQ(a.dims().rows, 2);
  // original + scalar const + add = 3 nodes
  EXPECT_EQ(tape.nodes.size(), 3);
}

TEST(Basic2DTest, ShapeMismatchThrows) {
  Tape tape(TensorClass::TENSOR_TWO);
  const TensorTwo a(Shape2{2, 3}, 1.0f, false, &tape);
  const TensorTwo b(Shape2{3, 2}, 1.0f, false, &tape);

  EXPECT_THROW({ [[maybe_unused]] auto c = a + b; }, std::invalid_argument);
}

TEST(Basic2DTest, DifferentTapesThrow) {
  Tape t1(TensorClass::TENSOR_TWO);
  Tape t2(TensorClass::TENSOR_TWO);
  const TensorTwo a(Shape2{2, 2}, 1.0f, false, &t1);
  const TensorTwo b(Shape2{2, 2}, 1.0f, false, &t2);

  EXPECT_THROW({ [[maybe_unused]] auto c = a + b; }, std::invalid_argument);
}

// requires_grad must survive the round trip through the tape. TensorOne gets
// this wrong -- it shadows the base member -- so it is pinned here explicitly.
TEST(Basic2DTest, RequiresGradPropagatesThroughOps) {
  Tape tape(TensorClass::TENSOR_TWO);
  const TensorTwo a(Shape2{2, 2}, 1.0f, true, &tape);
  const TensorTwo b(Shape2{2, 2}, 1.0f, false, &tape);

  const TensorTwo c = a + b;
  EXPECT_TRUE(c.requires_grad);

  const TensorTwo d(Shape2{2, 2}, 1.0f, false, &tape);
  const TensorTwo e = b + d;
  EXPECT_FALSE(e.requires_grad);
}

}  // namespace autograd_tests
