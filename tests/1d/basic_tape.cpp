//
// Created by Pranav C on 07/06/2026.
//
#include <gtest/gtest.h>

#include "../../include/operations.h"
#include "../../include/tape.h"
#include "../../include/tensor_one.h"

using namespace autograd;

namespace autograd_tests {

TEST(Basic1DTest, OperationWithTape) {
  Tape tape(TensorClass::TENSOR_ONE);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);
  TensorOne b(std::vector<double>{4.0, 5.0, 6.0}, true, &tape);

  TensorOne c = a + b;
  TensorOne d = c * a;
  TensorOne e = d - b;

  EXPECT_EQ(e.data[0], (1.0 + 4.0) * 1.0 - 4.0);
  EXPECT_EQ(e.data[1], (2.0 + 5.0) * 2.0 - 5.0);
  EXPECT_EQ(e.data[2], (3.0 + 6.0) * 3.0 - 6.0);

  // Constants (2) + add (1) + mul (1) + sub (1) = 5 nodes
  EXPECT_EQ(tape.nodes.size(), 5);
}

TEST(Basic1DTest, OperationWithTapeSelf) {
  Tape tape(TensorClass::TENSOR_ONE);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);

  TensorOne b = a + a;
  TensorOne c = b * a;

  EXPECT_EQ(c.data[0], 2.0);
  EXPECT_EQ(c.data[1], 8.0);
  EXPECT_EQ(c.data[2], 18.0);

  EXPECT_EQ(tape.nodes.size(), 3);
}

TEST(Basic1DTest, OperationWithTapeScalarAdd) {
  Tape tape(TensorClass::TENSOR_ONE);

  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, false, &tape);
  TensorOne b = a + a;
  b += 2.0;

  EXPECT_EQ(b.data[0], 4.0);
  EXPECT_EQ(b.data[1], 6.0);
  EXPECT_EQ(b.data[2], 8.0);

  EXPECT_EQ(tape.nodes.size(), 4);
}

TEST(Basic1DTest, OperationWithTapeAdd2) {
  Tape tape(TensorClass::TENSOR_ONE);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, false, &tape);
  TensorOne b = a + a;
  b += a;

  EXPECT_EQ(b.data[0], 3.0);
  EXPECT_EQ(b.data[1], 6.0);
  EXPECT_EQ(b.data[2], 9.0);

  EXPECT_EQ(tape.nodes.size(), 3);
}

TEST(Basic1DTest, BackwardSimpleAdd) {
  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);
  TensorOne b(std::vector<double>{4.0, 5.0, 6.0}, true, &tape);

  TensorOne c = a + b;
  c.backward();

  const std::vector<double> grad_a = tape.grad_one[a.node_id];
  const std::vector<double> grad_b = tape.grad_one[b.node_id];

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(grad_a[i], 1.0);
    EXPECT_DOUBLE_EQ(grad_b[i], 1.0);
  }
}

TEST(Basic1DTest, BackwardSimpleSub) {
  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  TensorOne a(std::vector<double>{5.0, 7.0, 9.0}, true, &tape);
  TensorOne b(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);

  TensorOne c = a - b;
  tape.backward(c.node_id);

  const std::vector<double> grad_a = tape.grad_one[a.node_id];
  const std::vector<double> grad_b = tape.grad_one[b.node_id];

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(grad_a[i], 1.0);
    EXPECT_DOUBLE_EQ(grad_b[i], -1.0);
  }
}

TEST(Basic1DTest, BackwardSimpleMul) {
  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  TensorOne a(std::vector<double>{2.0, 3.0, 4.0}, true, &tape);
  TensorOne b(std::vector<double>{5.0, 6.0, 7.0}, true, &tape);

  TensorOne c = a * b;
  tape.backward(c.node_id);

  const std::vector<double> grad_a = tape.grad_one[a.node_id];  // d(a*b)/da = b
  const std::vector<double> grad_b = tape.grad_one[b.node_id];  // d(a*b)/db = a

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(grad_a[i], b.data[i]);
    EXPECT_DOUBLE_EQ(grad_b[i], a.data[i]);
  }
}

TEST(Basic1DTest, BackwardChainedOps) {
  // e = (a+b)*a - b  =>  de/da = 2a+b, de/db = a-1
  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  TensorOne a(std::vector<double>{1.0, 2.0, 3.0}, true, &tape);
  TensorOne b(std::vector<double>{4.0, 5.0, 6.0}, true, &tape);

  TensorOne c = a + b;
  TensorOne d = c * a;
  TensorOne e = d - b;

  tape.backward(e.node_id);

  const std::vector<double> grad_a = tape.grad_one[a.node_id];
  const std::vector<double> grad_b = tape.grad_one[b.node_id];

  for (size_t i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(grad_a[i], 2.0 * a.data[i] + b.data[i]);
    EXPECT_DOUBLE_EQ(grad_b[i], a.data[i] - 1.0);
  }
}

TEST(Basic1DTest, BackwardSingleElementTensor) {
  Tape tape(TensorClass::TENSOR_ONE, 1.0);
  TensorOne a(std::vector<double>{3.0}, true, &tape);
  TensorOne b(std::vector<double>{5.0}, true, &tape);

  TensorOne c = a * b;
  tape.backward(c.node_id);

  EXPECT_DOUBLE_EQ(tape.grad_one[a.node_id][0], 5.0);
  EXPECT_DOUBLE_EQ(tape.grad_one[b.node_id][0], 3.0);
}
}  // namespace autograd_tests
