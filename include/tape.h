//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_TAPE_H
#define AUTOGRAD_TAPE_H
#include <string>
#include <tuple>
#include <vector>

#include "consts.h"

namespace autograd {

struct Node {
  NodeType type;
  std::vector<size_t> parents;  // ids of parent nodes
  [[nodiscard]] std::string to_string() const;
};

// Row-major (rows, cols) extent of a 2D tensor node.
struct Shape2 {
  size_t rows{}, cols{};
  [[nodiscard]] size_t numel() const { return rows * cols; }
  friend bool operator==(const Shape2& a, const Shape2& b) {
    return a.rows == b.rows && a.cols == b.cols;
  }
};

struct Tape {
  double seed_grad;
  TensorClass tape_class;
  Device device;  // which backend 2D ops on this tape run on
  std::vector<Node> nodes;

  std::vector<bool> required_grads;  // does a node require grad

  /* node->gradient mapping by index, faster than storing in Node struct */
  std::vector<double> grad_zero;
  std::vector<std::vector<double>> grad_one;

  std::vector<std::vector<Scalar>> grad_two;

  /* node->value mapping by index, faster than storing in Node struct  */
  std::vector<double> values_zero;
  std::vector<std::vector<double>> values_one;
  std::vector<std::vector<Scalar>> values_two;  // flat row-major

  /* node->shape mapping by index, 2D only */
  std::vector<Shape2> shapes_two;

  // device is defaulted, so every existing 1D call site keeps compiling.
  explicit Tape(const TensorClass tape_class, const double seed_grad = 0.0,
                const Device device = default_device)
      : seed_grad(seed_grad), tape_class(tape_class), device(device) {}

  /* get tensor values method */
  [[nodiscard]] std::tuple<int, std::vector<double>,
                           std::vector<std::vector<double>>>
  get_values_one(size_t a, size_t b) const;

  // All addition node functions return the node id (index in nodes vector)
  size_t add_node_zero(double value, bool requires_grad);  // add 0d tensor
  size_t add_node_one(std::vector<double> value,
                      bool requires_grad);  // add 1d tensor node

  /* 1d tensor operations */
  size_t add_one(size_t a, size_t b);
  size_t sub_one(size_t a, size_t b);
  size_t mul_one(size_t a, size_t b);
  // size_t div_one(size_t a, size_t b); investigate tensor division

  size_t add_node_two(std::vector<Scalar> value, Shape2 shape,
                      bool requires_grad);  // add 2d tensor node

  /* 2d tensor operations */
  size_t add_two(size_t a, size_t b);
  size_t sub_two(size_t a, size_t b);
  size_t mul_two(size_t a, size_t b);
  size_t matmul_two(size_t a, size_t b);  // (M,K) x (K,N) -> (M,N)
  size_t sum_two(size_t a);               // full reduction -> (1,1)
  size_t mean_two(size_t a);              // full reduction -> (1,1)

  void backward(size_t node_id);
  void backward_one(size_t node_id);
  void backward_two(size_t node_id);
  void to_string();
};
}  // namespace autograd

#endif  // AUTOGRAD_TAPE_H