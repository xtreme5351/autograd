//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_TAPE_H
#define AUTOGRAD_TAPE_H
#include <string>
#include <tuple>
#include <vector>

#include "../src/consts.h"

namespace autograd {

struct Node {
  NodeType type;
  std::vector<size_t> parents;  // ids of parent nodes
  [[nodiscard]] std::string to_string() const;
};

struct Tape {
  double seed_grad;
  TensorClass tape_class;
  std::vector<Node> nodes;

  std::vector<bool> required_grads;  // does a node require grad

  /* node->gradient mapping by index, faster than storing in Node struct */
  std::vector<double> grad_zero;
  std::vector<std::vector<double>> grad_one;

  /* node->value mapping by index, faster than storing in Node struct  */
  std::vector<double> values_zero;
  std::vector<std::vector<double>> values_one;

  explicit Tape(const TensorClass tape_class, const double seed_grad = 0.0)
      : seed_grad(seed_grad), tape_class(tape_class) {}

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
  void backward(size_t node_id);
  void backward_one(size_t node_id);
  void to_string();
};
}  // namespace autograd

#endif  // AUTOGRAD_TAPE_H