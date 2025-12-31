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
  TensorClass node_class;
  std::tuple<int, int> dims;  // m x n
  std::vector<int> parents;   // ids of parent nodes
  [[nodiscard]] std::string to_string() const;
};

struct Tape {
  double seed_grad;
  TensorClass tape_class;
  std::vector<Node> nodes;

  /* node->gradient mapping by index, faster than storing in Node struct */
  std::vector<double> grad_zero;
  std::vector<std::vector<double>> grad_one;

  explicit Tape(const TensorClass tape_class, const double seed_grad = 1.0)
      : seed_grad(seed_grad), tape_class(tape_class) {}

  size_t add_node(NodeType type, TensorClass node_class,
                  const std::tuple<int, int> &dims,
                  const std::vector<int> &parents);
  void backward(int node_id);
  void to_string();
};
}  // namespace autograd

#endif  // AUTOGRAD_TAPE_H