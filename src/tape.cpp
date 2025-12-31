//
// Created by Pranav C on 29/12/2025.
//

#include "tape.h"

#include <cassert>
#include <tuple>
#include <vector>

namespace autograd {

size_t Tape::add_node(const NodeType type, const TensorClass node_class,
                      const std::tuple<int, int>& dims,
                      const std::vector<int>& parents) {
  assert(node_class == tape_class);  // invariant, all tape nodes are same class

  nodes.push_back(Node{type, node_class, dims, parents});
  const size_t idx = nodes.size() - 1;

  switch (node_class) {
    case TENSOR_ONE:
      assert(std::get<1>(dims) == 1);  // ensure 1d tensor
      grad_one.emplace_back(std::get<0>(dims),
                            seed_grad);  // initialize to 1d seed
      assert(idx == grad_one.size() - 1);
      break;

    case TENSOR_ZERO:
      assert(std::get<1>(dims) == 0 &&
             std::get<0>(dims) == 0);     // ensure scalar
      grad_zero.emplace_back(seed_grad);  // initialize to seed
      assert(idx == grad_zero.size() - 1);
      break;

    default:
      throw std::runtime_error("Unknown node class");
  }

  return idx;
}

void Tape::backward(const int node_id) {}

void Tape::to_string() {}

}  // namespace autograd