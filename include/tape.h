//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_TAPE_H
#define AUTOGRAD_TAPE_H
#include <string>
#include <vector>

#include "../src/consts.h"

namespace autograd {

struct Node {
  NodeType type;
  std::vector<int> parents;  // ids of parent nodes
  [[nodiscard]] std::string to_string() const;
};

struct Tape {
  std::vector<Node> nodes;
  std::vector<double> gradients;  // node->gradient mapping by index, faster
                                  // than storing in Node struct

  size_t add_node(NodeType type, const std::vector<int>& parents);
  void backward(int node_id, double grad);
  void to_string();
};
}  // namespace autograd

#endif  // AUTOGRAD_TAPE_H