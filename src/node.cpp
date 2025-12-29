//
// Created by Pranav C on 29/12/2025.
//

#include <vector>

#include "tape.h"

namespace autograd {

std::string Node::to_string() const {
  std::string repr = "Node ID: " + std::to_string(id) +
                     ", Type: " + std::to_string(type) + ", Parents: [";
  for (size_t i = 0; i < parents.size(); ++i) {
    repr += std::to_string(parents[i]);
    if (i != parents.size() - 1) repr += ", ";
  }
  repr += "]";
  return repr;
}

}  // namespace autograd