//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_BASE_TENSOR_H
#define AUTOGRAD_BASE_TENSOR_H

#include "tape.h"

namespace autograd {

// Base struct to hold common attributes for all tensor types, again will expand
// later to higher dims
struct TensorBase {
  bool requires_grad;  // does this tensor actually contribute to gradient
  Tape* tape;          // pointer to the tape for comp graph
  int node_id;         // id for comp graph

  explicit TensorBase(bool requires_grad = false, Tape* tape = nullptr,
                      int node_id = -1);

  [[nodiscard]] size_t size() const;  // returns the size of the tensor
};
}  // namespace autograd

#endif  // AUTOGRAD_BASE_TENSOR_H