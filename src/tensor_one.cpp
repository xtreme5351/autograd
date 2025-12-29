//
// Created by Pranav C on 29/12/2025.
//

#include "tensor_one.h"

namespace autograd {
// TensorBase constructors and method defs

TensorOne::TensorOne(const std::vector<double>& data, const bool requires_grad, Tape* tape)
    : TensorBase(requires_grad, tape), data(data) {}

TensorOne::TensorOne(const int size, const double init_value, const bool requires_grad, Tape *tape)
  : TensorBase(requires_grad, tape) {
  this->data = std::vector<double>(size, init_value);
}

size_t TensorOne::size() const {
  return this->data.size();
}
}
