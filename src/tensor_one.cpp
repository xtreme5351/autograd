//
// Created by Pranav C on 29/12/2025.
//

#include "tensor_one.h"

namespace autograd {
// TensorBase constructors and method defs

TensorOne::TensorOne(const std::vector<double>& data, const bool requires_grad,
                     Tape* tape)
    : TensorBase(requires_grad, tape, -1), data(data) {
  if (tape != nullptr) {
    this->node_id = tape->add_node_one(this->data, requires_grad);
  }
}

TensorOne::TensorOne(const size_t size, const double init_value,
                     const bool requires_grad, Tape* tape)
    : TensorBase(requires_grad, tape, -1) {
  this->data = std::vector(size, init_value);
  if (tape != nullptr) {
    this->node_id = tape->add_node_one(this->data, requires_grad);
  }
}

TensorOne::TensorOne(Tape* tape, const size_t node_id)
    : TensorBase(tape->required_grads[node_id], tape, node_id) {
  this->data = tape->values_one[node_id];
}

size_t TensorOne::size() const { return this->data.size(); }

std::vector<double> TensorOne::value() const { return this->data; }

std::vector<double> TensorOne::grad() const {
  return tape->grad_one[this->node_id];
}

std::string TensorOne::to_string() const {
  std::string repr = "TensorOne(data=[";
  for (size_t i = 0; i < data.size(); ++i) {
    repr += std::to_string(data[i]);
    if (i != data.size() - 1) {
      repr += ", ";
    }
  }
  repr +=
      "], requires_grad=" + std::string(requires_grad ? "true" : "false") + ")";
  return repr;
}

void TensorOne::backward() { this->tape->backward_one(this->node_id); }

}  // namespace autograd
