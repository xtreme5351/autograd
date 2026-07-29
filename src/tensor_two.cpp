//
// Created by Pranav C on 29/07/2026.
//

#include "tensor_two.h"

#include <cassert>

namespace autograd {

TensorTwo::TensorTwo(const std::vector<Scalar>& data, const Shape2 shape,
                     const bool requires_grad, Tape* tape)
    : TensorBase(requires_grad, tape, -1), data(data), shape(shape) {
  assert(data.size() == shape.numel() && "Data size does not match shape");
  if (tape != nullptr) {
    this->node_id = tape->add_node_two(this->data, shape, requires_grad);
  }
}

TensorTwo::TensorTwo(const Shape2 shape, const Scalar init_value,
                     const bool requires_grad, Tape* tape)
    : TensorBase(requires_grad, tape, -1), shape(shape) {
  this->data = std::vector<Scalar>(shape.numel(), init_value);
  if (tape != nullptr) {
    this->node_id = tape->add_node_two(this->data, shape, requires_grad);
  }
}

TensorTwo::TensorTwo(Tape* tape, const size_t node_id)
    : TensorBase(tape->required_grads[node_id], tape, node_id) {
  this->data = tape->values_two[node_id];
  this->shape = tape->shapes_two[node_id];
}

size_t TensorTwo::size() const { return this->data.size(); }

Shape2 TensorTwo::dims() const { return this->shape; }

std::vector<Scalar> TensorTwo::value() const { return this->data; }

std::vector<Scalar> TensorTwo::grad() const {
  return tape->grad_two[this->node_id];
}

Scalar TensorTwo::at(const size_t row, const size_t col) const {
  assert(row < shape.rows && col < shape.cols && "Index out of range");
  return data[row * shape.cols + col];
}

std::string TensorTwo::to_string() const {
  std::string repr = "TensorTwo(data=[";
  for (size_t r = 0; r < shape.rows; ++r) {
    repr += "[";
    for (size_t c = 0; c < shape.cols; ++c) {
      repr += std::to_string(data[r * shape.cols + c]);
      if (c != shape.cols - 1) {
        repr += ", ";
      }
    }
    repr += "]";
    if (r != shape.rows - 1) {
      repr += ", ";
    }
  }
  repr += "], shape=(" + std::to_string(shape.rows) + ", " +
          std::to_string(shape.cols) +
          "), requires_grad=" + std::string(requires_grad ? "true" : "false") +
          ")";
  return repr;
}

void TensorTwo::backward() const { this->tape->backward_two(this->node_id); }

}  // namespace autograd
