//
// Created by Pranav C on 29/07/2026.
//

#include "operations.h"

namespace autograd {

TensorTwo operator+(const TensorTwo& a, const TensorTwo& b) {
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.shape != b.shape) {
    throw std::invalid_argument("Tensors must have the same shape");
  }
  if (a.tape == nullptr) {
    std::vector<Scalar> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a.data[i] + b.data[i];
    return TensorTwo(res, a.shape, false, nullptr);
  }
  const size_t new_id = a.tape->add_two(a.node_id, b.node_id);
  return TensorTwo(a.tape, new_id);
}

TensorTwo operator+(const TensorTwo& a, const Scalar k) {
  if (a.tape == nullptr) {
    std::vector<Scalar> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a.data[i] + k;
    return TensorTwo(res, a.shape, false, nullptr);
  }

  const TensorTwo b(a.shape, k, false, a.tape);
  const size_t new_id = a.tape->add_two(a.node_id, b.node_id);
  return TensorTwo(a.tape, new_id);
}

TensorTwo operator+(const Scalar k, const TensorTwo& a) { return a + k; }

TensorTwo& operator+=(TensorTwo& a, const Scalar k) {
  if (a.tape == nullptr) {
    for (size_t i = 0; i < a.size(); ++i) a.data[i] += k;
    return a;
  }

  std::vector<Scalar> scalar_value(a.size(), k);
  const size_t scalar_id =
      a.tape->add_node_two(std::move(scalar_value), a.shape, false);
  const size_t new_id = a.tape->add_two(a.node_id, scalar_id);

  a.data = a.tape->values_two[new_id];
  a.shape = a.tape->shapes_two[new_id];
  a.node_id = new_id;
  a.requires_grad = a.tape->required_grads[new_id];
  return a;
}

TensorTwo& operator+=(TensorTwo& a, const TensorTwo& b) {
  // adds b into a
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.shape != b.shape) {
    throw std::invalid_argument("Tensors must have the same shape");
  }
  if (a.tape == nullptr) {
    for (size_t i = 0; i < a.size(); ++i) a.data[i] += b.data[i];
    return a;
  }

  const size_t new_id = a.tape->add_two(a.node_id, b.node_id);
  a.data = a.tape->values_two[new_id];
  a.shape = a.tape->shapes_two[new_id];
  a.node_id = new_id;
  a.requires_grad = a.tape->required_grads[new_id];
  return a;
}

}  // namespace autograd
