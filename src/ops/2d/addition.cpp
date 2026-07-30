//
// Created by Pranav C on 29/07/2026.
//
// The untaped branches dispatch through ops:: on the tensor's own device, the
// same way the taped branches dispatch on the tape's. Neither writes its own
// loop -- the arithmetic lives in the backend primitives so there is exactly
// one implementation of each operation to keep correct.

#include "operations.h"

namespace autograd {

TensorTwo operator+(const TensorTwo& a, const TensorTwo& b) {
  check_binary_two(a, b);
  if (a.tape == nullptr) {
    std::vector<Scalar> res(a.size());
    ops::add(a.device, a.data.data(), b.data.data(), res.data(), a.size());
    return TensorTwo(res, a.shape, false, nullptr, a.device);
  }
  const size_t new_id = a.tape->add_two(a.node_id, b.node_id);
  return TensorTwo(a.tape, new_id);
}

TensorTwo operator+(const TensorTwo& a, const Scalar k) {
  if (a.tape == nullptr) {
    std::vector<Scalar> res(a.size());
    ops::affine(a.device, 1.0f, a.data.data(), k, res.data(), a.size());
    return TensorTwo(res, a.shape, false, nullptr, a.device);
  }

  const TensorTwo b(a.shape, k, false, a.tape);
  const size_t new_id = a.tape->add_two(a.node_id, b.node_id);
  return TensorTwo(a.tape, new_id);
}

TensorTwo operator+(const Scalar k, const TensorTwo& a) { return a + k; }

TensorTwo& operator+=(TensorTwo& a, const Scalar k) {
  if (a.tape == nullptr) {
    ops::add_bias(a.device, k, a.data.data(), a.size());
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
  check_binary_two(a, b);
  if (a.tape == nullptr) {
    ops::axpy(a.device, 1.0f, b.data.data(), a.data.data(), a.size());
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
