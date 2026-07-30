//
// Created by Pranav C on 29/07/2026.
//
// Elementwise multiplication. The matrix product lives in matmul.cpp.
//
// The in-place untaped forms pass their destination as the source too. That is
// documented as safe in backend.h: the GPU backends stage through their own
// device buffers, and the CPU ones touch each index once.

#include "operations.h"

namespace autograd {

TensorTwo operator*(const TensorTwo& a, const TensorTwo& b) {
  check_binary_two(a, b);
  if (a.tape == nullptr) {
    std::vector<Scalar> res(a.size());
    ops::mul(a.device, a.data.data(), b.data.data(), res.data(), a.size());
    return TensorTwo(res, a.shape, false, nullptr, a.device);
  }
  const size_t new_id = a.tape->mul_two(a.node_id, b.node_id);
  return TensorTwo(a.tape, new_id);
}

TensorTwo operator*(const TensorTwo& a, const Scalar k) {
  if (a.tape == nullptr) {
    std::vector<Scalar> res(a.size());
    ops::affine(a.device, k, a.data.data(), 0.0f, res.data(), a.size());
    return TensorTwo(res, a.shape, false, nullptr, a.device);
  }

  const TensorTwo b(a.shape, k, false, a.tape);
  const size_t new_id = a.tape->mul_two(a.node_id, b.node_id);
  return TensorTwo(a.tape, new_id);
}

TensorTwo operator*(const Scalar k, const TensorTwo& a) { return a * k; }

TensorTwo& operator*=(TensorTwo& a, const Scalar k) {
  if (a.tape == nullptr) {
    ops::affine(a.device, k, a.data.data(), 0.0f, a.data.data(), a.size());
    return a;
  }

  std::vector<Scalar> scalar_value(a.size(), k);
  const size_t scalar_id =
      a.tape->add_node_two(std::move(scalar_value), a.shape, false);
  const size_t new_id = a.tape->mul_two(a.node_id, scalar_id);

  a.data = a.tape->values_two[new_id];
  a.shape = a.tape->shapes_two[new_id];
  a.node_id = new_id;
  a.requires_grad = a.tape->required_grads[new_id];
  return a;
}

TensorTwo& operator*=(TensorTwo& a, const TensorTwo& b) {
  check_binary_two(a, b);
  if (a.tape == nullptr) {
    ops::mul(a.device, a.data.data(), b.data.data(), a.data.data(), a.size());
    return a;
  }

  const size_t new_id = a.tape->mul_two(a.node_id, b.node_id);
  a.data = a.tape->values_two[new_id];
  a.shape = a.tape->shapes_two[new_id];
  a.node_id = new_id;
  a.requires_grad = a.tape->required_grads[new_id];
  return a;
}

}  // namespace autograd
