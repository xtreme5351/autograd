//
// Created by Pranav C on 18/07/2026.
//

#include "operations.h"

namespace autograd {

TensorOne operator-(const TensorOne& a, const TensorOne& b) {
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.tape == nullptr) {
    std::vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a.data[i] - b.data[i];
    return TensorOne(res, false, nullptr);
  }
  const size_t new_id = a.tape->sub_one(a.node_id, b.node_id);
  return TensorOne(a.tape, new_id);
}

TensorOne operator-(const TensorOne& a, const double k) {
  const size_t n = a.size();

  if (a.tape == nullptr) {
    std::vector<double> res(n);
    for (size_t i = 0; i < n; ++i) res[i] = a.data[i] - k;
    return TensorOne(res, false, nullptr);
  }
  const TensorOne b(n, k, false, a.tape);
  const size_t new_id = a.tape->sub_one(a.node_id, b.node_id);
  return TensorOne(a.tape, new_id);
}

TensorOne operator-(const double k, const TensorOne& a) {
  const size_t n = a.size();

  if (a.tape == nullptr) {
    std::vector<double> res(n);
    for (size_t i = 0; i < n; ++i) res[i] = k - a.data[i];
    return TensorOne(res, false, nullptr);
  }

  const TensorOne b(n, k, false, a.tape);
  const size_t new_id = a.tape->sub_one(b.node_id, a.node_id);
  return TensorOne(a.tape, new_id);
}

TensorOne& operator-=(TensorOne& a, const double k) {
  const size_t n = a.size();

  if (a.tape == nullptr) {
    for (size_t i = 0; i < n; ++i) a.data[i] -= k;
    return a;
  }

  std::vector<double> scalar_value(n, k);
  const size_t scalar_id = a.tape->add_node_one(std::move(scalar_value), false);
  const size_t new_id = a.tape->sub_one(a.node_id, scalar_id);

  a.data = a.tape->values_one[new_id];
  a.node_id = new_id;
  a.requires_grad = a.tape->required_grads[new_id];
  return a;
}

TensorOne& operator-=(TensorOne& a, const TensorOne& b) {
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.tape == nullptr) {
    for (size_t i = 0; i < a.size(); ++i) a.data[i] -= b.data[i];
    return a;
  }

  const size_t new_id = a.tape->sub_one(a.node_id, b.node_id);
  a.data = a.tape->values_one[new_id];
  a.node_id = new_id;
  a.requires_grad = a.tape->required_grads[new_id];
  return a;
}

TensorOne& operator-=(const double k, TensorOne& a) {
  const size_t n = a.size();

  if (a.tape == nullptr) {
    for (size_t i = 0; i < n; ++i) a.data[i] = k - a.data[i];
    return a;
  }

  std::vector<double> scalar_value(n, k);
  const size_t scalar_id = a.tape->add_node_one(std::move(scalar_value), false);
  const size_t new_id =
      a.tape->sub_one(scalar_id, a.node_id);  // k - a, not a - k

  a.data = a.tape->values_one[new_id];
  a.node_id = new_id;
  a.requires_grad = a.tape->required_grads[new_id];
  return a;
}

}  // namespace autograd