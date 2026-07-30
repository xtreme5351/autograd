//
// Created by Pranav C on 29/07/2026.
//
// Full-tensor reductions. Both collapse to a (1,1) tensor, which is what gives
// backward() a scalar root to seed from.

#include "operations.h"

namespace autograd {

TensorTwo sum(const TensorTwo& a) {
  if (a.size() == 0) {
    throw std::invalid_argument("Cannot reduce an empty tensor");
  }
  if (a.tape == nullptr) {
    const Scalar total = ops::sum(a.device, a.data.data(), a.size());
    return TensorTwo(std::vector<Scalar>{total}, Shape2{1, 1}, false, nullptr,
                     a.device);
  }
  const size_t new_id = a.tape->sum_two(a.node_id);
  return TensorTwo(a.tape, new_id);
}

TensorTwo mean(const TensorTwo& a) {
  if (a.size() == 0) {
    throw std::invalid_argument("Cannot reduce an empty tensor");
  }
  if (a.tape == nullptr) {
    const Scalar total = ops::sum(a.device, a.data.data(), a.size()) /
                         static_cast<Scalar>(a.size());
    return TensorTwo(std::vector<Scalar>{total}, Shape2{1, 1}, false, nullptr,
                     a.device);
  }
  const size_t new_id = a.tape->mean_two(a.node_id);
  return TensorTwo(a.tape, new_id);
}

}  // namespace autograd
