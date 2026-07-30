//
// Created by Pranav C on 29/07/2026.
//

#include "operations.h"

namespace autograd {

TensorTwo matmul(const TensorTwo& a, const TensorTwo& b) {
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.shape.cols != b.shape.rows) {
    throw std::invalid_argument("Inner dimension mismatch in matmul");
  }

  const Shape2 out_shape{a.shape.rows, b.shape.cols};

  if (a.tape == nullptr) {
    if (a.device != b.device) {
      throw std::invalid_argument("Tensors must be on the same device");
    }
    std::vector<Scalar> res(out_shape.numel());
    ops::matmul(a.device, a.data.data(), b.data.data(), res.data(),
                MatMulSpec{a.shape.rows, b.shape.cols, a.shape.cols, false,
                           false, 0});
    return TensorTwo(res, out_shape, false, nullptr, a.device);
  }

  const size_t new_id = a.tape->matmul_two(a.node_id, b.node_id);
  return TensorTwo(a.tape, new_id);
}

}  // namespace autograd
