//
// Created by Pranav C on 29/07/2026.
//

#ifndef AUTOGRAD_TENSOR_TWO_H
#define AUTOGRAD_TENSOR_TWO_H
#include <string>
#include <vector>

#include "tensor_base.h"

// tensor_two refers to a 2D tensor, i.e., a matrix. Data is stored flat in
// row-major order rather than as a vector-of-vectors, so it can be handed to a
// GPU kernel as a single contiguous buffer.

namespace autograd {

class TensorTwo : public TensorBase {
 public:
  std::vector<Scalar> data;  // flat row-major, imp note: this makes a copy
  Shape2 shape;

  // Which backend this tensor's *untaped* ops run on. A taped tensor ignores
  // this and follows tape->device instead -- the tape stays the single source
  // of truth for anything recorded on it, so the two can never disagree about
  // a node. Every constructor keeps that invariant.
  //
  // Defaulted to Device::CPU (via default_device) so existing untaped code does
  // not silently start dispatching to the GPU.
  Device device;

  // Deliberately no `requires_grad` member here: TensorOne redeclares it and
  // ends up shadowing TensorBase::requires_grad, leaving the two permanently
  // out of sync. This class uses the base member only.

  // Constructors, one for data vector + shape, one for shape + init value.
  // The `device` argument applies only when tape == nullptr.
  explicit TensorTwo(const std::vector<Scalar>& data, Shape2 shape,
                     bool requires_grad = false, Tape* tape = nullptr,
                     Device device = default_device);
  explicit TensorTwo(Shape2 shape, Scalar init_value = 0.0f,
                     bool requires_grad = false, Tape* tape = nullptr,
                     Device device = default_device);
  explicit TensorTwo(Tape* tape, size_t node_id);

  [[nodiscard]] std::vector<Scalar> value() const;
  [[nodiscard]] std::vector<Scalar> grad() const;
  [[nodiscard]] size_t size() const;  // total element count
  [[nodiscard]] Shape2 dims() const;
  [[nodiscard]] Scalar at(size_t row, size_t col) const;
  // const: backprop mutates the tape through the pointer, never the handle.
  void backward() const;
  [[nodiscard]] std::string to_string() const;
};

}  // namespace autograd

#endif  // AUTOGRAD_TENSOR_TWO_H
