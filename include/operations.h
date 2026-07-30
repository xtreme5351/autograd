//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_OPERATIONS_H
#define AUTOGRAD_OPERATIONS_H

#include <stdexcept>

#include "backend.h"
#include "tensor_one.h"
#include "tensor_two.h"

namespace autograd {

// addition overloads
TensorOne operator+(const TensorOne& a, double k);
TensorOne operator+(double k, const TensorOne& a);
TensorOne operator+(const TensorOne& a, const TensorOne& b);
TensorOne& operator+=(TensorOne& a, double k);
TensorOne& operator+=(double k, TensorOne& a);
TensorOne& operator+=(TensorOne& a, const TensorOne& b);

// subtraction overloads
TensorOne operator-(const TensorOne& a, double k);
TensorOne operator-(double k, const TensorOne& a);
TensorOne operator-(const TensorOne& a, const TensorOne& b);
TensorOne& operator-=(TensorOne& a, double k);
TensorOne& operator-=(double k, TensorOne& a);
TensorOne& operator-=(TensorOne& a, const TensorOne& b);

// multiplication overloads
TensorOne operator*(const TensorOne& a, const TensorOne& b);

/* ---- 2D tensor operations ---- */

// Shared preconditions for the elementwise binary 2D ops. The device check only
// ever fires for untaped tensors: a taped pair shares a tape, and every
// TensorTwo constructor takes its device from the tape when there is one.
inline void check_binary_two(const TensorTwo& a, const TensorTwo& b) {
  if (a.tape != b.tape) {
    throw std::invalid_argument("Tensors must have the same tape");
  }
  if (a.shape != b.shape) {
    throw std::invalid_argument("Tensors must have the same shape");
  }
  if (a.device != b.device) {
    throw std::invalid_argument("Tensors must be on the same device");
  }
}

// addition overloads
TensorTwo operator+(const TensorTwo& a, Scalar k);
TensorTwo operator+(Scalar k, const TensorTwo& a);
TensorTwo operator+(const TensorTwo& a, const TensorTwo& b);
TensorTwo& operator+=(TensorTwo& a, Scalar k);
TensorTwo& operator+=(TensorTwo& a, const TensorTwo& b);

// subtraction overloads
TensorTwo operator-(const TensorTwo& a, Scalar k);
TensorTwo operator-(Scalar k, const TensorTwo& a);
TensorTwo operator-(const TensorTwo& a, const TensorTwo& b);
TensorTwo& operator-=(TensorTwo& a, Scalar k);
TensorTwo& operator-=(TensorTwo& a, const TensorTwo& b);

// multiplication overloads (elementwise, not matrix product)
TensorTwo operator*(const TensorTwo& a, Scalar k);
TensorTwo operator*(Scalar k, const TensorTwo& a);
TensorTwo operator*(const TensorTwo& a, const TensorTwo& b);
TensorTwo& operator*=(TensorTwo& a, Scalar k);
TensorTwo& operator*=(TensorTwo& a, const TensorTwo& b);

// matrix product: (M,K) x (K,N) -> (M,N)
TensorTwo matmul(const TensorTwo& a, const TensorTwo& b);

// full reductions, both producing a (1,1) tensor
TensorTwo sum(const TensorTwo& a);
TensorTwo mean(const TensorTwo& a);

}  // namespace autograd

#endif  // AUTOGRAD_OPERATIONS_H