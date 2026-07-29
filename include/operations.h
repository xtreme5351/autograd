//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_OPERATIONS_H
#define AUTOGRAD_OPERATIONS_H

#include "tensor_one.h"

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

}  // namespace autograd

#endif  // AUTOGRAD_OPERATIONS_H