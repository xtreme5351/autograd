//
// Created by Pranav C on 29/12/2025.
//

#ifndef AUTOGRAD_CONSTS_H
#define AUTOGRAD_CONSTS_H

namespace autograd {
enum NodeType { ADD, SUB, MUL, DIV, CONST };
enum TensorClass { TENSOR_ZERO, TENSOR_ONE };
}

#endif  // AUTOGRAD_CONSTS_H