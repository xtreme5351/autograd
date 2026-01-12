//
// Created by Pranav C on 12/01/2026.
//
#include <cassert>
#include "tape.h"

namespace autograd {

size_t Tape::add_node_one(std::vector<double> value, bool requires_grad) {
  assert(tape_class == TensorClass::TENSOR_ONE);
  return 0;
}

}