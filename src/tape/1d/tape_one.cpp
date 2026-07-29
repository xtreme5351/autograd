//
// Created by Pranav C on 12/01/2026.
//
#include <cassert>

#include "tape.h"

namespace autograd {

// Add a TENSOR_ONE constant node to the tape
size_t Tape::add_node_one(std::vector<double> value, const bool requires_grad) {
  assert(tape_class == TensorClass::TENSOR_ONE);
  const size_t id =
      nodes.size();  // new node id is current size of nodes vector

  nodes.push_back(Node{CONST, {}});

  // store node info
  required_grads.push_back(requires_grad);
  const size_t n = value.size();
  values_one.emplace_back(std::move(value));  // store a copy of value

  grad_one.emplace_back(n, seed_grad);
  return id;
}

// Addition of two TENSOR_ONE tensors
size_t Tape::add_one(size_t a, size_t b) {
  assert(tape_class == TensorClass::TENSOR_ONE);

  const std::vector<double> value_one = values_one[a];
  const std::vector<double> value_two = values_one[b];

  assert(value_one.size() == value_two.size() &&
         "Tensor size mismatch in addition");

  const size_t id =
      nodes.size();  // new node id is current size of nodes vector
  const size_t n = value_one.size();

  nodes.push_back(Node{ADD, {a, b}});

  // store node info
  required_grads.push_back(required_grads[a] || required_grads[b]);
  std::vector<double> result(n);

  for (size_t i = 0; i < n; ++i) {
    result[i] = value_one[i] + value_two[i];
  }

  values_one.emplace_back(std::move(result));  // store a copy of value
  grad_one.emplace_back(value_one.size(), seed_grad);
  return id;
}

// Subtraction of two TENSOR_ONE tensors, tensor a - tensor b
size_t Tape::sub_one(size_t a, size_t b) {
  assert(tape_class == TensorClass::TENSOR_ONE);

  const std::vector<double> value_one = values_one[a];
  const std::vector<double> value_two = values_one[b];

  assert(value_one.size() == value_two.size() &&
         "Tensor size mismatch in subtraction");

  const size_t id =
      nodes.size();  // new node id is current size of nodes vector

  nodes.push_back(Node{SUB, {a, b}});

  // store node info
  required_grads.push_back(required_grads[a] || required_grads[b]);
  std::vector<double> result(value_one.size());

  for (size_t i = 0; i < value_one.size(); ++i) {
    result[i] = value_one[i] - value_two[i];
  }

  values_one.emplace_back(std::move(result));  // store a copy of value
  grad_one.emplace_back(value_one.size(), seed_grad);
  return id;
}

// Multiplication of two TENSOR_ONE tensors
size_t Tape::mul_one(size_t a, size_t b) {
  assert(tape_class == TensorClass::TENSOR_ONE);

  const std::vector<double> value_one = values_one[a];
  const std::vector<double> value_two = values_one[b];

  assert(value_one.size() == value_two.size() &&
         "Tensor size mismatch in multiplication");

  const size_t id =
      nodes.size();  // new node id is current size of nodes vector

  nodes.push_back(Node{MUL, {a, b}});

  // store node info
  required_grads.push_back(required_grads[a] || required_grads[b]);
  std::vector<double> result(value_one.size());

  for (size_t i = 0; i < value_one.size(); ++i) {
    result[i] = value_one[i] * value_two[i];
  }

  values_one.emplace_back(std::move(result));  // store a copy of value
  grad_one.emplace_back(value_one.size(), seed_grad);
  return id;
}

void Tape::backward(const size_t node_id) {
  switch (tape_class) {
    case TensorClass::TENSOR_ONE:
      return this->backward_one(node_id);
    case TensorClass::TENSOR_TWO:
      return this->backward_two(node_id);
    default:
      break;
  }
}

void Tape::backward_one(const size_t node_id) {
  assert(tape_class == TensorClass::TENSOR_ONE);
  assert(required_grads[node_id] &&
         "Cannot backprop from a node that does not require grad");

  // Fresh pass: clear any gradients left over from a previous backward() call.
  for (size_t i = 0; i <= node_id; ++i) {
    std::ranges::fill(grad_one[i], 0.0);
  }
  std::ranges::fill(grad_one[node_id], seed_grad);

  // nodes[] is already topologically ordered (a parent's id is always less
  // than its child's), so a plain reverse scan is a valid reverse-topological
  // order -- no separate topo sort or visited-set needed.
  for (size_t i = node_id + 1; i-- > 0;) {
    // required_grads is monotonic under ||, so if this node doesn't need
    // grad, none of its ancestors do either -- safe to prune.
    if (!required_grads[i]) continue;

    const auto& [node_type, parents] = nodes[i];
    if (node_type == CONST) continue;

    const size_t a = parents[0];
    const size_t b = parents[1];
    const std::vector<double>& g =
        grad_one[i];  // upstream grad for this node's output

    switch (node_type) {
      case ADD:
        for (size_t k = 0; k < g.size(); ++k) {
          grad_one[a][k] += g[k];
          grad_one[b][k] += g[k];
        }
        break;
      case SUB:
        for (size_t k = 0; k < g.size(); ++k) {
          grad_one[a][k] += g[k];
          grad_one[b][k] -= g[k];
        }
        break;
      case MUL:
        for (size_t k = 0; k < g.size(); ++k) {
          grad_one[a][k] += g[k] * values_one[b][k];
          grad_one[b][k] += g[k] * values_one[a][k];
        }
        break;
      default:
        break;
    }
  }
}

}  // namespace autograd