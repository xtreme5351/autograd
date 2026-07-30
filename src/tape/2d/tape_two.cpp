//
// Created by Pranav C on 29/07/2026.
//
// Forward and backward for TENSOR_TWO tapes. Structure mirrors tape_one.cpp:
// append a node, record its parents, compute the value eagerly. The only real
// additions are per-node shape tracking (reductions and matmul do not preserve
// their parent's shape) and routing the arithmetic through ops:: so the tape's
// device field decides whether it lands on the CPU or the GPU.

#include <algorithm>  // std::fill
#include <cassert>

#include "backend.h"
#include "tape.h"

namespace autograd {

namespace {
// Every node pushes exactly one entry into each parallel array. Grad slots are
// sized to the *result* numel -- tape_one.cpp sizes them from the parent, which
// only works because 1D ops all preserve shape.
void push_node(Tape& t, const Node node, const bool requires_grad,
               std::vector<Scalar> value, const Shape2 shape) {
  assert(value.size() == shape.numel() && "Value size does not match shape");
  t.nodes.push_back(node);
  t.required_grads.push_back(requires_grad);
  t.values_two.emplace_back(std::move(value));
  t.grad_two.emplace_back(shape.numel(), static_cast<Scalar>(t.seed_grad));
  t.shapes_two.push_back(shape);
}
}  // namespace

// Add a TENSOR_TWO constant node to the tape
size_t Tape::add_node_two(std::vector<Scalar> value, const Shape2 shape,
                          const bool requires_grad) {
  assert(tape_class == TensorClass::TENSOR_TWO);
  const size_t id = nodes.size();
  push_node(*this, Node{CONST, {}}, requires_grad, std::move(value), shape);
  return id;
}

namespace {
// Shared prologue for the three elementwise ops. The result is computed into a
// local buffer *before* anything is pushed, so the references into values_two
// cannot be invalidated by the outer vector reallocating.
size_t elementwise(Tape& t, const NodeType type, const size_t a,
                   const size_t b) {
  assert(t.tape_class == TensorClass::TENSOR_TWO);
  assert(t.shapes_two[a] == t.shapes_two[b] &&
         "Tensor shape mismatch in elementwise operation");

  const Shape2 shape = t.shapes_two[a];
  const size_t n = shape.numel();
  const Scalar* va = t.values_two[a].data();
  const Scalar* vb = t.values_two[b].data();

  std::vector<Scalar> result(n);
  switch (type) {
    case ADD:
      ops::add(t.device, va, vb, result.data(), n);
      break;
    case SUB:
      ops::sub(t.device, va, vb, result.data(), n);
      break;
    case MUL:
      ops::mul(t.device, va, vb, result.data(), n);
      break;
    default:
      assert(false && "Not an elementwise node type");
      break;
  }

  const size_t id = t.nodes.size();
  push_node(t, Node{type, {a, b}},
            t.required_grads[a] || t.required_grads[b], std::move(result),
            shape);
  return id;
}
}  // namespace

size_t Tape::add_two(const size_t a, const size_t b) {
  return elementwise(*this, ADD, a, b);
}

size_t Tape::sub_two(const size_t a, const size_t b) {
  return elementwise(*this, SUB, a, b);
}

size_t Tape::mul_two(const size_t a, const size_t b) {
  return elementwise(*this, MUL, a, b);
}

// (M,K) x (K,N) -> (M,N)
size_t Tape::matmul_two(const size_t a, const size_t b) {
  assert(tape_class == TensorClass::TENSOR_TWO);
  const Shape2 sa = shapes_two[a];
  const Shape2 sb = shapes_two[b];
  assert(sa.cols == sb.rows && "Inner dimension mismatch in matmul");

  const Shape2 shape{sa.rows, sb.cols};
  std::vector<Scalar> result(shape.numel());
  ops::matmul(device, values_two[a].data(), values_two[b].data(), result.data(),
              MatMulSpec{sa.rows, sb.cols, sa.cols, false, false, 0});

  const size_t id = nodes.size();
  push_node(*this, Node{MATMUL, {a, b}},
            required_grads[a] || required_grads[b], std::move(result), shape);
  return id;
}

namespace {
// sum and mean differ only by a scale factor, and both collapse to (1,1) so
// backward has a single scalar to broadcast back over the parent.
size_t reduce(Tape& t, const NodeType type, const size_t a) {
  assert(t.tape_class == TensorClass::TENSOR_TWO);
  const size_t n = t.shapes_two[a].numel();
  assert(n > 0 && "Cannot reduce an empty tensor");

  Scalar total = ops::sum(t.device, t.values_two[a].data(), n);
  if (type == MEAN) total /= static_cast<Scalar>(n);

  const size_t id = t.nodes.size();
  push_node(t, Node{type, {a}}, t.required_grads[a],
            std::vector<Scalar>{total}, Shape2{1, 1});
  return id;
}
}  // namespace

size_t Tape::sum_two(const size_t a) { return reduce(*this, SUM, a); }

size_t Tape::mean_two(const size_t a) { return reduce(*this, MEAN, a); }

void Tape::backward_two(const size_t node_id) {
  assert(tape_class == TensorClass::TENSOR_TWO);
  assert(required_grads[node_id] &&
         "Cannot backprop from a node that does not require grad");

  // Fresh pass: clear any gradients left over from a previous backward() call.
  //
  // Deliberately host-side rather than ops::fill. While tape storage lives in
  // host memory (see backend.h), a device fill would allocate a device buffer,
  // launch a kernel, synchronise and download -- a full round trip per node,
  // with no input to transfer, purely to write zeros. On a 50k-node tape that
  // is 50k round trips the CPU does with a single memset per node. This becomes
  // worth reconsidering only once the grad buffers are device-resident, at
  // which point the fill should stay on whichever side already owns them.
  for (size_t i = 0; i <= node_id; ++i) {
    std::fill(grad_two[i].begin(), grad_two[i].end(), 0.0f);
  }
  std::fill(grad_two[node_id].begin(), grad_two[node_id].end(),
            static_cast<Scalar>(seed_grad));

  // Same argument as backward_one: a parent's id is always less than its
  // child's, so a plain reverse scan is already reverse-topological. Shapes
  // don't affect that, so matmul and reductions need no extra ordering work.
  for (size_t i = node_id + 1; i-- > 0;) {
    // required_grads is monotonic under ||, so if this node doesn't need
    // grad, none of its ancestors do either -- safe to prune.
    if (!required_grads[i]) continue;

    const Node& node = nodes[i];
    if (node.type == CONST) continue;

    const Scalar* g = grad_two[i].data();  // upstream grad for this node
    const size_t gn = shapes_two[i].numel();

    // Parents are read inside each case: SUM and MEAN have only one, so
    // hoisting parents[1] the way backward_one does would be out of bounds.
    switch (node.type) {
      case ADD: {
        const size_t a = node.parents[0], b = node.parents[1];
        ops::axpy(device, 1.0f, g, grad_two[a].data(), gn);
        ops::axpy(device, 1.0f, g, grad_two[b].data(), gn);
        break;
      }
      case SUB: {
        const size_t a = node.parents[0], b = node.parents[1];
        ops::axpy(device, 1.0f, g, grad_two[a].data(), gn);
        ops::axpy(device, -1.0f, g, grad_two[b].data(), gn);
        break;
      }
      case MUL: {
        const size_t a = node.parents[0], b = node.parents[1];
        // Sequential and synchronous, so c = x * x accumulating twice into the
        // same buffer is correct rather than a race.
        ops::mul_add(device, g, values_two[b].data(), grad_two[a].data(), gn);
        ops::mul_add(device, g, values_two[a].data(), grad_two[b].data(), gn);
        break;
      }
      case MATMUL: {
        const size_t a = node.parents[0], b = node.parents[1];
        const Shape2 sa = shapes_two[a];  // (M,K)
        const Shape2 sb = shapes_two[b];  // (K,N)
        // dA(M,K) = g(M,N) * B(K,N)^T
        ops::matmul(device, g, values_two[b].data(), grad_two[a].data(),
                    MatMulSpec{sa.rows, sa.cols, sb.cols, false, true, 1});
        // dB(K,N) = A(M,K)^T * g(M,N)
        ops::matmul(device, values_two[a].data(), g, grad_two[b].data(),
                    MatMulSpec{sb.rows, sb.cols, sa.rows, true, false, 1});
        break;
      }
      case SUM: {
        const size_t a = node.parents[0];
        // d(sum)/dx is 1 everywhere, so the scalar upstream grad broadcasts.
        ops::add_bias(device, g[0], grad_two[a].data(),
                      shapes_two[a].numel());
        break;
      }
      case MEAN: {
        const size_t a = node.parents[0];
        const size_t na = shapes_two[a].numel();
        ops::add_bias(device, g[0] / static_cast<Scalar>(na),
                      grad_two[a].data(), na);
        break;
      }
      default:
        break;
    }
  }
}

}  // namespace autograd
