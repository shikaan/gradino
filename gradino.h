#pragma once
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define Slice(Type)                                                            \
  struct {                                                                     \
    len_t len;                                                                 \
    Type *at;                                                                  \
  }

// The type of the underlying scalars used in the network.
typedef double value_t;

// Position of a value in the tape. We'll refer to this as values, as we never
// use scalars directly, only their index in the tape.
typedef unsigned long idx_t;

// Represents lengths (for slices and buffers) in the same type as idx_t.
typedef idx_t len_t;

// A contiguous view of value indices.
typedef Slice(idx_t) vec_t;

// Operation kinds recorded on the tape.
typedef enum {
  OP_CONST,
  OP_ADD,
  OP_SUB,
  OP_MUL,
  OP_TANH,
} optype_t;

// A single operation node on the tape.
typedef struct {
  optype_t type;
  idx_t input[2];
  idx_t output;
} op_t;

// Global tape holding values, gradients, and operations.
typedef struct {
  value_t *values;
  value_t *grads;
  op_t *ops;
  len_t len;
  len_t cap;
} tape_t;

// Perceptron: slice of parameter indices (weights + bias).
typedef vec_t ptron_t;

// Layer: a slice of perceptrons.
typedef Slice(ptron_t) layer_t;

// Network: a slice of layers.
typedef struct {
  Slice(layer_t) layers;
  vec_t params;
  vec_t scratch;
} net_t;

///
/// TAPE
/// ===

size_t tapesize(len_t len);
// Initialize global tape with provided buffers and capacity n.
void tapeinit(idx_t len, len_t nbuf, char *buffer);
// Read a value from the tape.
value_t tapeval(idx_t idx);
// Read the gradient of a value from the tape.
// It will be zero until a tapebackprop is called.
value_t tapegrad(idx_t idx);
// Checkpoint current tape length. Use the mark in tapereset to
// optimize tape usage.
idx_t tapemark(void);
// Reset tape length to a previous checkpoint.
void tapereset(idx_t mark);
// Calculate gradient components in the tape via backpropagation from start.
void tapebackprop(idx_t start);
// Zero the gradient component of all the values in the tape.
void tapezerograd(void);

///
/// VALUE
/// ===

// Push a constant scalar onto the tape.
idx_t vfrom(value_t a);
// Add two recorded values.
idx_t vadd(idx_t a, idx_t b);
// Multiply two recorded values.
idx_t vmul(idx_t a, idx_t b);
// Subtract two recorded values.
idx_t vsub(idx_t a, idx_t b);
// Apply tanh to a recorded value.
idx_t vtanh(idx_t a);
// Debug-print a single value.
void vdbg(idx_t a, const char *label);

///
/// VECTOR
/// ===

// Initialize a slice view of length n over an idx_t array.
void vecinit(vec_t *vec, len_t n, idx_t *data);
// Debug-print a slice.
void vecdbg(vec_t *vec, const char *label);

///
/// PERCEPTRON
/// ===

// Initialize a perceptron with n params (n-1 weights, 1 bias).
void pinit(ptron_t *p, len_t n, idx_t *params);
// Forward a perceptron (tanh) over input. Requires: input->len == p->len - 1.
idx_t pactivate(const ptron_t *p, const vec_t *input);
// Debug-print a perceptron.
void pdbg(ptron_t *p, const char *label);

///
/// LAYER
/// ===

// Initialize a layer with nout perceptrons, each with (nin+1) params.
void linit(layer_t *l, len_t nin, len_t nout, ptron_t *ptrons, idx_t *params);
// Forward a layer. Requires: result->len == layer->len.
void lactivate(const layer_t *l, const vec_t *input, vec_t *result);
// Debug-print a layer.
void ldbg(layer_t *l, const char *label);

///
/// NETWORK
/// ===

size_t netsize(len_t nlens, len_t *llens);
void netinit(net_t *n, len_t nlens, len_t *llens, len_t nbuf, char *buffer);
// Forward a network. Requires: result->len == last_layer->len and
// scratch->len >= max(layers.len).
void netfwd(net_t *n, const vec_t *input, vec_t *result);
// Performs a gradient descend step. It can be used for both stochastic and
// batch gradient descend.
void netgdstep(const net_t *n, double rate);
// Debug-print a network.
void netdbg(const net_t *n, const char *label);
