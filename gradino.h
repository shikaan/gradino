#pragma once
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

// The type of the underlying scalars used in the network.
typedef double value_t;

// Position of a value in the tape. We'll refer to this as values, as we never
// use scalars directly, only their index in the tape.
typedef unsigned long idx_t;

// Represents lengths (for slices and buffers) in the same type as idx_t.
typedef idx_t len_t;

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

// A contiguous view of value indices.
typedef struct {
  idx_t *values;
  len_t len;
} slice_t;

// Perceptron: slice of parameter indices (weights + bias).
typedef slice_t ptron_t;

// Layer: a slice of perceptrons.
typedef struct {
  ptron_t *ptrons;
  len_t len;
} layer_t;

// Network: a slice of layers.
typedef struct {
  layer_t *layers;
  len_t len;
} net_t;

// Initialize global tape with provided buffers and capacity n.
void tinit(idx_t n, value_t *data, value_t *grads, op_t *ops);
// Read a scalar value from the tape.
value_t tvalat(idx_t idx);
// Checkpoint current tape length.
idx_t tmark(void);
// Reset tape length to a previous checkpoint.
void treset(idx_t mark);
// Backward pass starting from a given value.
void tbackpass(idx_t start);

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

// Initialize a slice view of length n over an idx_t array .
void slinit(slice_t *sl, len_t n, idx_t *data);
// Initialize a perceptron with n params (n-1 weights, 1 bias).
void pinit(ptron_t *p, len_t n, idx_t *params);
// Initialize a layer with nout perceptrons, each with (nin+1) params.
void linit(layer_t *l, len_t nin, len_t nout, ptron_t *ptrons, idx_t *params);
// Initialize a network: input size nin, nlayers with lengths in llens.
void ninit(net_t *n, len_t nin, len_t nlayers, len_t *llens, layer_t *layers,
           ptron_t *ptrons, idx_t *values);

// Forward a perceptron (tanh) over input. Requires: input->len == p->len - 1.
idx_t pactivate(const ptron_t *p, const slice_t *input);
// Forward a layer. Requires: result->len == layer->len.
void lactivate(const layer_t *l, const slice_t *input, slice_t *result);
// Forward a network. Requires: result->len == last_layer->len and
// scratch->len >= max(layers.len).
void nactivate(const net_t *n, const slice_t *input, slice_t *scratch,
               slice_t *result);

// Debug-print a single value.
void vdbg(idx_t a, const char *label);
// Debug-print a slice.
void sldbg(slice_t *sl, const char *label);
// Debug-print a perceptron.
void pdbg(ptron_t *p, const char *label);
// Debug-print a layer.
void ldbg(layer_t *l, const char *label);
// Debug-print a network.
void ndbg(const net_t *n, const char *label);
