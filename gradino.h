#pragma once
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

///
/// CONFIGURATION
/// ===
///
/// The following values are here to be tweaked by you, the user. You can change
/// allocators, use different precision for the values, and different number
/// type as a tape handle.

// Define GRADINO_ALLOC and GRADINO_FREE before including this header
// to use a custom allocator with tapecreate/netcreate.
#ifndef GRADINO_ALLOC
#define GRADINO_ALLOC malloc
#endif

#ifndef GRADINO_FREE
#define GRADINO_FREE free
#endif

// The type of the underlying scalars used in the network.
typedef double value_t;

// Position of a value in the tape. We'll refer to this as values, as we never
// use scalars directly, only their index in the tape.
typedef unsigned long idx_t;

// Represents lengths (for slices and buffers) in the same type as idx_t.
typedef idx_t len_t;

///
/// STRUCTURES
/// ===

// Generic view over an array
#define Slice(Type)                                                            \
  struct {                                                                     \
    len_t len;                                                                 \
    Type *at;                                                                  \
  }

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
///
/// The tape is a global, append-only log of operations. Every math op (vadd,
/// vmul, vtanh, ...) appends a record. This is the foundation for autodiff.
///
/// The tape must be initialized before any other call. Either provide your own
/// buffer or let the library allocate:
///
///   // Option 1: caller-managed buffer
///   static char buf[4096];
///   tapeinit(1024, sizeof(buf), buf);
///
///   // Option 2: heap allocation
///   void *tape = tapecreate(1024);
///   // ... use the tape ...
///   GRADINO_FREE(tape);

// Return the buffer size required for a tape with given capacity.
size_t tapesize(len_t n);
// Initialize global tape with given capacity using provided buffer.
void tapeinit(len_t n, len_t nbuf, char *buffer);
// Allocate and initialize a tape with given capacity. Free with GRADINO_FREE.
void *tapecreate(len_t n);
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
///
/// Values are indices into the tape. Every operation returns a new index
/// and records the operation so gradients can be computed later.
///
///   idx_t a = vfrom(2.0);
///   idx_t b = vfrom(3.0);
///   idx_t c = vmul(a, b);    // c = 6.0
///   tapebackprop(c);
///   tapegrad(a);              // dc/da = 3.0
///   tapegrad(b);              // dc/db = 2.0

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
///
/// A vec_t is a contiguous view over an array of value indices. It does not
/// own the underlying memory.
///
///   vec_t v;
///   idx_t data[3] = {vfrom(1.0), vfrom(2.0), vfrom(3.0)};
///   vecinit(&v, 3, data);

// Initialize a slice view of length n over an idx_t array.
void vecinit(vec_t *vec, len_t n, idx_t *data);
// Debug-print a slice.
void vecdbg(vec_t *vec, const char *label);

///
/// NETWORK
/// ===
///
/// A feed-forward network of dense layers with tanh activation. Layer sizes
/// are specified as an array: {input, hidden..., output}.
///
///   // Option 1: caller-managed buffer
///   len_t layers[] = {2, 4, 1};
///   net_t net;
///   static char buf[2048];
///   netinit(&net, 3, layers, sizeof(buf), buf);
///
///   // Option 2: heap allocation
///   net_t *net = netcreate(3, layers);
///
///   // Forward pass
///   vec_t input, result;
///   idx_t idata[2] = {vfrom(1.0), vfrom(0.5)};
///   idx_t rdata[1];
///   vecinit(&input, 2, idata);
///   vecinit(&result, 1, rdata);
///   netfwd(net, &input, &result);
///
///   // Backprop and update
///   idx_t loss = vsub(result.at[0], vfrom(0.8));
///   tapebackprop(loss);
///   netgdstep(net, 0.01);
///
///   GRADINO_FREE(net);

// Return the buffer size required for a network with given layer sizes.
// nlens is the number of elements in llens, llens[i] is the size of layer i.
size_t netsize(len_t nlens, len_t *llens);
// Initialize a network with given layer sizes using provided buffer.
// nlens is the number of elements in llens, llens[i] is the size of layer i.
void netinit(net_t *n, len_t nlens, len_t *llens, len_t nbuf, char *buffer);
// Allocate and initialize a network with given layer sizes. Free with
// GRADINO_FREE.
net_t *netcreate(len_t nlens, len_t *llens);
// Forward pass through the network.
// Requires: input->len == llens[0], result->len == llens[nlens-1].
void netfwd(net_t *n, const vec_t *input, vec_t *result);
// Performs a gradient descend step. It can be used for both stochastic and
// batch gradient descend.
void netgdstep(const net_t *n, double rate);
// Debug-print a network.
void netdbg(const net_t *n, const char *label);
