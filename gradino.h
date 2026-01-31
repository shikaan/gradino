#pragma once
#include <assert.h>
#include <stdbool.h>
#include <stdio.h>

typedef double value_t;
typedef unsigned long idx_t;
typedef idx_t len_t;

typedef enum {
  OP_INIT,
  OP_ADD,
  OP_MUL,
  OP_TANH,
} optype_t;

typedef struct {
  optype_t type;
  idx_t input[2];
  idx_t output;
} op_t;

typedef struct {
  value_t *values;
  value_t *grads;
  op_t *ops;
  len_t len;
  len_t cap;
} tape_t;

typedef struct {
  idx_t *data;
  len_t len;
} slice_t;

// [ ...weights | bias ]
typedef slice_t ptron_t;

typedef struct {
  ptron_t *ptrons;
  len_t len;
} layer_t;

typedef struct {
  layer_t *layers;
  len_t *llens;
  len_t len;
} net_t;

void tinit(idx_t n, value_t *data, value_t *grads, op_t *ops);

idx_t vinit(value_t a);
idx_t vadd(idx_t a, idx_t b);
idx_t vmul(idx_t a, idx_t b);
idx_t vtanh(idx_t a);
void vbackward(idx_t start);
// Prints the value on stdout
void vdbg(idx_t a, const char *label);

// Initialize a slice with memory for data
// `data` must be an array of at least length `n`
void slinit(slice_t *sl, len_t n, idx_t *data);
// Prints the slice on stdout
void sldbg(slice_t *sl, const char *label);

// Intialize a perceptron of size `n` (n-1 wieghts, 1 bias)
void pinit(ptron_t *p, len_t n, idx_t *data);
// Activate the perceptron (with ReLU) against `input`
// `input->len` must be `p->len - 1`
idx_t pactivate(const ptron_t *p, const slice_t *input);
// Prints the perceptron on stdout
void pdbg(ptron_t *p, const char *label);

void pparams(const ptron_t *l, slice_t *params);

// Initialize a layer of `nout` perceptrons of size `nin`
void linit(layer_t *l, len_t nin, len_t nout, ptron_t *ptrons, idx_t *values);
// Activate the perceptrons in the layer against `input`
// `input->len` must be `nin - 1` (see `linit`)
// `result->len` must be `nout`
void lactivate(const layer_t *l, const slice_t *input, slice_t *result);
// Prints the layer on stdout
void ldbg(layer_t *l, const char *label);

void lparams(const layer_t *l, slice_t *params);

void ninit(net_t *n, len_t nlayers, len_t *llens, layer_t *layers,
           ptron_t *ptrons, idx_t *values);

void nactivate(const net_t *n, const slice_t *input, slice_t *scratch,
               slice_t *result);
void ndbg(const net_t *n, const char *label);

void nparams(const net_t *n, slice_t *params);
