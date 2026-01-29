#pragma once
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef double value_t;
typedef size_t idx_t;

typedef enum {
  OP_INIT,
  OP_ADD,
  OP_MUL,
  OP_RELU,
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
  idx_t len;
  idx_t cap;
} tape_t;

typedef struct {
  idx_t *data;
  idx_t len;
} slice_t;

void tinit(tape_t *self, idx_t len, value_t *data, value_t *grads, op_t *ops);

idx_t vinit(tape_t *self, value_t a);
idx_t vadd(tape_t *self, idx_t a, idx_t b);
idx_t vmul(tape_t *self, idx_t a, idx_t b);
idx_t vReLU(tape_t *self, idx_t a);
void vbackward(tape_t *self, idx_t start);
void vdbg(tape_t *self, idx_t a, const char *label);

void sldbg(tape_t *self, slice_t *b, const char *label);
void slinit(slice_t *b, idx_t len, idx_t *data);

// [ ...weights | bias ]
typedef slice_t ptron_t;
void pinit(tape_t *self, ptron_t *p, idx_t len, idx_t *data);
idx_t pactivate(tape_t *self, ptron_t *perceptron, slice_t *input);
void pdbg(tape_t *self, ptron_t *p, const char *label);

typedef struct {
  ptron_t *ptrons;
  idx_t len;
} layer_t;
void linit(tape_t *self, layer_t *l, idx_t nptrons, ptron_t *ptrons,
           idx_t nvalues, idx_t *values);
void lactivate(tape_t *self, layer_t *layer, slice_t *input, slice_t *result);
void ldbg(tape_t *self, layer_t *l, const char *label);
