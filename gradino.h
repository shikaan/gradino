#pragma once
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef double value_t;
typedef size_t vptr_t;

typedef enum {
  OP_INIT,
  OP_ADD,
  OP_MUL,
  OP_RELU,
} optype_t;

typedef struct {
  optype_t type;
  vptr_t input[2];
  vptr_t output;
} op_t;

typedef struct {
  value_t *values;
  value_t *grads;
  op_t *ops;
  vptr_t len;
  vptr_t cap;
} tape_t;

typedef struct {
  vptr_t *data;
  vptr_t len;
} buffer_t;

void tinit(tape_t *self, vptr_t len, value_t *data, value_t *grads, op_t *ops);

vptr_t vinit(tape_t *self, value_t a);
vptr_t vadd(tape_t *self, vptr_t a, vptr_t b);
vptr_t vmul(tape_t *self, vptr_t a, vptr_t b);
vptr_t vReLU(tape_t *self, vptr_t a);
void vbackward(tape_t *self, vptr_t start);
void vdbg(tape_t *self, vptr_t a, const char *label);

void bdbg(tape_t* self, buffer_t* b, const char *label);
void binit(buffer_t *b, vptr_t len, vptr_t *data);

// [ weights | bias ]
typedef buffer_t ptron_t;
void pinit(tape_t *self, ptron_t *p, vptr_t len, vptr_t *data);
vptr_t pactivate(tape_t *self, ptron_t *perceptron, buffer_t* input);
void pdbg(tape_t* self, ptron_t* p, const char *label);
