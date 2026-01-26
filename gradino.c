#include "gradino.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef NDEBUG
#define panicif(Assertion, Fmt, ...) ((void)0)
#else
#define panicif(Assertion, Fmt, ...)                                           \
  if (Assertion) {                                                             \
    fprintf(stderr, "%s:%i panic: ", __FILE__, __LINE__);                      \
    fprintf(stderr, Fmt __VA_OPT__(, ) __VA_ARGS__);                           \
    fprintf(stderr, "\n");                                                     \
    exit(1);                                                                   \
  }
#endif

#define unreacheable()                                                         \
  {                                                                            \
    fprintf(stderr, "%s:%i reached unreacheable point", __FILE__, __LINE__);   \
    exit(1);                                                                   \
  }

static inline vptr_t tpushval(tape_t *self, value_t val) {
  panicif(self->len >= self->cap, "buffer full (cap=%lu)", self->cap);
  vptr_t idx = self->len;
  self->values[idx] = val;
  self->len++;
  return idx;
}

value_t tat(tape_t *self, vptr_t idx) {
  panicif(idx < 0 || idx >= self->len,
          "index %lu out of bounds (len=%lu, cap=%lu)", idx, self->len,
          self->cap);
  return self->values[idx];
}

static op_t *topat(tape_t *self, vptr_t idx) {
  panicif(idx < 0 || idx >= self->len,
          "index %lu out of bounds (len=%lu, cap=%lu)", idx, self->len,
          self->cap);
  return &self->ops[idx];
}

void tinit(tape_t *self, vptr_t len, value_t *data, value_t *grads, op_t *ops) {
  self->values = data;
  self->grads = grads;
  self->ops = ops;
  self->len = 0;
  self->cap = len;
}

vptr_t vinit(tape_t *self, value_t value) {
  vptr_t pushed = tpushval(self, value);
  self->ops[pushed].type = OP_INIT;
  self->ops[pushed].input[0] = pushed;
  self->ops[pushed].output = pushed;
  return pushed;
}

vptr_t vadd(tape_t *self, vptr_t a, vptr_t b) {
  vptr_t pushed = tpushval(self, tat(self, a) + tat(self, b));
  self->ops[pushed].type = OP_ADD;
  self->ops[pushed].input[0] = a;
  self->ops[pushed].input[1] = b;
  self->ops[pushed].output = pushed;
  return pushed;
}

vptr_t vmul(tape_t *self, vptr_t a, vptr_t b) {
  vptr_t pushed = tpushval(self, tat(self, a) * tat(self, b));
  self->ops[pushed].type = OP_MUL;
  self->ops[pushed].input[0] = a;
  self->ops[pushed].input[1] = b;
  self->ops[pushed].output = pushed;
  return pushed;
}

vptr_t vReLU(tape_t *self, vptr_t a) {
  value_t val = tat(self, a);
  vptr_t pushed = tpushval(self, val > 0 ? val : 0);
  self->ops[pushed].type = OP_RELU;
  self->ops[pushed].input[0] = a;
  self->ops[pushed].output = pushed;
  return pushed;
}

void vback(tape_t *self, vptr_t start) {
  self->grads[start] = 1.0;
  for (vptr_t i = start + 1; i-- > 0;) {
    op_t *op = topat(self, i);
    vptr_t in0 = op->input[0];
    vptr_t in1 = op->input[1];
    vptr_t out = op->output;

    switch (op->type) {
    case OP_INIT:
      break;
    case OP_ADD:
      self->grads[in0] += self->grads[out];
      self->grads[in1] += self->grads[out];
      break;
    case OP_MUL:
      self->grads[in0] += self->grads[out] * self->values[in1];
      self->grads[in1] += self->grads[out] * self->values[in0];
      break;
    case OP_RELU:
      self->grads[in0] += (self->values[out] > 0) * self->grads[out];
      break;
    default:
      unreacheable();
      break;
    }
  }
}

void vdbg(tape_t *self, vptr_t a, const char *label) {
  printf("%s = Value{ % 4.3f | % 4.3f }; ", label, tat(self, a),
         self->grads[a]);

  printf("// ");

  // Safe. At this point tat would have already panic-ed otherwise
  op_t op = self->ops[a];
  switch (op.type) {
  case OP_INIT:
    printf("% 4.3f", tat(self, op.input[0]));
    break;
  case OP_ADD:
    printf("% 4.3f + % 4.3f", tat(self, op.input[0]), tat(self, op.input[1]));
    break;
  case OP_MUL:
    printf("% 4.3f * % 4.3f", tat(self, op.input[0]), tat(self, op.input[1]));
    break;
  case OP_RELU:
    printf(" ReLU(% 4.3f)", tat(self, op.input[0]));
    break;
  default:
    break;
  }

  printf("\n");
}
