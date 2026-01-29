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
    char buf[256];                                                             \
    snprintf(buf, sizeof(buf), "%s:%i panic: ", __FILE__, __LINE__);           \
    fputs(buf, stderr);                                                        \
    snprintf(buf, sizeof(buf), Fmt __VA_OPT__(, ) __VA_ARGS__);                \
    fputs(buf, stderr);                                                        \
    fputs("\n", stderr);                                                       \
    exit(1);                                                                   \
  }
#endif

#define unreacheable()                                                         \
  {                                                                            \
    char buf[256];                                                             \
    snprintf(buf, sizeof(buf), "%s:%i reached unreacheable point", __FILE__,   \
             __LINE__);                                                        \
    fputs(buf, stderr);                                                        \
    exit(1);                                                                   \
  }

static value_t vrand(void) {
  return (double)((double)arc4random() / RAND_MAX) * 2.0 - 1.0;
}

static inline idx_t tpushval(tape_t *self, value_t val) {
  panicif(self->len >= self->cap, "buffer full (cap=%lu)", self->cap);
  idx_t idx = self->len;
  self->values[idx] = val;
  self->len++;
  return idx;
}

static value_t tvalat(tape_t *self, idx_t idx) {
  panicif(idx < 0 || idx >= self->len,
          "index %lu out of bounds (len=%lu, cap=%lu)", idx, self->len,
          self->cap);
  return self->values[idx];
}

static op_t *topat(tape_t *self, idx_t idx) {
  panicif(idx < 0 || idx >= self->len,
          "index %lu out of bounds (len=%lu, cap=%lu)", idx, self->len,
          self->cap);
  return &self->ops[idx];
}

void tinit(tape_t *self, idx_t len, value_t *data, value_t *grads, op_t *ops) {
  self->values = data;
  self->grads = grads;
  self->ops = ops;
  self->len = 0;
  self->cap = len;

  // seed the rng
  sranddev();
}

idx_t vinit(tape_t *self, value_t value) {
  idx_t pushed = tpushval(self, value);
  self->ops[pushed].type = OP_INIT;
  self->ops[pushed].input[0] = pushed;
  self->ops[pushed].output = pushed;
  self->grads[pushed] = 0;
  return pushed;
}

idx_t vadd(tape_t *self, idx_t a, idx_t b) {
  idx_t pushed = tpushval(self, tvalat(self, a) + tvalat(self, b));
  self->ops[pushed].type = OP_ADD;
  self->ops[pushed].input[0] = a;
  self->ops[pushed].input[1] = b;
  self->ops[pushed].output = pushed;
  self->grads[pushed] = 0;
  return pushed;
}

idx_t vmul(tape_t *self, idx_t a, idx_t b) {
  idx_t pushed = tpushval(self, tvalat(self, a) * tvalat(self, b));
  self->ops[pushed].type = OP_MUL;
  self->ops[pushed].input[0] = a;
  self->ops[pushed].input[1] = b;
  self->ops[pushed].output = pushed;
  self->grads[pushed] = 0;
  return pushed;
}

idx_t vReLU(tape_t *self, idx_t a) {
  value_t val = tvalat(self, a);
  idx_t pushed = tpushval(self, val > 0 ? val : 0);
  self->ops[pushed].type = OP_RELU;
  self->ops[pushed].input[0] = a;
  self->ops[pushed].output = pushed;
  self->grads[pushed] = 0;
  return pushed;
}

void vbackward(tape_t *self, idx_t start) {
  self->grads[start] = 1.0;
  for (idx_t i = start + 1; i-- > 0;) {
    op_t *op = topat(self, i);
    idx_t in0 = op->input[0];
    idx_t in1 = op->input[1];
    idx_t out = op->output;

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

void vdbg(tape_t *self, idx_t a, const char *label) {
  printf("%s = Value{ % 4.3f | % 4.3f }; ", label, tvalat(self, a),
         self->grads[a]);

  printf("// ");

  // Safe. At this point tat would have already panic-ed otherwise
  op_t op = self->ops[a];
  switch (op.type) {
  case OP_INIT:
    printf("% 4.3f", tvalat(self, op.input[0]));
    break;
  case OP_ADD:
    printf("% 4.3f + % 4.3f", tvalat(self, op.input[0]),
           tvalat(self, op.input[1]));
    break;
  case OP_MUL:
    printf("% 4.3f * % 4.3f", tvalat(self, op.input[0]),
           tvalat(self, op.input[1]));
    break;
  case OP_RELU:
    printf(" ReLU(%4.3f)", tvalat(self, op.input[0]));
    break;
  default:
    break;
  }
  printf("\n");
}

void slinit(slice_t *p, idx_t len, idx_t *data) {
  p->data = data;
  p->len = len;
}

void pinit(tape_t *self, ptron_t *p, idx_t len, idx_t *data) {
  // TODO: check tape capacity
  slinit(p, len, data);
  for (idx_t i = 0; i < len; i++) {
    p->data[i] = vinit(self, vrand());
  }
}

idx_t pactivate(tape_t *self, ptron_t *p, slice_t *input) {
  panicif(input->len != p->len - 1, "invalid input len: expected %lu, got %lu",
          p->len = 1, input->len);

  // dot product
  idx_t sum = vinit(self, 0);
  for (idx_t i = 0; i < input->len; i++) {
    idx_t w = p->data[i];
    idx_t x = input->data[i];
    idx_t prd = vmul(self, w, x);
    sum = vadd(self, sum, prd);
  }

  idx_t bias = p->data[p->len - 1];
  idx_t activation = vadd(self, sum, bias);
  return vReLU(self, activation);
}

void pdbg(tape_t *self, ptron_t *p, const char *label) {
  printf("  %s\n", label);
  char buf[16];
  for (idx_t i = 0; i < p->len - 1; i++) {
    snprintf(buf, sizeof(buf), "    w[%lu]", i);
    vdbg(self, p->data[i], buf);
  }
  vdbg(self, p->data[p->len - 1], "    b");
}

void sldbg(tape_t *self, slice_t *p, const char *label) {
  printf("%s\n", label);
  for (idx_t i = 0; i < p->len; i++) {
    char buf[16];
    snprintf(buf, sizeof(buf), "  %s[%lu]", label, i);
    vdbg(self, p->data[i], buf);
  }
}

void linit(tape_t *self, layer_t *l, idx_t nptrons, ptron_t *ptrons,
           idx_t nvalues, idx_t *values) {
  l->len = nptrons;
  l->ptrons = ptrons;

  for (idx_t i = 0; i < nptrons; i++) {
    idx_t *pvalues = values + nvalues * i;
    pinit(self, &ptrons[i], nvalues, pvalues);
  }
}

void lactivate(tape_t *self, layer_t *layer, slice_t *input, slice_t *result) {
  panicif(result->len != layer->ptrons[0].len,
          "unexpected result len: expected %lu, got %lu", layer->ptrons[0].len,
          result->len);
  for (idx_t i = 0; i < layer->len; i++) {
    result->data[i] = pactivate(self, &layer->ptrons[i], input);
  }
}

void ldbg(tape_t *self, layer_t *l, const char *label) {
  printf("%s\n", label);

  char buf[16];
  for (idx_t i = 0; i < l->len; i++) {
    snprintf(buf, sizeof(buf), "ptron[%lu]", i);
    pdbg(self, &l->ptrons[i], buf);
  }
}
