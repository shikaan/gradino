#include "gradino.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static tape_t TAPE;

#ifdef NDEBUG
#define panicif(Assertion, Fmt, ...) ((void)0)
#else
#include <execinfo.h>

static void stacktrace(FILE *out) {
  void *addrs[128];
  int n = backtrace(addrs, 128);
  if (n <= 0)
    return;

  char **syms = backtrace_symbols(addrs, n);
  if (syms) {
    for (int i = 1; i < n; ++i) {
      fprintf(out, "  %s\n", syms[i]);
    }
    free(syms);
  }
}

#define panicif(Assertion, Fmt, ...)                                           \
  if (Assertion) {                                                             \
    char buf[256];                                                             \
    snprintf(buf, sizeof(buf), "%s:%i panic: ", __FILE__, __LINE__);           \
    fputs(buf, stderr);                                                        \
    snprintf(buf, sizeof(buf), Fmt __VA_OPT__(, ) __VA_ARGS__);                \
    fputs(buf, stderr);                                                        \
    fputs("\n", stderr);                                                       \
    stacktrace(stderr);                                                        \
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

static inline idx_t tpushval(value_t val) {
  panicif(TAPE.len >= TAPE.cap, "buffer full (cap=%lu)", TAPE.cap);
  idx_t idx = TAPE.len;
  TAPE.values[idx] = val;
  TAPE.len++;
  return idx;
}

static value_t tvalat(idx_t idx) {
  panicif(idx < 0 || idx >= TAPE.len,
          "index %lu out of bounds (len=%lu, cap=%lu)", idx, TAPE.len,
          TAPE.cap);
  return TAPE.values[idx];
}

static op_t *topat(idx_t idx) {
  panicif(idx < 0 || idx >= TAPE.len,
          "index %lu out of bounds (len=%lu, cap=%lu)", idx, TAPE.len,
          TAPE.cap);
  return &TAPE.ops[idx];
}

void tinit(idx_t len, value_t *data, value_t *grads, op_t *ops) {
  TAPE.values = data;
  TAPE.grads = grads;
  TAPE.ops = ops;
  TAPE.len = 0;
  TAPE.cap = len;

  // seed the rng
  sranddev();
}

idx_t vinit(value_t value) {
  idx_t pushed = tpushval(value);
  TAPE.ops[pushed].type = OP_INIT;
  TAPE.ops[pushed].input[0] = pushed;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

idx_t vadd(idx_t a, idx_t b) {
  idx_t pushed = tpushval(tvalat(a) + tvalat(b));
  TAPE.ops[pushed].type = OP_ADD;
  TAPE.ops[pushed].input[0] = a;
  TAPE.ops[pushed].input[1] = b;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

idx_t vmul(idx_t a, idx_t b) {
  idx_t pushed = tpushval(tvalat(a) * tvalat(b));
  TAPE.ops[pushed].type = OP_MUL;
  TAPE.ops[pushed].input[0] = a;
  TAPE.ops[pushed].input[1] = b;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

idx_t vReLU(idx_t a) {
  value_t val = tvalat(a);
  idx_t pushed = tpushval(val > 0 ? val : 0);
  TAPE.ops[pushed].type = OP_RELU;
  TAPE.ops[pushed].input[0] = a;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

void vbackward(idx_t start) {
  TAPE.grads[start] = 1.0;
  for (idx_t i = start + 1; i-- > 0;) {
    op_t *op = topat(i);
    idx_t in0 = op->input[0];
    idx_t in1 = op->input[1];
    idx_t out = op->output;

    switch (op->type) {
    case OP_INIT:
      break;
    case OP_ADD:
      TAPE.grads[in0] += TAPE.grads[out];
      TAPE.grads[in1] += TAPE.grads[out];
      break;
    case OP_MUL:
      TAPE.grads[in0] += TAPE.grads[out] * TAPE.values[in1];
      TAPE.grads[in1] += TAPE.grads[out] * TAPE.values[in0];
      break;
    case OP_RELU:
      TAPE.grads[in0] += (TAPE.values[out] > 0) * TAPE.grads[out];
      break;
    default:
      unreacheable();
      break;
    }
  }
}

void vdbg(idx_t a, const char *label) {
  printf("%s = Value{ % 4.3f | % 4.3f }; ", label, tvalat(a), TAPE.grads[a]);

  printf("// ");

  // Safe. At this point tat would have already panic-ed otherwise
  op_t op = TAPE.ops[a];
  switch (op.type) {
  case OP_INIT:
    printf("% 4.3f", tvalat(op.input[0]));
    break;
  case OP_ADD:
    printf("% 4.3f + % 4.3f", tvalat(op.input[0]), tvalat(op.input[1]));
    break;
  case OP_MUL:
    printf("% 4.3f * % 4.3f", tvalat(op.input[0]), tvalat(op.input[1]));
    break;
  case OP_RELU:
    printf(" ReLU(%4.3f)", tvalat(op.input[0]));
    break;
  default:
    break;
  }
  printf("\n");
}

void slinit(slice_t *sl, idx_t n, idx_t *data) {
  // TODO: check tape capacity
  sl->data = data;
  sl->len = n;
}

void pinit(ptron_t *p, idx_t n, idx_t *data) {
  slinit(p, n, data);
  for (idx_t i = 0; i < n; i++) {
    p->data[i] = vinit(vrand());
  }
}

idx_t pactivate(ptron_t *p, slice_t *input) {
  panicif(input->len != p->len - 1, "invalid input len: expected %lu, got %lu",
          p->len - 1, input->len);

  // dot product
  idx_t sum = vinit(0);
  for (idx_t i = 0; i < input->len; i++) {
    idx_t w = p->data[i];
    idx_t x = input->data[i];
    idx_t prd = vmul(w, x);
    sum = vadd(sum, prd);
  }

  idx_t bias = p->data[p->len - 1];
  idx_t activation = vadd(sum, bias);
  return vReLU(activation);
}

void pdbg(ptron_t *p, const char *label) {
  printf("    %s\n", label);
  char buf[16];
  for (idx_t i = 0; i < p->len - 1; i++) {
    snprintf(buf, sizeof(buf), "      w[%lu]", i);
    vdbg(p->data[i], buf);
  }
  vdbg(p->data[p->len - 1], "      b");
}

void sldbg(slice_t *sl, const char *label) {
  printf("%s\n", label);
  for (idx_t i = 0; i < sl->len; i++) {
    char buf[16];
    snprintf(buf, sizeof(buf), "  %s[%lu]", label, i);
    vdbg(sl->data[i], buf);
  }
}

void linit(layer_t *l, idx_t nin, idx_t nout, ptron_t *ptrons, idx_t *values) {
  // TODO: check tape capacity
  l->len = nout;
  l->ptrons = ptrons;

  for (idx_t i = 0; i < nout; i++) {
    idx_t *pvalues = values + nin * i;
    pinit(&ptrons[i], nin, pvalues);
  }
}

void lactivate(layer_t *l, slice_t *input, slice_t *result) {
  panicif(l->len == 0, "layer is empty", NULL);
  panicif(result->len != l->len, "unexpected result len: expected %lu, got %lu",
          l->len, result->len);
  for (idx_t i = 0; i < l->len; i++) {
    result->data[i] = pactivate(&l->ptrons[i], input);
  }
}

void ldbg(layer_t *l, const char *label) {
  printf("  %s\n", label);

  char buf[16];
  for (idx_t i = 0; i < l->len; i++) {
    snprintf(buf, sizeof(buf), "ptron[%lu]", i);
    pdbg(&l->ptrons[i], buf);
  }
}

void ninit(net_t *n, len_t nlayers, len_t *llens, layer_t *layers,
           ptron_t *ptrons, idx_t *params) {
  // TODO: preconditions
  n->len = nlayers - 1;
  n->layers = layers;
  n->llens = llens;

  linit(&layers[0], llens[0], llens[1], ptrons, params);

  ptron_t *lptrons = ptrons;
  idx_t *lparams = params;
  for (len_t i = 1; i < nlayers - 1; i++) {
    len_t lin = llens[i] + 1; // number of weights plus the bias
    len_t lout = llens[i + 1];
    lptrons += llens[i];
    lparams += llens[i - 1] * llens[i];

    linit(&layers[i], lin, lout, lptrons, lparams);
  }
}

void nactivate(net_t *n, slice_t *input, slice_t *scratch, slice_t *result) {
  // TODO: preconditions
  len_t max = n->llens[0];
  for (len_t i = 1; i < n->len + 1; i++) {
    if (n->llens[i] > max)
      max = n->llens[i];
  }

  panicif(scratch->len < max, "invalid input len: expected %lu, got %lu", max,
          scratch->len);

  len_t initlen = scratch->len;
  for (idx_t i = 0; i < n->len - 1; i++) {
    scratch->len = n->llens[i + 1];
    lactivate(&n->layers[i], input, scratch);
    input->data = scratch->data;
    input->len = scratch->len;
  }
  lactivate(&n->layers[n->len - 1], input, result);
  scratch->len = initlen;
}

void ndbg(net_t* n, const char* label) {
  printf("%s\n", label);

  char buf[16];
  for (idx_t i = 0; i < n->len; i++) {
    snprintf(buf, sizeof(buf), "layer[%lu]", i);
    ldbg(&n->layers[i], buf);
  }
}
