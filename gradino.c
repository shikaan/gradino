#include "gradino.h"
#include <math.h>
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
  return (double)((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

static inline idx_t tpushval(value_t val) {
  panicif(TAPE.len >= TAPE.cap, "buffer full (cap=%lu)", TAPE.cap);
  idx_t idx = TAPE.len;
  TAPE.values[idx] = val;
  TAPE.len++;
  return idx;
}

value_t tvalat(idx_t idx) {
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

idx_t tmark(void) { return TAPE.len; }

void treset(idx_t mark) {
  panicif(mark >= TAPE.cap, "invalid mark: expect less than %lu, got %lu",
          TAPE.cap, mark);
  TAPE.len = mark;
}

idx_t vfrom(value_t value) {
  idx_t pushed = tpushval(value);
  TAPE.ops[pushed].type = OP_CONST;
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

idx_t vsub(idx_t a, idx_t b) {
  idx_t pushed = tpushval(tvalat(a) - tvalat(b));
  TAPE.ops[pushed].type = OP_SUB;
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

idx_t vtanh(idx_t a) {
  value_t val = tvalat(a);
  idx_t pushed = tpushval(tanh(val));
  TAPE.ops[pushed].type = OP_TANH;
  TAPE.ops[pushed].input[0] = a;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

void tbackpass(idx_t start) {
  TAPE.grads[start] = 1.0;
  for (idx_t i = start + 1; i-- > 0;) {
    op_t *op = topat(i);
    idx_t in0 = op->input[0];
    idx_t in1 = op->input[1];
    idx_t out = op->output;

    switch (op->type) {
    case OP_CONST:
      break;
    case OP_ADD:
      TAPE.grads[in0] += TAPE.grads[out];
      TAPE.grads[in1] += TAPE.grads[out];
      break;
    case OP_MUL:
      TAPE.grads[in0] += TAPE.grads[out] * TAPE.values[in1];
      TAPE.grads[in1] += TAPE.grads[out] * TAPE.values[in0];
      break;
    case OP_TANH:
      TAPE.grads[in0] +=
          (1.0 - TAPE.values[out] * TAPE.values[out]) * TAPE.grads[out];
      break;
    case OP_SUB:
      TAPE.grads[in0] += TAPE.grads[out];
      TAPE.grads[in1] -= TAPE.grads[out];
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
  case OP_CONST:
    printf("% 4.3f", tvalat(op.input[0]));
    break;
  case OP_ADD:
    printf("% 4.3f + % 4.3f", tvalat(op.input[0]), tvalat(op.input[1]));
    break;
  case OP_MUL:
    printf("% 4.3f * % 4.3f", tvalat(op.input[0]), tvalat(op.input[1]));
    break;
  case OP_TANH:
    printf(" tanh(%4.3f)", tvalat(op.input[0]));
    break;
  case OP_SUB:
    printf("% 4.3f - % 4.3f", tvalat(op.input[0]), tvalat(op.input[1]));
    break;
  default:
    break;
  }
  printf("\n");
}

void slinit(slice_t *sl, idx_t n, idx_t *data) {
  sl->values = data;
  sl->len = n;
}

void pinit(ptron_t *p, idx_t nparams, idx_t *params) {
  panicif(!p, "ptron cannot be empty", NULL);
  panicif(nparams == 0, "must have at least one param", NULL);
  panicif(!params, "must provide params", NULL);

  slinit(p, nparams, params);
  for (idx_t i = 0; i < nparams; i++) {
    p->values[i] = vfrom(vrand());
  }
}

idx_t pactivate(const ptron_t *p, const slice_t *input) {
  panicif(!p, "ptron cannot be null", NULL);
  panicif(!input, "input cannot be null", NULL);
  panicif(input->len != p->len - 1, "invalid input len: expected %lu, got %lu",
          p->len - 1, input->len);

  // dot product
  idx_t sum = vfrom(0);
  for (idx_t i = 0; i < input->len; i++) {
    idx_t w = p->values[i];
    idx_t x = input->values[i];
    idx_t prd = vmul(w, x);
    sum = vadd(sum, prd);
  }

  idx_t activation = vadd(sum, p->values[p->len - 1]);
  return vtanh(activation);
}

void pdbg(ptron_t *p, const char *label) {
  printf("%s\n", label);
  slice_t weights;
  weights.values = p->values;
  weights.len = p->len - 1;
  sldbg(&weights, "w");
  vdbg(p->values[p->len - 1], "b");
}

void sldbg(slice_t *sl, const char *label) {
  printf("%s\n", label);
  for (idx_t i = 0; i < sl->len; i++) {
    char buf[16];
    snprintf(buf, sizeof(buf), "  %s[%lu]", label, i);
    vdbg(sl->values[i], buf);
  }
}

void linit(layer_t *l, idx_t nin, idx_t nout, ptron_t *ptrons, idx_t *params) {
  panicif(!l, "layer cannot be empty", NULL);
  panicif(nin == 0, "input size must be positive", NULL);
  panicif(nout == 0, "output size must be positive", NULL);
  panicif(!ptrons, "must provide ptrons", NULL);
  panicif(!params, "must provide params", NULL);

  l->len = nout;
  l->ptrons = ptrons;

  // nparams for a perceptron is size of input + 1, since the input needs to be
  // as big as the weights
  len_t pnparams = nin + 1;
  for (idx_t i = 0; i < nout; i++) {
    idx_t *pvalues = params + pnparams * i;
    pinit(&ptrons[i], pnparams, pvalues);
  }
}

void lactivate(const layer_t *l, const slice_t *input, slice_t *result) {
  panicif(l->len == 0, "layer is empty", NULL);
  panicif(result->len != l->len, "unexpected result len: expected %lu, got %lu",
          l->len, result->len);
  for (idx_t i = 0; i < l->len; i++) {
    result->values[i] = pactivate(&l->ptrons[i], input);
  }
}

void ldbg(layer_t *l, const char *label) {
  printf("%s\n", label);
  char buf[16];
  for (idx_t i = 0; i < l->len; i++) {
    snprintf(buf, sizeof(buf), "ptron[%lu]", i);
    pdbg(&l->ptrons[i], buf);
  }
}

void ninit(net_t *n, len_t nin, len_t nlayers, len_t *llens, layer_t *layers,
           ptron_t *ptrons, idx_t *params) {
  panicif(!n, "net cannot be empty", NULL);
  panicif(nin == 0, "input size must be positive", NULL);
  panicif(nlayers == 0, "must have at least one layer", NULL);
  panicif(!llens, "must provide layer lengths", NULL);
  panicif(!layers, "must provide layers", NULL);
  panicif(!ptrons, "must provide ptrons", NULL);
  panicif(!params, "must provide params", NULL);

  n->len = nlayers;
  n->layers = layers;

  linit(&layers[0], nin, llens[0], ptrons, params);

  if (nlayers > 1) {
    len_t ptron_offset = llens[0];             // sum of previous nout
    len_t param_offset = llens[0] * (nin + 1); // previous nout * (prev nin + 1)

    ptron_t *lptrons;
    idx_t *lparams;
    for (len_t i = 1; i < nlayers; i++) {
      len_t lin = llens[i - 1];
      len_t lout = llens[i];

      lptrons = ptrons + ptron_offset;
      lparams = params + param_offset;

      linit(&layers[i], lin, lout, lptrons, lparams);

      ptron_offset += lout;
      param_offset += lout * (lin + 1);
    }
  }
}

void nactivate(const net_t *n, const slice_t *input, slice_t *scratch,
               slice_t *result) {
  len_t max = n->layers[0].len;
  for (len_t i = 1; i < n->len; i++) {
    if (n->layers[i].len > max)
      max = n->layers[i].len;
  }

  panicif(scratch->len < max, "invalid scratch len: expected %lu, got %lu", max,
          scratch->len);

  panicif(input->len != n->layers->ptrons[0].len - 1,
          "invalid invalid len: expected %lu, got %lu", max,
          n->layers->ptrons[0].len - 1);

  len_t initlen = scratch->len;
  slice_t linput = *input;
  for (idx_t i = 0; i < n->len - 1; i++) {
    scratch->len = n->layers[i].len;
    lactivate(&n->layers[i], &linput, scratch);
    linput.values = scratch->values;
    linput.len = scratch->len;
  }
  lactivate(&n->layers[n->len - 1], &linput, result);
  scratch->len = initlen;
}

void ndbg(const net_t *n, const char *label) {
  printf("%s\n", label);

  char buf[16];
  for (idx_t i = 0; i < n->len; i++) {
    snprintf(buf, sizeof(buf), "layer[%lu]", i);
    ldbg(&n->layers[i], buf);
  }
}
