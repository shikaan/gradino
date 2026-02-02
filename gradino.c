#include "gradino.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static tape_t TAPE;

///
/// UTILS
/// =====

#ifdef NDEBUG
#define paniciff(Assertion, Fmt, ...) ((void)0)
#define panicif(Assertion, Fmt) ((void)0)
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

// Abort when a condition is met. Prints a formatted message and a stack trace.
#define paniciff(Assertion, Fmt, ...)                                          \
  if (Assertion) {                                                             \
    char buf[256];                                                             \
    snprintf(buf, sizeof(buf), "%s:%i panic: ", __FILE__, __LINE__);           \
    fputs(buf, stderr);                                                        \
    snprintf(buf, sizeof(buf), Fmt, __VA_ARGS__);                              \
    fputs(buf, stderr);                                                        \
    fputs("\n", stderr);                                                       \
    stacktrace(stderr);                                                        \
    exit(1);                                                                   \
  }

// Abort when a condition is met. Prints a static message and a stack trace.
// Use paniciff for the formatted version
#define panicif(Assertion, Fmt)                                                \
  if (Assertion) {                                                             \
    char buf[256];                                                             \
    snprintf(buf, sizeof(buf), "%s:%i panic: ", __FILE__, __LINE__);           \
    fputs(buf, stderr);                                                        \
    snprintf(buf, sizeof(buf), Fmt);                                           \
    fputs(buf, stderr);                                                        \
    fputs("\n", stderr);                                                       \
    stacktrace(stderr);                                                        \
    exit(1);                                                                   \
  }
#endif

// Abort execution when a codepath that was thought unreacheable is reached
#define unreacheable()                                                         \
  {                                                                            \
    char buf[256];                                                             \
    snprintf(buf, sizeof(buf), "%s:%i reached unreacheable point", __FILE__,   \
             __LINE__);                                                        \
    fputs(buf, stderr);                                                        \
    exit(1);                                                                   \
  }

///
/// TAPE
/// ===

union maxalign {
  value_t value;
  op_t operation;
};

#define MAX_ALIGN sizeof(union maxalign)

size_t tapesize(len_t len) {
  return MAX_ALIGN + (sizeof(value_t) * 2 + sizeof(op_t)) * len;
}

tape_t *tapeinit(idx_t len, char *buffer) {
  uintptr_t addr = (uintptr_t)buffer;
  uintptr_t aligned = (addr + MAX_ALIGN - 1) & ~(MAX_ALIGN - 1);

  void *ptr = (void *)aligned;

  TAPE.values = ptr;
  ptr = (value_t *)ptr + len;

  TAPE.grads = ptr;
  ptr = (value_t *)ptr + len;

  TAPE.ops = ptr;

  TAPE.len = 0;
  TAPE.cap = len;

  // seed the rng
  srand((unsigned)time(NULL));

  return &TAPE;
}

#undef MAX_ALIGN

// The only way to add to the tape is through pushing. This ensures that the
// tape will always be topologically sorted, and backpropagation will work.
static inline idx_t tpushval(value_t val) {
  paniciff(TAPE.len >= TAPE.cap, "buffer full (cap=%lu)", TAPE.cap);
  idx_t idx = TAPE.len;
  TAPE.values[idx] = val;
  TAPE.len++;
  return idx;
}

value_t tapeval(idx_t idx) {
  paniciff(idx >= TAPE.len, "index %lu out of bounds (len=%lu, cap=%lu)", idx,
           TAPE.len, TAPE.cap);
  return TAPE.values[idx];
}

value_t tapegrad(idx_t idx) {
  paniciff(idx >= TAPE.len, "index %lu out of bounds (len=%lu, cap=%lu)", idx,
           TAPE.len, TAPE.cap);
  return TAPE.grads[idx];
}

static op_t tapeop(idx_t idx) {
  paniciff(idx >= TAPE.len, "index %lu out of bounds (len=%lu, cap=%lu)", idx,
           TAPE.len, TAPE.cap);
  return TAPE.ops[idx];
}

idx_t tapemark(void) { return TAPE.len; }

void tapereset(idx_t mark) {
  paniciff(mark >= TAPE.cap, "expected mark less than %lu, got %lu", TAPE.cap,
           mark);
  TAPE.len = mark;
}

///
/// VALUE
/// ===

static value_t vrand(void) {
  return (double)((double)rand() / RAND_MAX) * 2.0 - 1.0;
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
  idx_t pushed = tpushval(tapeval(a) + tapeval(b));
  TAPE.ops[pushed].type = OP_ADD;
  TAPE.ops[pushed].input[0] = a;
  TAPE.ops[pushed].input[1] = b;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

idx_t vsub(idx_t a, idx_t b) {
  idx_t pushed = tpushval(tapeval(a) - tapeval(b));
  TAPE.ops[pushed].type = OP_SUB;
  TAPE.ops[pushed].input[0] = a;
  TAPE.ops[pushed].input[1] = b;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

idx_t vmul(idx_t a, idx_t b) {
  idx_t pushed = tpushval(tapeval(a) * tapeval(b));
  TAPE.ops[pushed].type = OP_MUL;
  TAPE.ops[pushed].input[0] = a;
  TAPE.ops[pushed].input[1] = b;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

idx_t vtanh(idx_t a) {
  value_t val = tapeval(a);
  idx_t pushed = tpushval(tanh(val));
  TAPE.ops[pushed].type = OP_TANH;
  TAPE.ops[pushed].input[0] = a;
  TAPE.ops[pushed].output = pushed;
  TAPE.grads[pushed] = 0;
  return pushed;
}

void tapebackprop(idx_t start) {
  TAPE.grads[start] = 1.0;
  for (idx_t i = start + 1; i-- > 0;) {
    op_t op = tapeop(i);
    idx_t in0 = op.input[0];
    idx_t in1 = op.input[1];
    idx_t out = op.output;

    switch (op.type) {
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
  printf("%s = Value{ % 4.3f | % 4.3f }; ", label, tapeval(a), TAPE.grads[a]);

  printf("// ");

  // Safe. At this point tat would have already panic-ed otherwise
  op_t op = TAPE.ops[a];
  switch (op.type) {
  case OP_CONST:
    printf("% 4.3f", tapeval(op.input[0]));
    break;
  case OP_ADD:
    printf("% 4.3f + % 4.3f", tapeval(op.input[0]), tapeval(op.input[1]));
    break;
  case OP_MUL:
    printf("% 4.3f * % 4.3f", tapeval(op.input[0]), tapeval(op.input[1]));
    break;
  case OP_TANH:
    printf(" tanh(%4.3f)", tapeval(op.input[0]));
    break;
  case OP_SUB:
    printf("% 4.3f - % 4.3f", tapeval(op.input[0]), tapeval(op.input[1]));
    break;
  default:
    break;
  }
  printf("\n");
}

void vecinit(vec_t *sl, idx_t n, idx_t *data) {
  sl->at = data;
  sl->len = n;
}

void pinit(ptron_t *p, idx_t nparams, idx_t *params) {
  panicif(!p, "ptron cannot be empty");
  panicif(nparams == 0, "must have at least one param");
  panicif(!params, "must provide params");

  vecinit(p, nparams, params);
  for (idx_t i = 0; i < nparams; i++) {
    p->at[i] = vfrom(vrand());
  }
}

idx_t pactivate(const ptron_t *p, const vec_t *input) {
  panicif(!p, "ptron cannot be null");
  panicif(!input, "input cannot be null");
  paniciff(input->len != p->len - 1, "invalid input len: expected %lu, got %lu",
           p->len - 1, input->len);

  // dot product
  idx_t sum = vfrom(0);
  for (idx_t i = 0; i < input->len; i++) {
    idx_t w = p->at[i];
    idx_t x = input->at[i];
    idx_t prd = vmul(w, x);
    sum = vadd(sum, prd);
  }

  idx_t activation = vadd(sum, p->at[p->len - 1]);
  return vtanh(activation);
}

void pdbg(ptron_t *p, const char *label) {
  printf("%s\n", label);
  vec_t weights;
  weights.at = p->at;
  weights.len = p->len - 1;
  vecdbg(&weights, "w");
  vdbg(p->at[p->len - 1], "b");
}

void vecdbg(vec_t *sl, const char *label) {
  printf("%s\n", label);
  for (idx_t i = 0; i < sl->len; i++) {
    char buf[32];
    snprintf(buf, sizeof(buf), "  %s[%lu]", label, i);
    vdbg(sl->at[i], buf);
  }
}

void linit(layer_t *l, idx_t nin, idx_t nout, ptron_t *ptrons, idx_t *params) {
  panicif(!l, "layer cannot be empty");
  panicif(nin == 0, "input size must be positive");
  panicif(nout == 0, "output size must be positive");
  panicif(!ptrons, "must provide ptrons");
  panicif(!params, "must provide params");

  l->len = nout;
  l->at = ptrons;

  // nparams for a perceptron is size of input + 1, since the input needs to be
  // as big as the weights
  len_t pnparams = nin + 1;
  for (idx_t i = 0; i < nout; i++) {
    idx_t *pvalues = params + pnparams * i;
    pinit(&ptrons[i], pnparams, pvalues);
  }
}

void lactivate(const layer_t *l, const vec_t *input, vec_t *result) {
  panicif(l->len == 0, "layer is empty");
  paniciff(result->len != l->len,
           "unexpected result len: expected %lu, got %lu", l->len, result->len);
  for (idx_t i = 0; i < l->len; i++) {
    result->at[i] = pactivate(&l->at[i], input);
  }
}

void ldbg(layer_t *l, const char *label) {
  printf("%s\n", label);
  char buf[32];
  for (idx_t i = 0; i < l->len; i++) {
    snprintf(buf, sizeof(buf), "ptron[%lu]", i);
    pdbg(&l->at[i], buf);
  }
}

void ninit(net_t *n, len_t nin, len_t nlayers, len_t *llens, layer_t *layers,
           ptron_t *ptrons, idx_t *params) {
  panicif(!n, "net cannot be empty");
  panicif(nin == 0, "input size must be positive");
  panicif(nlayers == 0, "must have at least one layer");
  panicif(!llens, "must provide layer lengths");
  panicif(!layers, "must provide layers");
  panicif(!ptrons, "must provide ptrons");
  panicif(!params, "must provide params");

  n->len = nlayers;
  n->at = layers;

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

void nactivate(const net_t *n, const vec_t *input, vec_t *scratch,
               vec_t *result) {
  len_t max = n->at[0].len;
  for (len_t i = 1; i < n->len; i++) {
    if (n->at[i].len > max)
      max = n->at[i].len;
  }

  paniciff(scratch->len < max, "invalid scratch len: expected %lu, got %lu",
           max, scratch->len);

  paniciff(input->len != n->at->at[0].len - 1,
           "invalid invalid len: expected %lu, got %lu", max,
           n->at->at[0].len - 1);

  len_t initlen = scratch->len;
  vec_t linput = *input;
  for (idx_t i = 0; i < n->len - 1; i++) {
    scratch->len = n->at[i].len;
    lactivate(&n->at[i], &linput, scratch);
    linput.at = scratch->at;
    linput.len = scratch->len;
  }
  lactivate(&n->at[n->len - 1], &linput, result);
  scratch->len = initlen;
}

void ndbg(const net_t *n, const char *label) {
  printf("%s\n", label);

  char buf[32];
  for (idx_t i = 0; i < n->len; i++) {
    snprintf(buf, sizeof(buf), "layer[%lu]", i);
    ldbg(&n->at[i], buf);
  }
}
