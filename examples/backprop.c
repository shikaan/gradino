#include "../gradino.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define asserteqf(actual, expected)                                            \
  if (fabs(actual - expected) > 1e-6) {                                        \
    fprintf(stderr, "%s:%d Assertion error: expected %f; got %f\n", __FILE__,  \
            __LINE__, expected, actual);                                       \
    exit(1);                                                                   \
  }

enum { SIZE = 8 };

int main(void) {
  tape_t tape;
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(&tape, SIZE, values, grads, ops);

  vptr_t a = vinit(&tape, 2.0);
  vptr_t b = vinit(&tape, -3.0);
  vptr_t c = vinit(&tape, 10.0);
  vptr_t f = vinit(&tape, -2.0);
  vptr_t e = vmul(&tape, a, b);
  vptr_t d = vadd(&tape, e, c);
  vptr_t L = vmul(&tape, d, f);

  vback(&tape, L);

  asserteqf(grads[L], 1.0);
  asserteqf(grads[d], -2.0);
  asserteqf(grads[e], -2.0);
  asserteqf(grads[c], -2.0);
  asserteqf(grads[b], -4.0);
  asserteqf(grads[a], 6.0);

  return 0;
}
