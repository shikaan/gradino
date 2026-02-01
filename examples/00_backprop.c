#include "../gradino.h"
#include <math.h>
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
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(SIZE, values, grads, ops);

  // Forward pass
  idx_t a = vfrom(2.0);
  idx_t b = vfrom(-3.0);
  idx_t c = vfrom(10.0);
  idx_t f = vfrom(-2.0);
  idx_t e = vmul(a, b);
  idx_t d = vadd(e, c);
  idx_t L = vmul(d, f);

  tbackpass(L);

  asserteqf(grads[L], 1.0);
  asserteqf(grads[d], -2.0);
  asserteqf(grads[e], -2.0);
  asserteqf(grads[c], -2.0);
  asserteqf(grads[b], -4.0);
  asserteqf(grads[a], 6.0);

  return 0;
}
