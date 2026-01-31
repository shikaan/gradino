#include "../gradino.h"
#include <complex.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

enum { SIZE = 16 };

int main(void) {
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(SIZE, values, grads, ops);

  ptron_t ptron;
  idx_t params[2];
  pinit(&ptron, 2, params);
  pdbg(&ptron, "ptron");

  slice_t input;
  idx_t data[2] = {vinit(2.0), vinit(1.0)};
  slinit(&input, 2, data);
  sldbg(&input, "x");

  // activation
  idx_t activation = pactivate(&ptron, &input);
  vdbg(activation, "activation");

  return 0;
}
