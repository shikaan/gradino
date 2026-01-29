#include "../gradino.h"
#include <complex.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

enum { SIZE = 16 };

int main(void) {
  tape_t tape;
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(&tape, SIZE, values, grads, ops);

  ptron_t ptron;
  idx_t params[3];
  pinit(&tape, &ptron, 3, params);
  pdbg(&tape, &ptron, "ptron");

  idx_t x1 = vinit(&tape, 2.0);
  idx_t x2 = vinit(&tape, 1.0);

  slice_t input;
  idx_t data[2] = {x1, x2};
  slinit(&input, 2, data);
  sldbg(&tape, &input, "x");

  // activation
  idx_t activation = pactivate(&tape, &ptron, &input);
  vdbg(&tape, activation, "activation");

  return 0;
}
