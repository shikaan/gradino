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
  vptr_t params[3];
  pinit(&tape, &ptron, 3, params);
  pdbg(&tape, &ptron, "ptron");

  vptr_t x1 = vinit(&tape, 2.0);
  vptr_t x2 = vinit(&tape, 1.0);

  buffer_t input;
  vptr_t data[2] = {x1, x2};
  binit(&input, 2, data);
  bdbg(&tape, &input, "x");

  // activation
  vptr_t activation = pactivate(&tape, &ptron, &input);
  vdbg(&tape, activation, "activation");

  return 0;
}
