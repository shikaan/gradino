#include "../gradino.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

enum { SIZE = 32 };

int main(void) {
  tape_t tape;
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(&tape, SIZE, values, grads, ops);

  // Layer of size 3 (# of ptrons) and input size 2 (# of weights per ptron)
  // It requires (# of ptrons) * (# of params) values
  layer_t layer;
  ptron_t ptrons[3];
  idx_t params[9];
  linit(&tape, &layer, 3, ptrons, 3, params);

  idx_t x1 = vinit(&tape, 2.0);
  idx_t x2 = vinit(&tape, 1.0);

  slice_t input;
  idx_t data[2] = {x1, x2};
  slinit(&input, 2, data);
  sldbg(&tape, &input, "input");

  // activation
  slice_t result;
  idx_t rdata[3];
  slinit(&result, 3, rdata);

  lactivate(&tape, &layer, &input, &result);
  ldbg(&tape, &layer, "layer");

  sldbg(&tape, &result, "result");

  return 0;
}
