#include "../gradino.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

enum { SIZE = 32 };

int main(void) {
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(SIZE, values, grads, ops);

  // Layer of size 3 (# of ptrons) and input size 3 (# params of ptrons)
  // It requires (# of ptrons) * (# of params) values
  layer_t layer;
  ptron_t ptrons[3];
  idx_t params[9];
  linit(&layer, 3, 3, ptrons, params);

  slice_t input;
  idx_t data[2] = {vinit(2.0), vinit(1.0)};
  slinit(&input, 2, data);
  sldbg(&input, "input");

  // activation
  slice_t result;
  idx_t rdata[3];
  slinit(&result, 3, rdata);

  lactivate(&layer, &input, &result);
  ldbg(&layer, "layer");

  sldbg(&result, "result");

  return 0;
}
