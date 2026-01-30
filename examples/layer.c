#include "../gradino.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

enum { SIZE = 64 };

int main(void) {
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(SIZE, values, grads, ops);

  // Layer of size 3 (# of ptrons) and input size 4 (# params of ptrons)
  // It requires (# of ptrons) * (# of params) values
  layer_t layer;
  ptron_t ptrons[3];
  idx_t params[12];
  linit(&layer, 4, 3, ptrons, params);

  slice_t input;
  idx_t data[3] = {vinit(2.0), vinit(1.0), vinit(-1.0)};
  slinit(&input, 3, data);
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
