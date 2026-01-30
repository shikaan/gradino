#include "../gradino.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

enum { SIZE = 1 << 14 };

int main(void) {
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(SIZE, values, grads, ops);

  // A Neural Network of two layers
  // Each layer output size equates to next layer's input size
  net_t net;
  len_t layer_lens[3] = {3, 4, 2};
  layer_t layers[2];
  ptron_t ptrons[8];
  idx_t params[22];
  ninit(&net, len(layer_lens), layer_lens, layers, ptrons, params);

  slice_t input;
  idx_t data[2] = {vinit(2.0), vinit(1.0)};
  slinit(&input, len(data), data);
  sldbg(&input, "input");

  slice_t result;
  idx_t rdata[2];
  slinit(&result, len(rdata), rdata);

  slice_t scratch;
  idx_t sdata[4];
  slinit(&scratch, len(sdata), sdata);

  nactivate(&net, &input, &scratch, &result);
  ndbg(&net, "net");

  sldbg(&result, "result");

  return 0;
}
