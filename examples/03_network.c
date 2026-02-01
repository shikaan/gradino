#include "../gradino.h"

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

enum { SIZE = 1 << 14 };

int main(void) {
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(SIZE, values, grads, ops);

  // A Neural Network of two layers
  // Each layer output size equates to next layer's input size
  net_t net;
  len_t layer_lens[2] = {4, 2};
  layer_t layers[2];
  ptron_t ptrons[8];
  idx_t params[22];
  ninit(&net, 2, len(layer_lens), layer_lens, layers, ptrons, params);

  vec_t input;
  idx_t data[2] = {vfrom(2.0), vfrom(1.0)};
  vecinit(&input, len(data), data);
  vecdbg(&input, "input");

  vec_t result;
  idx_t rdata[2];
  vecinit(&result, len(rdata), rdata);

  vec_t scratch;
  idx_t sdata[4];
  vecinit(&scratch, len(sdata), sdata);

  nactivate(&net, &input, &scratch, &result);
  nactivate(&net, &input, &scratch, &result);
  nactivate(&net, &input, &scratch, &result);
  nactivate(&net, &input, &scratch, &result);
  ndbg(&net, "net");

  vecdbg(&result, "result");

  return 0;
}
