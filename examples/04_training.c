#include "../gradino.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

enum { SIZE = 1 << 16 };

int main(void) {
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(SIZE, values, grads, ops);

  net_t net;
  len_t layer_lens[3] = {4, 4, 1};
  layer_t layers[3];
  ptron_t ptrons[9];
  idx_t params[41];
  ninit(&net, 3, len(layer_lens), layer_lens, layers, ptrons, params);

  slice_t input;
  idx_t data[3] = {vfrom(2), vfrom(3), vfrom(-1)};
  slinit(&input, len(data), data);

  slice_t result;
  idx_t rdata[1];
  slinit(&result, len(rdata), rdata);

  slice_t scratch;
  idx_t sdata[4];
  slinit(&scratch, len(sdata), sdata);

  nactivate(&net, &input, &scratch, &result);
  sldbg(&result, "result");

  const idx_t target = vfrom(1);
  const idx_t mone = vfrom(-1);

  for (int i = 0; i < 20; i++) {
    nactivate(&net, &input, &scratch, &result);

    idx_t diff = vadd(target, vmul(result.values[0], mone));
    idx_t loss = vmul(diff, diff);

    for (len_t j = 0; j < SIZE; j++)
      grads[j] = 0;

    tbackpass(loss);

    for (len_t j = 0; j < len(params); j++) {
      idx_t idx = params[j];
      values[idx] += grads[idx] * -0.005;
    }

    vdbg(loss, "loss");
  }

  sldbg(&result, "result");

  return 0;
}
