#include "../gradino.h"
#include <stdlib.h>

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

enum { SIZE = 1<<16 };

int main(void) {
  void* BUFFER = malloc(tapesize(SIZE));
  tape_t* tape = tapeinit(SIZE, BUFFER);

  net_t net;
  len_t layer_lens[3] = {4, 4, 1};
  layer_t layers[3];
  ptron_t ptrons[9];
  idx_t params[41];
  ninit(&net, 3, len(layer_lens), layer_lens, layers, ptrons, params);

  vec_t input;
  idx_t data[3] = {vfrom(2), vfrom(3), vfrom(-1)};
  vecinit(&input, len(data), data);

  vec_t result;
  idx_t rdata[1];
  vecinit(&result, len(rdata), rdata);

  vec_t scratch;
  idx_t sdata[4];
  vecinit(&scratch, len(sdata), sdata);

  nactivate(&net, &input, &scratch, &result);
  vecdbg(&result, "result");

  const idx_t target = vfrom(1);
  const idx_t mone = vfrom(-1);

  for (int i = 0; i < 20; i++) {
    nactivate(&net, &input, &scratch, &result);

    idx_t diff = vadd(target, vmul(result.at[0], mone));
    idx_t loss = vmul(diff, diff);

    for (len_t j = 0; j < SIZE; j++)
      tape->grads[j] = 0;

    tbackpass(loss);

    for (len_t j = 0; j < len(params); j++) {
      idx_t idx = params[j];
      tape->values[idx] += tape->grads[idx] * -0.005;
    }

    vdbg(loss, "loss");
  }

  vecdbg(&result, "result");

  return 0;
}
