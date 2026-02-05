#include "../gradino.h"
#include <stdlib.h>

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

enum { SIZE = 1<<16 };

int main(void) {
  // See examples/05_inference for static allocation examples
  void* tapebuf = malloc(tapesize(SIZE));
  tapeinit(SIZE, tapebuf);

  net_t net;
  len_t layer_lens[4] = {3, 4, 4, 1};
  
  // See examples/05_inference for static allocation examples
  len_t netsz = netsize(len(layer_lens), layer_lens);
  void* netbuf = malloc(netsz);
  if (!netbuf) return 1;

  netinit(&net, len(layer_lens), layer_lens, netsz, netbuf);

  vec_t input;
  idx_t data[3] = {vfrom(2), vfrom(3), vfrom(-1)};
  vecinit(&input, len(data), data);

  vec_t result;
  idx_t rdata[1];
  vecinit(&result, len(rdata), rdata);

  netfwd(&net, &input, &result);
  vecdbg(&result, "result");

  const idx_t target = vfrom(1);
  const idx_t mone = vfrom(-1);

  for (int i = 0; i < 20; i++) {
    netfwd(&net, &input, &result);

    idx_t diff = vadd(target, vmul(result.at[0], mone));
    idx_t loss = vmul(diff, diff);

    tapezerograd();
    tapebackprop(loss);
    netgdstep(&net, 0.005);

    vdbg(loss, "loss");
  }

  vecdbg(&result, "result");

  return 0;
}
