#include "../gradino.h"
#include <stdlib.h>

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

int main(void) {
  len_t tapesz = tapesize(1 << 14);
  void *tapebuf = malloc(tapesz);
  tapeinit(1 << 14, tapesz, tapebuf);

  // A Neural Network of two layers and input of size two
  // Each layer output size equates to next layer's input size
  net_t net;
  len_t layer_lens[3] = {2, 4, 2};
  static char netbuf[2048];
  netinit(&net, len(layer_lens), layer_lens, sizeof(netbuf), netbuf);

  vec_t input;
  idx_t data[2] = {vfrom(2.0), vfrom(1.0)};
  vecinit(&input, len(data), data);
  vecdbg(&input, "input");

  vec_t result;
  idx_t rdata[2];
  vecinit(&result, len(rdata), rdata);

  netfwd(&net, &input, &result);
  netdbg(&net, "net");

  vecdbg(&result, "result");

  return 0;
}
