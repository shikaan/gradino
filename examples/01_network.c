#include "../gradino.h"
#include <stdlib.h>

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

// This example uses heap allocation via tapecreate/netcreate.
// See 02_training for static allocation via tapeinit/netinit.
int main(void) {
  void *tapebuf = tapecreate(1 << 14);

  // A Neural Network of two layers and input of size two
  // Each layer output size equates to next layer's input size
  len_t layer_lens[3] = {2, 4, 2};
  net_t *net = netcreate(len(layer_lens), layer_lens);

  vec_t input;
  idx_t data[2] = {vfrom(2.0), vfrom(1.0)};
  vecinit(&input, len(data), data);
  vecdbg(&input, "input");

  vec_t result;
  idx_t rdata[2];
  vecinit(&result, len(rdata), rdata);

  netfwd(net, &input, &result);
  netdbg(net, "net");

  vecdbg(&result, "result");

  free(net);
  free(tapebuf);

  return 0;
}
