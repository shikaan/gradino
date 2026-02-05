#include "../gradino.h"

int main(void) {
  // or 
  // void* BUFFER = malloc(tapesize(64));
  static char tapebuf[4096];
  tapeinit(64, sizeof(tapebuf), tapebuf);

  // Layer of size 3 (# of ptrons) and input size 3 (# weights ptrons)
  // It requires (# of ptrons) * (# of params) values
  layer_t layer;
  ptron_t ptrons[3];
  idx_t params[12];
  linit(&layer, 3, 3, ptrons, params);

  vec_t input;
  idx_t data[3] = {vfrom(2.0), vfrom(1.0), vfrom(-1.0)};
  vecinit(&input, 3, data);
  vecdbg(&input, "input");

  // activation
  vec_t result;
  idx_t rdata[3];
  vecinit(&result, 3, rdata);

  lactivate(&layer, &input, &result);
  ldbg(&layer, "layer");

  vecdbg(&result, "result");

  return 0;
}
