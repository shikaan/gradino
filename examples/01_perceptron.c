#include "../gradino.h"

int main(void) {
  static char tapebuf[1024];
  tapeinit(16, sizeof(tapebuf), tapebuf);

  ptron_t ptron;
  idx_t params[3];
  pinit(&ptron, 3, params);
  pdbg(&ptron, "ptron");

  vec_t input;
  idx_t data[2] = {vfrom(2.0), vfrom(1.0)};
  vecinit(&input, 2, data);
  vecdbg(&input, "x");

  // activation
  idx_t activation = pactivate(&ptron, &input);
  vdbg(activation, "activation");

  return 0;
}
