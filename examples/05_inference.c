#include "../gradino.h"
#include <ctype.h>

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

enum { SIZE = 1024, EPOCHS = 10000 };

typedef struct {
  vec_t input;
  vec_t target;
} sample_t;

static sample_t samples[14];
void prepare(void) {
#define sample(Idx, A, B, C, D, E, F, G)                                       \
  static idx_t input##Idx[7];                                                  \
  static idx_t target##Idx[11];                                                \
  input##Idx[0] = vfrom(A);                                                    \
  input##Idx[1] = vfrom(B);                                                    \
  input##Idx[2] = vfrom(C);                                                    \
  input##Idx[3] = vfrom(D);                                                    \
  input##Idx[4] = vfrom(E);                                                    \
  input##Idx[5] = vfrom(F);                                                    \
  input##Idx[6] = vfrom(G);                                                    \
  samples[Idx].input.len = 7;                                                  \
  samples[Idx].input.at = input##Idx;                                          \
  samples[Idx].target.len = 11;                                                \
  for (int i = 0; i < 11; i++) {                                               \
    target##Idx[i] = vfrom(i == Idx ? 1.0 : -1.0);                             \
  }                                                                            \
  samples[Idx].target.at = target##Idx;

  sample(0, 1, 1, 1, 1, 1, 1, -1);
  sample(1, -1, 1, 1, -1, -1, -1, -1);
  sample(2, 1, 1, -1, 1, 1, -1, 1);
  sample(3, 1, 1, 1, 1, -1, -1, 1);
  sample(4, -1, 1, 1, -1, -1, 1, 1);
  sample(5, 1, -1, 1, 1, -1, 1, 1);
  sample(6, 1, -1, 1, 1, 1, 1, 1);
  sample(7, 1, 1, 1, -1, -1, -1, -1);
  sample(8, 1, 1, 1, 1, 1, 1, 1);
  sample(9, 1, 1, 1, 1, -1, 1, 1);

#define invalid_sample(Idx, A, B, C, D, E, F, G)                               \
  static idx_t input##Idx[7];                                                  \
  static idx_t target##Idx[11];                                                \
  input##Idx[0] = vfrom(A);                                                    \
  input##Idx[1] = vfrom(B);                                                    \
  input##Idx[2] = vfrom(C);                                                    \
  input##Idx[3] = vfrom(D);                                                    \
  input##Idx[4] = vfrom(E);                                                    \
  input##Idx[5] = vfrom(F);                                                    \
  input##Idx[6] = vfrom(G);                                                    \
  samples[Idx].input.len = 7;                                                  \
  samples[Idx].input.at = input##Idx;                                          \
  samples[Idx].target.len = 11;                                                \
  for (int i = 0; i < 11; i++) {                                               \
    target##Idx[i] = vfrom(i == 10 ? 1.0 : -1.0);                              \
  }                                                                            \
  samples[Idx].target.at = target##Idx;

  invalid_sample(10, 1, -1, -1, -1, 1, -1, 1);
  invalid_sample(11, -1, -1, 1, -1, 1, -1, -1);
  invalid_sample(12, -1, 1, -1, 1, -1, 1, -1);
  invalid_sample(13, 1, -1, 1, -1, 1, -1, -1);

#undef invalid_sample
#undef sample
}

void prompt(net_t *net, vec_t *scratch, vec_t *result, idx_t mark) {
  while (1) {
    treset(mark);
    printf("enter a 7-bit sequence (e.g., 0110000 for 1, 1101101 for 2): ");

    char buf[16];
    fgets(buf, sizeof(buf), stdin);

    char raw[7];
    size_t rlen = 0;
    for (size_t i = 0; i < sizeof(buf) && rlen < 7; i++) {
      if (!isspace(buf[i]))
        raw[rlen++] = buf[i];
    }

    idx_t xvals[7];
    for (int b = 0; b < 7; b++) {
      value_t xv = (raw[b] == '1') ? 1.0 : -1.0;
      xvals[b] = vfrom(xv);
    }
    vec_t input;
    vecinit(&input, 7, xvals);

    nactivate(net, &input, scratch, result);

    len_t pred = 0;
    value_t best_val = tvalat(result->at[0]);
    for (len_t k = 1; k < 11; k++) {
      value_t v = tvalat(result->at[k]);
      if (v > best_val) {
        best_val = v;
        pred = k;
      }
    }

    if (pred == 10) {
      printf("~> invalid\n");
    } else {
      printf("~> %lu\n", pred);
    }
  }
}

int main(void) {
  value_t values[SIZE], grads[SIZE];
  op_t ops[SIZE];
  tinit(SIZE, values, grads, ops);

  prepare();

  net_t net;
  len_t llens[2] = {8, 11};
  layer_t layers[2];
  ptron_t ptrons[19];
  idx_t params[163];
  ninit(&net, 7, len(llens), llens, layers, ptrons, params);
  idx_t mark = tmark();

  vec_t result;
  idx_t rdata[11];
  vecinit(&result, len(rdata), rdata);

  vec_t scratch;
  idx_t sdata[11];
  vecinit(&scratch, len(sdata), sdata);

  puts("Training the network. This might take some seconds...");
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
#ifndef NDEBUG
    value_t epoch_sum = 0.0;
#endif
    for (size_t i = 0; i < len(samples); i++) {
      treset(mark);
      nactivate(&net, &samples[i].input, &scratch, &result);

      vec_t target = samples[i].target;
      idx_t loss = vfrom(0);
      for (int k = 0; k < 11; k++) {
        idx_t yk = result.at[k];
        idx_t tk = target.at[k];
        idx_t diff = vsub(tk, yk);
        idx_t lk = vmul(diff, diff);
        loss = vadd(loss, lk);
      }

#ifndef NDEBUG
      epoch_sum += tvalat(loss);
#endif

      for (len_t k = 0; k < SIZE; k++)
        grads[k] = 0;
      tbackpass(loss);

      for (len_t j = 0; j < len(params); j++) {
        idx_t idx = params[j];
        values[idx] += grads[idx] * -0.005;
      }
    }

#ifndef NDEBUG
    if (epoch % 10 == 0) {
      printf("epoch %d avg loss: %f\n", epoch, epoch_sum / len(samples));
    }
#endif
  }

  prompt(&net, &scratch, &result, mark);
  return 0;
}
