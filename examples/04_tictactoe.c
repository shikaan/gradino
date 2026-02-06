#include "../gradino.h"
#include <string.h>

#define len(Arr) sizeof(Arr) / sizeof(Arr[0])

enum { CELLS = 9, MAX_SAMPLES = 1024, EPOCHS = 1000 };

// Win lines: rows, columns, diagonals
static const int WIN_LINES[8][3] = {
    {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
    {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6},
};

// Returns 1 (X wins), -1 (O wins), or 0 (no winner)
static int winner(const int *b) {
  for (int i = 0; i < 8; i++) {
    int s = b[WIN_LINES[i][0]] + b[WIN_LINES[i][1]] + b[WIN_LINES[i][2]];
    if (s == 3)
      return 1;
    if (s == -3)
      return -1;
  }
  return 0;
}

// Returns 1 when the board is full
static int full(const int *b) {
  for (int i = 0; i < CELLS; i++)
    if (b[i] == 0)
      return 0;
  return 1;
}

// Returns the best score for player and sets *best to the best move.
// player: 1 for X (maximizing), -1 for O (minimizing).
static int minimax(int *b, int player, int *best) {
  int w = winner(b);
  if (w != 0)
    return w * 10;

  if (full(b))
    return 0;

  int best_score = player > 0 ? -100 : 100;
  *best = -1;
  for (int i = 0; i < CELLS; i++) {
    if (b[i] != 0)
      continue;
    b[i] = player;
    int dummy;
    int score = minimax(b, -player, &dummy);
    b[i] = 0;
    if (player > 0 ? score > best_score : score < best_score) {
      best_score = score;
      *best = i;
    }
  }
  return best_score;
}

typedef struct {
  int cells[CELLS];
  int move;
} sample_t;

static sample_t samples[MAX_SAMPLES];
static int nsamples;

// 3^9 possible board encodings for deduplication
#define BOARD_STATES 19683
static bool seen[BOARD_STATES];

// Hash a board configuration into an identifier
static int hash(const int *board) {
  int h = 0;
  for (int i = 0; i < CELLS; i++)
    h = h * 3 + (board[i] + 1);
  return h;
}

// Recursively explore all reachable positions and collect
// (board, best_move) pairs for X-to-move states.
static void generate(int *board, int player) {
  if (winner(board) || full(board) || nsamples >= MAX_SAMPLES)
    return;

  int best;
  if (player > 0) {
    int h = hash(board);
    if (!seen[h]) {
      seen[h] = true;
      minimax(board, player, &best);
      memcpy(samples[nsamples].cells, board, sizeof(int) * CELLS);
      samples[nsamples].move = best;
      nsamples++;
    }
  }

  for (int i = 0; i < CELLS; i++) {
    if (board[i] != 0)
      continue;
    board[i] = player;
    generate(board, -player);
    board[i] = 0;
  }
}

static void display(const int *board) {
  printf("\nYou are O, the CPU is X.\n\n");

  for (int r = 0; r < 3; r++) {
    if (r > 0)
      puts("-----------");
    for (int c = 0; c < 3; c++) {
      int i = r * 3 + c;
      if (c > 0)
        printf(" | ");
      else
        printf(" ");
      if (board[i] == 1)
        printf("X");
      else if (board[i] == -1)
        printf("O");
      else
        printf("%d", i + 1);
    }
    puts("");
  }
  puts("");
}

static void play(net_t *net, idx_t mark) {
  vec_t input, result;
  idx_t idata[CELLS], rdata[CELLS];
  vecinit(&input, CELLS, idata);
  vecinit(&result, CELLS, rdata);

  while (1) {
    int board[CELLS] = {0};
    int turn = -1;

    // Don't start game with a random first move
    int firstmove = rand() % CELLS;
    board[firstmove] = 1;

    while (!winner(board) && !full(board)) {
      display(board);

      if (turn < 0) {
        printf("Your move (1-9): ");
        char buf[16];
        if (!fgets(buf, sizeof(buf), stdin))
          return;
        int cell = buf[0] - '1';
        if (cell < 0 || cell >= CELLS || board[cell] != 0) {
          printf("Invalid move.\n");
          continue;
        }
        board[cell] = -1;
      } else {
        tapereset(mark);

        for (int i = 0; i < CELLS; i++)
          idata[i] = vfrom((value_t)board[i]);
        netfwd(net, &input, &result);

        int maxscorecell = -1;
        value_t maxscore = -1000; // A low number to be overriden
        for (int i = 0; i < CELLS; i++) {
          if (board[i] != 0)
            continue;
          value_t v = tapeval(result.at[i]);
          if (v > maxscore) {
            maxscore = v;
            maxscorecell = i;
          }
        }
        printf("X plays %d\n\n", maxscorecell + 1);
        board[maxscorecell] = 1;
      }
      turn = -turn;
    }

    display(board);
    int w = winner(board);
    if (w > 0)
      puts("CPU wins!");
    else if (w < 0)
      puts("You win!");
    else
      puts("Draw!");

    puts("\nPlay again? (y/N): ");
    char buf[16];
    if (!fgets(buf, sizeof(buf), stdin))
      return;
    if (buf[0] != 'y' && buf[0] != 'Y')
      break;
  }
}

int main(void) {
  // Generate training data from minimax
  int board[CELLS] = {0};
  generate(board, 1);
  printf("Generated %d training positions.\n", nsamples);

  static char tapebuf[1 << 19];
  tapeinit(1 << 13, sizeof(tapebuf), tapebuf);

  net_t net;
  len_t llens[3] = {CELLS, 27, CELLS};
  static char netbuf[1 << 14];
  netinit(&net, len(llens), llens, sizeof(netbuf), netbuf);

  idx_t mark = tapemark();
  vec_t result;
  idx_t rdata[CELLS];
  vecinit(&result, CELLS, rdata);

  puts("Training the network. This might take some seconds...");
  for (int epoch = 0; epoch < EPOCHS; epoch++) {
#ifndef NDEBUG
    value_t epoch_sum = 0.0;
#endif
    for (int s = 0; s < nsamples; s++) {
      tapereset(mark);

      idx_t idata[CELLS];
      for (int i = 0; i < CELLS; i++)
        idata[i] = vfrom((value_t)samples[s].cells[i]);
      vec_t input;
      vecinit(&input, CELLS, idata);

      netfwd(&net, &input, &result);

      idx_t loss = vfrom(0);
      for (int i = 0; i < CELLS; i++) {
        idx_t target = vfrom(i == samples[s].move ? 1.0 : -1.0);
        idx_t diff = vsub(target, result.at[i]);
        loss = vadd(loss, vmul(diff, diff));
      }

#ifndef NDEBUG
      epoch_sum += tapeval(loss);
#endif
      tapezerograd();
      tapebackprop(loss);
      netgdstep(&net, 0.005);
    }
#ifndef NDEBUG
    printf("epoch %d avg loss: %f\n", epoch, epoch_sum / nsamples);
#endif
  }

  play(&net, mark);
  return 0;
}
