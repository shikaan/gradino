import random
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Constants to match the C implementation
CELLS = 9
MAX_SAMPLES = 1024
EPOCHS = 1000

WIN_LINES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],  # rows
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],  # cols
    [0, 4, 8],
    [2, 4, 6],  # diagonals
]


def winner(board: List[int]) -> int:
    for a, b, c in WIN_LINES:
        s = board[a] + board[b] + board[c]
        if s == 3:
            return 1
        if s == -3:
            return -1
    return 0


def full(board: List[int]) -> bool:
    return all(v != 0 for v in board)


def minimax(board: List[int], player: int) -> Tuple[int, int]:
    w = winner(board)
    if w != 0:
        return (w * 10, -1)
    if full(board):
        return (0, -1)

    best_score = -100 if player > 0 else 100
    best_move = -1
    for i in range(CELLS):
        if board[i] != 0:
            continue
        board[i] = player
        score, _ = minimax(board, -player)
        board[i] = 0
        if player > 0:
            if score > best_score:
                best_score = score
                best_move = i
        else:
            if score < best_score:
                best_score = score
                best_move = i
    return (best_score, best_move)


def hash_board(board: List[int]) -> int:
    h = 0
    for v in board:
        h = h * 3 + (v + 1)
    return h


def generate_samples(max_samples: int = MAX_SAMPLES) -> List[Tuple[List[int], int]]:
    seen = [False] * (3**CELLS)
    samples: List[Tuple[List[int], int]] = []

    def recurse(board: List[int], player: int):
        nonlocal samples
        if winner(board) != 0 or full(board):
            return
        if len(samples) >= max_samples:
            return

        if player > 0:
            h = hash_board(board)
            if not seen[h]:
                seen[h] = True
                _, best = minimax(board, player)
                if best != -1:
                    samples.append((board.copy(), best))

        for i in range(CELLS):
            if board[i] != 0:
                continue
            board[i] = player
            recurse(board, -player)
            board[i] = 0

    recurse([0] * CELLS, 1)
    return samples


class TicTacToeNet(nn.Module):
    # 9 -> 27 -> 9; use Tanh to roughly match [-1, 1] target scale
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(CELLS, 27)
        self.fc2 = nn.Linear(27, CELLS)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


def build_dataset(samples: List[Tuple[List[int], int]]):
    inputs = []
    targets = []
    for board, best_move in samples:
        inp = torch.tensor(board, dtype=torch.float32)  # (9,)
        tgt = torch.full((CELLS,), -1.0, dtype=torch.float32)
        if 0 <= best_move < CELLS:
            tgt[best_move] = 1.0
        inputs.append(inp)
        targets.append(tgt)
    X = torch.stack(inputs)  # (N, 9)
    Y = torch.stack(targets)  # (N, 9)
    return X, Y


def train_per_sample_fair(
    net: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    epochs: int = EPOCHS,
    lr: float = 0.005,
):
    # Force single-threaded CPU to avoid multi-thread BLAS speedups
    torch.set_num_threads(1)

    optimizer = optim.SGD(net.parameters(), lr=lr)
    # Sum reduction to match your C accumulation of squared errors
    criterion = nn.MSELoss(reduction="sum")

    print(f"Generated {X.size(0)} training positions.")
    print(
        "Training the network (per-sample SGD, fair comparison). This might take some seconds..."
    )

    start_time = time.perf_counter()
    for epoch in range(epochs):
        # No shuffling; iterate in order like the C loops
        epoch_loss = 0.0
        for i in range(X.size(0)):
            optimizer.zero_grad()
            out = net(X[i].unsqueeze(0))  # (1, 9)
            loss = criterion(out.squeeze(0), Y[i])
            epoch_loss += float(loss.item())
            loss.backward()
            optimizer.step()
        # Optional: print average loss similar to C debug output
        if epoch % 10 == 0:
            avg = epoch_loss / X.size(0)
            print(f"epoch {epoch} avg loss: {avg:.6f}")
    end_time = time.perf_counter()
    print(f"Training time: {end_time - start_time:.3f} seconds")


def display(board: List[int]):
    print("\nYou are O, the CPU is X.\n")
    for r in range(3):
        if r > 0:
            print("-----------")
        row = []
        for c in range(3):
            i = r * 3 + c
            if board[i] == 1:
                cell = "X"
            elif board[i] == -1:
                cell = "O"
            else:
                cell = str(i + 1)
            row.append(cell)
        print(" " + " | ".join(row))
    print("")


def play(net: nn.Module):
    while True:
        board = [0] * CELLS
        turn = -1

        firstmove = random.randrange(CELLS)
        board[firstmove] = 1  # X starts with random move

        while winner(board) == 0 and not full(board):
            display(board)
            if turn < 0:
                try:
                    move = input("Your move (1-9): ")
                except EOFError:
                    return
                if not move:
                    print("Invalid move.")
                    continue
                cell = ord(move[0]) - ord("1")
                if cell < 0 or cell >= CELLS or board[cell] != 0:
                    print("Invalid move.")
                    continue
                board[cell] = -1
            else:
                with torch.no_grad():
                    inp = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
                    out = net(inp).squeeze(0)
                    best = -1
                    best_val = -float("inf")
                    for i in range(CELLS):
                        if board[i] != 0:
                            continue
                        v = float(out[i].item())
                        if v > best_val:
                            best_val = v
                            best = i
                print(f"X plays {best + 1}\n")
                board[best] = 1
            turn = -turn

        display(board)
        w = winner(board)
        if w > 0:
            print("CPU wins!")
        elif w < 0:
            print("You win!")
            # else draw
        else:
            print("Draw!")

        try:
            ans = input("\nPlay again? (y/N): ")
        except EOFError:
            return
        if not ans or ans[0] not in ("y", "Y"):
            break


def main():
    # Fix seeds for determinism similar to C (if you set srand in C).
    random.seed(0)
    torch.manual_seed(0)

    # Generate the same-style dataset via minimax, up to MAX_SAMPLES
    samples = generate_samples(MAX_SAMPLES)

    # Build tensors
    X, Y = build_dataset(samples)

    # Initialize network 9 -> 27 -> 9
    net = TicTacToeNet()

    # Train with per-sample SGD and sum-of-squares loss
    train_per_sample_fair(net, X, Y, EPOCHS, lr=0.005)

    # # Interactive play
    # play(net)


if __name__ == "__main__":
    main()
