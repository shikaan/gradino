<h1 align="center">gradino</h1>

<p align="center">
A tiny autodiff engine and neural network library in C.
</p>

gradino is a small ISO C99 library that:
- records scalar ops on a global tape and supports reverse-mode autodiff
- builds simple feed-forward networks (tanh)
- lets you train with squared error and do inference via argmax
- can perform **zero heap allocations** — you provide all buffers

### Zero allocation mode

gradino can run without calling `malloc`. You allocate memory once (stack or heap) and pass buffers to the library via `tapeinit`/`netinit`. Convenience wrappers (`tapecreate`/`netcreate`) are also available for heap allocation. 

When using heap allocation, you can bring your own allocator by defining `GRADINO_ALLOC` and `GRADINO_FREE` before including the header:

```c
#define GRADINO_ALLOC my_alloc
#define GRADINO_FREE  my_free
#include "gradino.h"
```

The zero-allocation mode makes it:

- **Embedded-friendly**: no heap, no allocator, predictable memory footprint
- **Cache-friendly**: all data lives in contiguous, caller-controlled arrays
- **Debuggable**: no hidden state, no surprise allocations

## Usage

- Copy-paste `gradino.c` and `gradino.h` in your project and you're done.
- See [`gradino.h`](./gradino.h) for the full API documentation and examples.
- [Train a tiny network to recognize 7-part digits](./examples/03_inference.c)

```sh
# Build example number 03
make example NR=03
```

## How it works

- **Tape**: A linear log of operations. Every math op (`vadd`, `vmul`, `vtanh`, ...) appends a record of what happened and where the result went. This is the foundation for autodiff.
- **Reverse-mode autodiff**: `tapebackprop(idx)` walks the tape backward from `idx`, applying the chain rule to accumulate gradients in `tape->grads`.
- **Abstractions**: Values compose into perceptrons, perceptrons into layers, layers into networks. These internals are hidden behind the network API (`netinit`/`netcreate`, `netfwd`, `netgdstep`).
- **Memory model**: All values, gradients, and ops live in contiguous buffers. You can provide your own (`tapeinit`/`netinit`) or let the library allocate (`tapecreate`/`netcreate`).

## Status and limitations

- Single activation function (`tanh`)
- No built-in loss functions or optimizers — you write the training loop
- Global tape, not thread-safe

## License

[MIT](./LICENSE).
