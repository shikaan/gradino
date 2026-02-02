<h1 align="center">gradino</h1>

<p align="center">
A tiny, zero-allocation autodiff engine and neural network library in C.
</p>

> [!WARNING]
> Experimental and evolving. APIs may change and things may break.

gradino is a small ISO C99 library that:
- records scalar ops on a global tape and supports reverse-mode autodiff
- builds perceptrons, layers, and simple feed-forward networks (tanh)
- lets you train with squared error and do inference via argmax
- performs **zero heap allocations** — you provide all buffers

It's intended for learning and tinkering, not production.

### Why zero allocation?

gradino never calls `malloc`. You allocate memory once (stack or heap) and pass buffers to the library. This makes it:

- **Embedded-friendly**: no heap, no allocator, predictable memory footprint
- **Cache-friendly**: all data lives in contiguous, caller-controlled arrays
- **Debuggable**: no hidden state, no surprise allocations

## Usage

- Copy-paste `gradino.c` and `gradino.h` in your project and you're done.

- [Train a tiny network to recognize 7-part digits](./examples/05_inference.c)

```sh
# Build example number 05
make example NR=05
```

## How it works

- **Tape**: A linear log of operations. Every math op (`vadd`, `vmul`, `vtanh`, ...) appends a record of what happened and where the result went. This is the foundation for autodiff.
- **Reverse-mode autodiff**: `tapebackprop(idx)` walks the tape backward from `idx`, applying the chain rule to accumulate gradients in `tape->grads`.
- **Abstractions**: Values compose into perceptrons (weights + bias + tanh), perceptrons into layers, layers into networks. Each level is just a thin wrapper over the one below.
- **Memory model**: You allocate one contiguous buffer and pass it to `tapeinit`. All values, gradients, and ops live there. The library never allocates.

## Status and limitations

- Single activation function (`tanh`)
- No built-in loss functions or optimizers — you write the training loop
- Global tape, not thread-safe
- No GPU support
- API is minimal and subject to change

## License

[MIT](./LICENSE).
