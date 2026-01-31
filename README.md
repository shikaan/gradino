<h1 align="center">gradino</h1>

<p align="center">
A tiny autodiff tape and neural network playground in C.
</p>

> [!WARNING]  
> Experimental and evolving. APIs may change and things may break.

gradino is a small ISO C99 library that:
- records scalar ops on a global tape and supports reverse-mode autodiff
- builds perceptrons, layers, and simple feed-forward networks (tanh)
- lets you train with squared error and do inference via argmax

It’s intended for learning and tinkering, not production.

## Usage

- Copy-paste `gradino.c` and `gradino.h` in your project and you're done.

- [Train a tiny network to recognize 7-part digits](./examples/05_inference.c)

```sh
# Build example number 05
make example NR=05
```

## How it works

- Tape: every `vfrom/vadd/vsub/vmul/vtanh` call appends to a global tape (values + ops).
- Backprop: `tbackpass`/`vbackward` traverses the tape in reverse to accumulate grads.
- Perceptron: weights + bias; activation is `tanh`.
- Layer/Net: convenience wrappers to wire perceptrons into layers and sequential nets.

## Status and limitations

- Only `tanh` activation is implemented.
- Losses and training loops are DIY (squared error shown in examples).
- Global tape is simple and not thread-safe.
- API surface is small and likely to change.

## License

[MIT](./LICENSE).
