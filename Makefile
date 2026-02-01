# Default to debug build
BUILD_TYPE ?= debug

# Chose the compiler you prefer
CC ?= clang

# Common flags for both builds
COMMON_CFLAGS := -std=c99 \
	-Wall \
	-Wextra \
	-Werror \
	-pedantic \
	-fdiagnostics-color=always \
	-fno-common \
	-Winit-self \
	-Wfloat-equal \
	-Wundef \
	-Wshadow \
	-Wpointer-arith \
	-Wcast-align \
	-Wstrict-prototypes \
	-Wstrict-overflow=5 \
	-Wwrite-strings \
	-Waggregate-return \
	-Wcast-qual \
	-Wswitch-default \
	-Wswitch-enum \
	-Wconversion \
	-Wno-ignored-qualifiers \
	-Wno-aggregate-return

DEBUG_CFLAGS := -g -O0 -fsanitize=address,undefined -DDEBUG

# Release-specific flags
RELEASE_CFLAGS := -O2 -DNDEBUG

# Set flags based on build type
ifeq ($(BUILD_TYPE),release)
    CFLAGS := $(COMMON_CFLAGS) $(RELEASE_CFLAGS)
else
    CFLAGS := $(COMMON_CFLAGS) $(DEBUG_CFLAGS)
endif

# Release flags. Set by CI in release builds
VERSION := v0.0.0
SHA := dev

###############################################################################

UNAME_S := $(shell uname)

ifeq ($(UNAME_S),Linux)
    LDFLAGS += -lm
endif

examples/00_backprop: gradino.o
examples/01_perceptron: gradino.o
examples/02_layer: gradino.o
examples/03_network: gradino.o
examples/04_training: gradino.o
examples/05_inference: gradino.o

examples: examples/00_backprop examples/01_perceptron \
	examples/02_layer examples/03_network examples/04_training \
	examples/05_inference

EXAMPLE := $(wildcard examples/${NR}*.c)
example:
	@make $(EXAMPLE:.c=) && ./$(EXAMPLE:.c=)

.PHONY: clean
clean:
	rm -rf *.o **/*.o **/*.dSYM main *.dSYM *.plist
	find ./examples -maxdepth 1 -type f ! -name '*.c' -delete
