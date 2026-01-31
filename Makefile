# Default to debug build
BUILD_TYPE ?= debug

# Chose the compiler you prefer
CC := clang

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
	-Wassign-enum \
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

examples/backprop: gradino.o
examples/perceptron: gradino.o
examples/layer: gradino.o
examples/network: gradino.o
examples/training: gradino.o

.PHONY: clean
clean:
	rm -rf *.o **/*.o **/*.dSYM main *.dSYM *.plist
