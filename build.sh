#!/bin/sh

set -xe

clang -Wall -Wextra -o mlp_nn mlp_nn.c -lm 