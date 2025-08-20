#!/usr/bin/env sh

CC=clang-19 CXX=clang++-19 cmake \
    -DPROSPER_USE_PCH=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -G Ninja \
    -S . \
    -B build_analysis
