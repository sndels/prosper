#!/usr/bin/env sh

sh $(dirname "$0")/generate_compile_commands.sh

run-clang-tidy-15 -p='build-analysis' '.*/prosper/src/.*.cpp'
