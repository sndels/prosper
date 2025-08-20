#!/usr/bin/env sh

sh $(dirname "$0")/generate_compile_commands.sh

run-clang-tidy-19 -p='build_analysis' '.*/prosper/src/.*.cpp'
