#!/usr/bin/env bash

if [ "$#" -lt "1" ]; then
    echo "Pass in the full path to ClanBuildAnalyzer"
    exit 1
fi

initial_path=$(pwd)
script_path=${0%/*}
cba_path=$1

cd $script_path

mkdir -p build
find src -type f -exec touch {} +
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-ftime-trace -G Ninja -B build
cd build
eval "$cba_path --start ."
time ninja
eval "$cba_path --stop . time-trace.capture"
eval "$cba_path --analyze time-trace.capture"
cd ..

cd $script_path

