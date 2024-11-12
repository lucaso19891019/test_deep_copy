#!/bin/bash

export Kokkos_DIR=$PWD/install/kokkos/lib64/cmake/Kokkos/

cmake -S src -B build/test \
 -D Kokkos_DIR=$Kokkos_DIR \
 -D CMAKE_CXX_FLAGS="-Wno-deprecated-declarations" \
 -D CMAKE_C_FLAGS="-Wno-stringop-overflow" \
 -D CMAKE_CXX_COMPILER=mpicxx\
 -D CMAKE_C_COMPILER=mpicc \
 -D CMAKE_C_FLAGS="-std=gnu99"

make -C build/test
