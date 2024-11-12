#!/bin/bash

# Clone Kokkos, develop branch
#git clone --single-branch -b specialize_scratch_space_sycl https://github.com/masterleinad/kokkos.git src/kokkos

git clone -b develop https://github.com/kokkos/kokkos.git src/kokkos

#cp Kokkos_SYCL_ParallelScan_Range.hpp src/kokkos/core/src/SYCL/Kokkos_SYCL_ParallelScan_Range.hpp

# Configure Kokkos
cmake -S src/kokkos -B build/kokkos \
  -DCMAKE_CXX_COMPILER:STRING="mpicxx" \
  -DCMAKE_BUILD_TYPE:STRING="RelWithDebInfo" \
  -DCMAKE_INSTALL_PREFIX=$PWD/install/kokkos \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
  -DKokkos_ARCH_INTEL_SPR:BOOL=ON \
  -DKokkos_ARCH_INTEL_PVC:BOOL=ON \
  -DKokkos_ENABLE_COMPILER_WARNINGS:BOOL=ON \
  -DKokkos_ENABLE_SERIAL:BOOL=ON \
  -DKokkos_ENABLE_OPENMP:BOOL=ON \
  -DKokkos_ENABLE_SYCL:BOOL=ON

# Build and install Kokkos
cd build/kokkos
make -j8 install
cd ../..

