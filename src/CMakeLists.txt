cmake_minimum_required(VERSION 3.23)

project(test LANGUAGES CXX C)

find_package(Kokkos REQUIRED)

add_executable(test main.cpp)


target_link_libraries(test Kokkos::kokkos)

target_compile_options(test PUBLIC -g -O3 -fopenmp)
target_link_options(test PUBLIC -g -O3 -fopenmp)


