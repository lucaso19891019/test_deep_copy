cmake_minimum_required(VERSION 3.23)

project(test LANGUAGES CXX C)

find_package(OpenMP REQUIRED)
add_executable(test main.cpp)


target_compile_options(test PRIVATE -fopenmp)

target_compile_options(test PUBLIC -g -O3)
target_link_options(test PUBLIC -g -O3)
target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)

