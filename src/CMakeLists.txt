cmake_minimum_required(VERSION 3.23)

project(test LANGUAGES CXX C)


add_executable(test main.cpp)

target_link_libraries(test PRIVATE sycl)

target_include_directories(test SYSTEM PRIVATE /opt/aurora/24.180.1/CNDA/oneapi/compiler/eng-20240629/include)
target_compile_options(test PRIVATE -fsycl -fopenmp)
target_link_directories(test PRIVATE /opt/aurora/24.180.1/CNDA/oneapi/compiler/eng-20240629/lib/)

target_compile_options(test PUBLIC -g -O3)
target_link_options(test PUBLIC -g -O3)


