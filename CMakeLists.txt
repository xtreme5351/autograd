cmake_minimum_required(VERSION 4.1)
project(autograd)

set(CMAKE_CXX_STANDARD 20)

add_library(autograd INTERFACE
        include/tensor_one.h
        include/tape.h
        include/operations.h
        src/operations.cpp)
