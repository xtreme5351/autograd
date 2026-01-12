cmake_minimum_required(VERSION 4.1)
project(autograd)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(autograd STATIC
        src/operations.cpp
        src/tensor_one.cpp
        src/consts.h
        include/tensor_base.h
        src/node.cpp
        src/tape/tape.cpp
        src/tape/tape_one.cpp
        src/tape/tape_backward.cpp
        src/tape/tape_string.cpp
)

target_include_directories(autograd PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(autograd PRIVATE
        include/tensor_one.h
        include/tape.h
        include/operations.h
)