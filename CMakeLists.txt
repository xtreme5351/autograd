cmake_minimum_required(VERSION 3.20)

# Keep the existing macOS behaviour, but don't force clang++ on a CUDA host:
# CMake does not propagate CMAKE_CXX_COMPILER into CMAKE_CUDA_HOST_COMPILER, so
# an unconditional pin leaves nvcc picking its own host compiler and you end up
# linking two of them. NOT DEFINED also makes -DCMAKE_CXX_COMPILER=... work.
if (APPLE AND NOT DEFINED CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER "clang++")
endif ()

project(autograd LANGUAGES CXX)  # no OBJCXX needed -- metal-cpp is pure C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- GPU backend selection ----
option(AUTOGRAD_ENABLE_METAL "Build the Metal GPU backend" ON)
option(AUTOGRAD_ENABLE_CUDA "Build the CUDA GPU backend" ON)
set(AUTOGRAD_GPU_BACKEND "none")

if (APPLE AND AUTOGRAD_ENABLE_METAL)
    set(AUTOGRAD_GPU_BACKEND "metal")
elseif (AUTOGRAD_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)  # probes without hard-failing when absent
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # 17, not 20: nvcc's C++20 support is partial. include/backend.h is the
        # only header nvcc sees and is deliberately kept C++17-clean.
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Mandatory since CMake 3.18; `native` would need 3.24+.
            set(CMAKE_CUDA_ARCHITECTURES 70 75 86 89)
        endif ()
        set(AUTOGRAD_GPU_BACKEND "cuda")
    endif ()
endif ()
message(STATUS "autograd GPU backend: ${AUTOGRAD_GPU_BACKEND}")

add_library(autograd STATIC
        src/tensor_one.cpp
        src/tensor_two.cpp
        src/node.cpp
        src/tape/tape_string.cpp
        src/tape/1d/tape_one.cpp
        src/tape/2d/tape_two.cpp
        src/ops/1d/addition.cpp
        src/ops/1d/subtraction.cpp
        src/ops/1d/multiplication.cpp
        src/ops/2d/addition.cpp
        src/ops/2d/subtraction.cpp
        src/ops/2d/multiplication.cpp
        src/ops/2d/matmul.cpp
        src/ops/2d/reduction.cpp
        src/gpu/dispatch.cpp
        src/gpu/cpu/cpu_backend.cpp
)

target_include_directories(autograd PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(autograd PRIVATE
        include/consts.h
        include/tensor_base.h
        include/tensor_one.h
        include/tensor_two.h
        include/tape.h
        include/operations.h
        include/backend.h
)

include(FetchContent)

# ---- GPU backend wiring ----
if (AUTOGRAD_GPU_BACKEND STREQUAL "metal")
    # metal-cpp is header-only and ships no CMakeLists, so wrap it ourselves.
    FetchContent_Declare(metal_cpp
            GIT_REPOSITORY https://github.com/apple/metal-cpp.git
            GIT_TAG main
    )
    FetchContent_MakeAvailable(metal_cpp)

    add_library(metal_cpp INTERFACE)
    target_include_directories(metal_cpp INTERFACE ${metal_cpp_SOURCE_DIR})
    # QuartzCore is needed because CA_PRIVATE_IMPLEMENTATION in the impl TU
    # emits CoreAnimation symbols, even though we never touch a CAMetalLayer.
    target_link_libraries(metal_cpp INTERFACE
            "-framework Metal" "-framework Foundation" "-framework QuartzCore")

    # Kernels are compiled offline so a typo is a build error rather than a
    # runtime newLibrary() failure. Requires the Metal Toolchain:
    #   sudo xcodebuild -downloadComponent MetalToolchain
    set(AUTOGRAD_METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels.metal)
    set(AUTOGRAD_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/autograd.metallib)
    add_custom_command(
            OUTPUT ${AUTOGRAD_METALLIB}
            COMMAND xcrun -sdk macosx metal -fno-fast-math -frecord-sources
            -c ${AUTOGRAD_METAL_SRC} -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.air
            COMMAND xcrun -sdk macosx metallib
            ${CMAKE_CURRENT_BINARY_DIR}/kernels.air -o ${AUTOGRAD_METALLIB}
            DEPENDS ${AUTOGRAD_METAL_SRC}
            COMMENT "Compiling Metal kernels -> autograd.metallib"
            VERBATIM)
    add_custom_target(autograd_metallib DEPENDS ${AUTOGRAD_METALLIB})

    target_sources(autograd PRIVATE
            src/gpu/metal/metal_cpp_impl.cpp
            src/gpu/metal/metal_backend.cpp)
    add_dependencies(autograd autograd_metallib)
    target_link_libraries(autograd PRIVATE metal_cpp)
    # Absolute path, so the tests locate the metallib from any working dir.
    target_compile_definitions(autograd PRIVATE
            AUTOGRAD_METALLIB_PATH="${AUTOGRAD_METALLIB}")
    target_compile_definitions(autograd PUBLIC AUTOGRAD_HAS_METAL=1)

elseif (AUTOGRAD_GPU_BACKEND STREQUAL "cuda")
    target_sources(autograd PRIVATE src/gpu/cuda/cuda_backend.cu)
    target_compile_definitions(autograd PUBLIC AUTOGRAD_HAS_CUDA=1)

else ()
    target_sources(autograd PRIVATE src/gpu/stub/stub_backend.cpp)
endif ()

# ---- gtest ----
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.15.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(autograd_tests
        tests/1d/basic.cpp
        tests/1d/basic_tape.cpp
        tests/1d/advanced_tape.cpp
        tests/1d/stress_tape.cpp
        tests/2d/basic_tape.cpp
        tests/2d/backward_tape.cpp
        tests/gpu/cross_check.cpp
)

# Belt-and-braces: force C++ link language for tests
set_target_properties(autograd_tests PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(autograd_tests
        PRIVATE
        autograd
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(autograd_tests)
