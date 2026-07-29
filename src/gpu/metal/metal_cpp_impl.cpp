//
// Created by Pranav C on 29/07/2026.
//
// Generates metal-cpp's out-of-line symbols. These three macros must be
// defined in exactly ONE translation unit in the whole program -- defining
// them a second time gives duplicate-symbol link errors. Every other file
// includes the same headers without them.
//
// This file exists purely to isolate that requirement, so metal_backend.cpp
// can include the headers normally.

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>
