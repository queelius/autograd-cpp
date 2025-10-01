#pragma once

// Main header for the autograd library
// A fast, header-only automatic differentiation library for C++

#include "tensor.hpp"
#include "ops.hpp"
#include "optim.hpp"
#include "jacobian.hpp"
#include "hessian.hpp"

namespace autograd {

// Version information
constexpr const char* VERSION = "1.0.0";
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// Convenience namespace aliases
namespace ops = autograd;
namespace F = autograd;  // Functional interface

} // namespace autograd