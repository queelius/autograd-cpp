#pragma once

// Core module system
#include "core/module.hpp"

// Attention mechanisms
#include "attention/scaled_dot_product.hpp"

// Recurrent networks
#include "recurrent/rnn_base.hpp"
#include "recurrent/lstm.hpp"

// Keep backward compatibility by including old nn.hpp content
#include "nn.hpp"

namespace autograd {
namespace nn {

// Convenience namespace for users
// Usage: using namespace autograd::nn;

} // namespace nn
} // namespace autograd