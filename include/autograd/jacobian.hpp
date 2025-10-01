#pragma once

#include "tensor.hpp"
#include "ops.hpp"
#include <functional>
#include <vector>

namespace autograd {

// ============================================================================
// Jacobian Computation
// ============================================================================

// Compute Jacobian matrix for vector-valued function
// J[i,j] = ∂f_i/∂x_j
inline TensorPtr compute_jacobian(std::function<TensorPtr(TensorPtr)> f, TensorPtr x) {
    size_t input_size = x->size();
    
    // Evaluate function
    auto y = f(x);
    size_t output_size = y->size();
    
    // Jacobian matrix
    auto jacobian = zeros({output_size, input_size}, false);
    
    // Compute each row of Jacobian
    for (size_t i = 0; i < output_size; ++i) {
        // Reset gradients
        x->zero_grad();
        
        // Create unit vector for output i
        auto grad_output = zeros({output_size}, false);
        grad_output->data[i] = 1.0f;
        
        // Backward pass from output i
        y->grad[i] = 1.0f;
        y->backward();
        
        // Copy gradients to Jacobian row
        std::memcpy(jacobian->data + i * input_size,
                   x->grad,
                   input_size * sizeof(float));
    }
    
    return jacobian;
}

// Efficient Jacobian computation using forward-mode AD
// Better for functions with few inputs, many outputs
inline TensorPtr compute_jacobian_forward(std::function<TensorPtr(TensorPtr)> f, TensorPtr x) {
    size_t input_size = x->size();
    
    // Evaluate function once to get output size
    auto y = f(x);
    size_t output_size = y->size();
    
    auto jacobian = zeros({output_size, input_size}, false);
    
    // Compute each column using forward-mode AD
    for (size_t j = 0; j < input_size; ++j) {
        // Create perturbation vector
        auto dx = zeros({input_size}, false);
        dx->data[j] = 1.0f;
        
        // Forward-mode differentiation
        auto x_dual = x->clone();
        x_dual->requires_grad = true;
        
        // Perturb input
        x_dual->data[j] += 1e-7f;
        auto y_perturbed = f(x_dual);
        
        // Finite difference approximation
        for (size_t i = 0; i < output_size; ++i) {
            jacobian->data[i * input_size + j] = 
                (y_perturbed->data[i] - y->data[i]) / 1e-7f;
        }
    }
    
    return jacobian;
}

// Vector-Jacobian product (more efficient than computing full Jacobian)
// Computes v^T * J where J is the Jacobian of f at x
inline TensorPtr vjp(std::function<TensorPtr(TensorPtr)> f, TensorPtr x, TensorPtr v) {
    // Forward pass
    auto y = f(x);
    
    if (v->size() != y->size()) {
        throw std::runtime_error("Vector size must match output size for VJP");
    }
    
    // Set output gradient to v
    x->zero_grad();
    std::memcpy(y->grad, v->data, v->size() * sizeof(float));
    
    // Backward pass computes v^T * J
    y->backward();
    
    // Return gradient
    auto result = zeros({x->size()}, false);
    std::memcpy(result->data, x->grad, x->size() * sizeof(float));
    
    return result;
}

// Jacobian-vector product
// Computes J * v where J is the Jacobian of f at x
inline TensorPtr jvp(std::function<TensorPtr(TensorPtr)> f, TensorPtr x, TensorPtr v) {
    if (v->size() != x->size()) {
        throw std::runtime_error("Vector size must match input size for JVP");
    }
    
    // Use finite differences for now (could implement forward-mode AD)
    const float eps = 1e-7f;
    
    // f(x + eps*v)
    auto x_perturbed = x->clone();
    for (size_t i = 0; i < x->size(); ++i) {
        x_perturbed->data[i] += eps * v->data[i];
    }
    
    auto y = f(x);
    auto y_perturbed = f(x_perturbed);
    
    // (f(x + eps*v) - f(x)) / eps ≈ J*v
    auto result = zeros({y->size()}, false);
    for (size_t i = 0; i < y->size(); ++i) {
        result->data[i] = 
            (y_perturbed->data[i] - y->data[i]) / eps;
    }
    
    return result;
}

// ============================================================================
// Optimized Batched Operations
// ============================================================================

// Compute Jacobian for batched inputs
inline TensorPtr compute_jacobian_batched(
    std::function<TensorPtr(TensorPtr)> f,
    TensorPtr x_batch,
    size_t batch_dim = 0) {
    
    if (x_batch->ndim() < 2) {
        throw std::runtime_error("Batched Jacobian requires at least 2D input");
    }
    
    size_t batch_size = x_batch->shape()[batch_dim];
    size_t input_size = x_batch->size() / batch_size;
    
    // Process each batch element
    std::vector<TensorPtr> jacobians;
    
    #pragma omp parallel for
    for (size_t b = 0; b < batch_size; ++b) {
        // Extract batch element
        auto x = zeros({input_size}, true);
        std::memcpy(x->data,
                   x_batch->data + b * input_size,
                   input_size * sizeof(float));
        
        // Compute Jacobian for this element
        auto J = compute_jacobian(f, x);
        
        #pragma omp critical
        jacobians.push_back(J);
    }
    
    // Stack results
    size_t output_size = jacobians[0]->shape()[0];
    auto result = zeros({batch_size, output_size, input_size}, false);
    
    for (size_t b = 0; b < batch_size; ++b) {
        std::memcpy(result->data + b * output_size * input_size,
                   jacobians[b]->data,
                   output_size * input_size * sizeof(float));
    }
    
    return result;
}

// ============================================================================
// Specialized Derivatives for Common Functions
// ============================================================================

// Gradient of scalar function (returns vector)
inline TensorPtr gradient(TensorPtr scalar_output, TensorPtr x) {
    if (scalar_output->size() != 1) {
        throw std::runtime_error("gradient() requires scalar output");
    }
    
    x->zero_grad();
    scalar_output->backward();
    
    auto grad = zeros({x->size()}, false);
    std::memcpy(grad->data, x->grad, x->size() * sizeof(float));
    
    return grad;
}

// Divergence of vector field
inline TensorPtr divergence(std::function<TensorPtr(TensorPtr)> vector_field, TensorPtr x) {
    auto J = compute_jacobian(vector_field, x);
    
    // Divergence is trace of Jacobian
    size_t n = x->size();
    float div = 0;
    for (size_t i = 0; i < n; ++i) {
        div += J->data[i * n + i];
    }
    
    auto result = zeros({1}, false);
    result->data[0] = div;
    
    return result;
}


} // namespace autograd