#pragma once

#include "tensor.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace autograd {

// ============================================================================
// Basic Operations
// ============================================================================

// Addition
inline TensorPtr add(TensorPtr a, TensorPtr b) {
    if (a->shape() != b->shape()) {
        throw std::runtime_error("Shape mismatch in add");
    }
    
    auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
    out->children = {a, b};
    out->is_leaf = false;
    
    // Forward pass with SIMD
    #pragma omp parallel for simd
    for (size_t i = 0; i < a->size(); ++i) {
        out->data[i] = a->data[i] + b->data[i];
    }
    
    // Backward pass
    if (out->requires_grad) {
        out->backward_fn = [a, b, out]() {
            if (a->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < a->size(); ++i) {
                    a->grad[i] += out->grad[i];
                }
            }
            if (b->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < b->size(); ++i) {
                    b->grad[i] += out->grad[i];
                }
            }
        };
    }
    
    return out;
}

// Subtraction
inline TensorPtr sub(TensorPtr a, TensorPtr b) {
    auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
    out->children = {a, b};
    out->is_leaf = false;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < a->size(); ++i) {
        out->data[i] = a->data[i] - b->data[i];
    }
    
    if (out->requires_grad) {
        out->backward_fn = [a, b, out]() {
            if (a->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < a->size(); ++i) {
                    a->grad[i] += out->grad[i];
                }
            }
            if (b->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < b->size(); ++i) {
                    b->grad[i] -= out->grad[i];
                }
            }
        };
    }
    
    return out;
}

// Element-wise multiplication
inline TensorPtr mul(TensorPtr a, TensorPtr b) {
    auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
    out->children = {a, b};
    out->is_leaf = false;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < a->size(); ++i) {
        out->data[i] = a->data[i] * b->data[i];
    }
    
    if (out->requires_grad) {
        out->backward_fn = [a, b, out]() {
            if (a->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < a->size(); ++i) {
                    a->grad[i] += out->grad[i] * b->data[i];
                }
            }
            if (b->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < b->size(); ++i) {
                    b->grad[i] += out->grad[i] * a->data[i];
                }
            }
        };
    }
    
    return out;
}

// Scalar multiplication
inline TensorPtr mul(TensorPtr a, float scalar) {
    auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad);
    out->children = {a};
    out->is_leaf = false;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < a->size(); ++i) {
        out->data[i] = a->data[i] * scalar;
    }
    
    if (out->requires_grad) {
        out->backward_fn = [a, out, scalar]() {
            if (a->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < a->size(); ++i) {
                    a->grad[i] += out->grad[i] * scalar;
                }
            }
        };
    }
    
    return out;
}

// Element-wise division
inline TensorPtr div(TensorPtr a, TensorPtr b) {
    auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
    out->children = {a, b};
    out->is_leaf = false;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < a->size(); ++i) {
        out->data[i] = a->data[i] / b->data[i];
    }
    
    if (out->requires_grad) {
        out->backward_fn = [a, b, out]() {
            if (a->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < a->size(); ++i) {
                    a->grad[i] += out->grad[i] / b->data[i];
                }
            }
            if (b->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < b->size(); ++i) {
                    b->grad[i] += out->grad[i] * (-a->data[i] / (b->data[i] * b->data[i]));
                }
            }
        };
    }
    
    return out;
}

// Scalar division
inline TensorPtr div(TensorPtr a, float scalar) {
    return mul(a, 1.0f / scalar);
}

// ============================================================================
// Matrix Operations
// ============================================================================

// Matrix multiplication (2D only for now)
inline TensorPtr matmul(TensorPtr a, TensorPtr b) {
    if (a->ndim() != 2 || b->ndim() != 2) {
        throw std::runtime_error("matmul currently only supports 2D tensors");
    }
    
    size_t m = a->shape()[0];
    size_t k = a->shape()[1];
    size_t n = b->shape()[1];
    
    if (b->shape()[0] != k) {
        throw std::runtime_error("Shape mismatch in matmul");
    }
    
    auto out = std::make_shared<Tensor>(std::initializer_list<size_t>{m, n}, 
                                        a->requires_grad || b->requires_grad);
    out->children = {a, b};
    out->is_leaf = false;
    
    // Forward pass - optimized with blocking for cache
    const size_t BLOCK = 64;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i += BLOCK) {
        for (size_t j = 0; j < n; j += BLOCK) {
            for (size_t ii = i; ii < std::min(i + BLOCK, m); ++ii) {
                for (size_t jj = j; jj < std::min(j + BLOCK, n); ++jj) {
                    float sum = 0;
                    #pragma omp simd reduction(+:sum)
                    for (size_t kk = 0; kk < k; ++kk) {
                        sum += a->data[ii * k + kk] * b->data[kk * n + jj];
                    }
                    out->data[ii * n + jj] = sum;
                }
            }
        }
    }
    
    // Backward pass
    if (out->requires_grad) {
        out->backward_fn = [a, b, out, m, k, n]() {
            // Gradient w.r.t a: out_grad @ b^T
            if (a->requires_grad) {
                #pragma omp parallel for collapse(2)
                for (size_t i = 0; i < m; ++i) {
                    for (size_t j = 0; j < k; ++j) {
                        float sum = 0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t p = 0; p < n; ++p) {
                            sum += out->grad[i * n + p] * b->data[j * n + p];
                        }
                        a->grad[i * k + j] += sum;
                    }
                }
            }
            
            // Gradient w.r.t b: a^T @ out_grad
            if (b->requires_grad) {
                #pragma omp parallel for collapse(2)
                for (size_t i = 0; i < k; ++i) {
                    for (size_t j = 0; j < n; ++j) {
                        float sum = 0;
                        #pragma omp simd reduction(+:sum)
                        for (size_t p = 0; p < m; ++p) {
                            sum += a->data[p * k + i] * out->grad[p * n + j];
                        }
                        b->grad[i * n + j] += sum;
                    }
                }
            }
        };
    }
    
    return out;
}

// ============================================================================
// Activation Functions
// ============================================================================

// ReLU
inline TensorPtr relu(TensorPtr x) {
    auto out = std::make_shared<Tensor>(x->shape(), x->requires_grad);
    out->children = {x};
    out->is_leaf = false;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < x->size(); ++i) {
        out->data[i] = std::max(0.0f, x->data[i]);
    }
    
    if (out->requires_grad) {
        out->backward_fn = [x, out]() {
            if (x->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < x->size(); ++i) {
                    x->grad[i] += out->grad[i] * (x->data[i] > 0 ? 1.0f : 0.0f);
                }
            }
        };
    }
    
    return out;
}

// Tanh
inline TensorPtr tanh(TensorPtr x) {
    auto out = std::make_shared<Tensor>(x->shape(), x->requires_grad);
    out->children = {x};
    out->is_leaf = false;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < x->size(); ++i) {
        out->data[i] = std::tanh(x->data[i]);
    }
    
    if (out->requires_grad) {
        out->backward_fn = [x, out]() {
            if (x->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < x->size(); ++i) {
                    float t = out->data[i];
                    x->grad[i] += out->grad[i] * (1.0f - t * t);
                }
            }
        };
    }
    
    return out;
}

// Sigmoid
inline TensorPtr sigmoid(TensorPtr x) {
    auto out = std::make_shared<Tensor>(x->shape(), x->requires_grad);
    out->children = {x};
    out->is_leaf = false;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < x->size(); ++i) {
        out->data[i] = 1.0f / (1.0f + std::exp(-x->data[i]));
    }
    
    if (out->requires_grad) {
        out->backward_fn = [x, out]() {
            if (x->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < x->size(); ++i) {
                    float s = out->data[i];
                    x->grad[i] += out->grad[i] * s * (1.0f - s);
                }
            }
        };
    }
    
    return out;
}

// GELU (Gaussian Error Linear Unit)
inline TensorPtr gelu(TensorPtr x) {
    auto out = std::make_shared<Tensor>(x->shape(), x->requires_grad);
    out->children = {x};
    out->is_leaf = false;
    
    const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
    const float coeff = 0.044715f;
    
    #pragma omp parallel for simd
    for (size_t i = 0; i < x->size(); ++i) {
        float val = x->data[i];
        float cube = val * val * val;
        float inner = sqrt_2_over_pi * (val + coeff * cube);
        out->data[i] = 0.5f * val * (1.0f + std::tanh(inner));
    }
    
    if (out->requires_grad) {
        out->backward_fn = [x, out, sqrt_2_over_pi, coeff]() {
            if (x->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < x->size(); ++i) {
                    float val = x->data[i];
                    float cube = val * val * val;
                    float inner = sqrt_2_over_pi * (val + coeff * cube);
                    float tanh_inner = std::tanh(inner);
                    
                    float sech2 = 1.0f - tanh_inner * tanh_inner;
                    float grad_inner = sqrt_2_over_pi * (1.0f + 3.0f * coeff * val * val);
                    
                    x->grad[i] += out->grad[i] * (0.5f * (1.0f + tanh_inner) + 
                                                  0.5f * val * sech2 * grad_inner);
                }
            }
        };
    }
    
    return out;
}

// ============================================================================
// Reduction Operations
// ============================================================================

// Sum
inline TensorPtr sum(TensorPtr x, int dim = -1, bool keepdim = false) {
    std::vector<size_t> out_shape;
    
    if (dim == -1) {
        // Sum all elements
        out_shape = {1};
    } else {
        // Sum along specific dimension
        out_shape = x->shape();
        if (keepdim) {
            out_shape[dim] = 1;
        } else {
            out_shape.erase(out_shape.begin() + dim);
        }
    }
    
    auto out = std::make_shared<Tensor>(out_shape, x->requires_grad);
    out->children = {x};
    out->is_leaf = false;
    
    // Forward pass
    if (dim == -1) {
        float total = 0;
        #pragma omp parallel for simd reduction(+:total)
        for (size_t i = 0; i < x->size(); ++i) {
            total += x->data[i];
        }
        out->data[0] = total;
    } else {
        // TODO: Implement dimensional sum
        throw std::runtime_error("Dimensional sum not yet implemented");
    }
    
    // Backward pass
    if (out->requires_grad) {
        out->backward_fn = [x, out, dim]() {
            if (x->requires_grad) {
                if (dim == -1) {
                    float grad_val = out->grad[0];
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < x->size(); ++i) {
                        x->grad[i] += grad_val;
                    }
                }
            }
        };
    }
    
    return out;
}

// Mean
inline TensorPtr mean(TensorPtr x, int dim = -1, bool keepdim = false) {
    auto sum_result = sum(x, dim, keepdim);
    float scale = 1.0f / x->size();
    return mul(sum_result, scale);
}

// Exponential function
inline TensorPtr exp(TensorPtr x) {
    auto out = std::make_shared<Tensor>(x->shape(), x->requires_grad);
    out->children = {x};
    out->is_leaf = false;
    
    // Forward pass
    #pragma omp parallel for simd
    for (size_t i = 0; i < x->size(); ++i) {
        out->data[i] = std::exp(x->data[i]);
    }
    
    // Backward pass: d(exp(x))/dx = exp(x)
    if (out->requires_grad) {
        out->backward_fn = [x, out]() {
            if (x->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < x->size(); ++i) {
                    x->grad[i] += out->grad[i] * out->data[i];  // Reuse computed exp(x)
                }
            }
        };
    }
    
    return out;
}

// Natural logarithm
inline TensorPtr log(TensorPtr x) {
    auto out = std::make_shared<Tensor>(x->shape(), x->requires_grad);
    out->children = {x};
    out->is_leaf = false;
    
    // Forward pass
    #pragma omp parallel for simd
    for (size_t i = 0; i < x->size(); ++i) {
        out->data[i] = std::log(x->data[i]);
    }
    
    // Backward pass: d(log(x))/dx = 1/x
    if (out->requires_grad) {
        out->backward_fn = [x, out]() {
            if (x->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < x->size(); ++i) {
                    x->grad[i] += out->grad[i] / x->data[i];
                }
            }
        };
    }
    
    return out;
}

} // namespace autograd