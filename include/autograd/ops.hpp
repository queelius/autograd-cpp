#pragma once

#include "tensor.hpp"
#include <cmath>
#include <algorithm>
#include <omp.h>

namespace autograd {

// ============================================================================
// Basic Operations
// ============================================================================

// Addition with broadcasting support for scalars
inline TensorPtr add(TensorPtr a, TensorPtr b) {
    // Check if shapes match exactly
    if (a->shape() == b->shape()) {
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
    // Broadcasting: b is scalar, a is larger
    else if (b->size() == 1) {
        auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float b_val = b->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < a->size(); ++i) {
            out->data[i] = a->data[i] + b_val;
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
                    // Sum gradients for broadcast scalar
                    float grad_sum = 0;
                    #pragma omp parallel for simd reduction(+:grad_sum)
                    for (size_t i = 0; i < out->size(); ++i) {
                        grad_sum += out->grad[i];
                    }
                    b->grad[0] += grad_sum;
                }
            };
        }

        return out;
    }
    // Broadcasting: a is scalar, b is larger
    else if (a->size() == 1) {
        auto out = std::make_shared<Tensor>(b->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float a_val = a->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < b->size(); ++i) {
            out->data[i] = a_val + b->data[i];
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                if (a->requires_grad) {
                    // Sum gradients for broadcast scalar
                    float grad_sum = 0;
                    #pragma omp parallel for simd reduction(+:grad_sum)
                    for (size_t i = 0; i < out->size(); ++i) {
                        grad_sum += out->grad[i];
                    }
                    a->grad[0] += grad_sum;
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
    else {
        throw std::runtime_error("Shape mismatch in add: incompatible shapes for broadcasting");
    }
}

// Subtraction with broadcasting support for scalars
inline TensorPtr sub(TensorPtr a, TensorPtr b) {
    // Check if shapes match exactly
    if (a->shape() == b->shape()) {
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
    // Broadcasting: b is scalar, a is larger
    else if (b->size() == 1) {
        auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float b_val = b->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < a->size(); ++i) {
            out->data[i] = a->data[i] - b_val;
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
                    // Sum gradients for broadcast scalar (with negative sign)
                    float grad_sum = 0;
                    #pragma omp parallel for simd reduction(+:grad_sum)
                    for (size_t i = 0; i < out->size(); ++i) {
                        grad_sum += out->grad[i];
                    }
                    b->grad[0] -= grad_sum;
                }
            };
        }

        return out;
    }
    // Broadcasting: a is scalar, b is larger
    else if (a->size() == 1) {
        auto out = std::make_shared<Tensor>(b->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float a_val = a->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < b->size(); ++i) {
            out->data[i] = a_val - b->data[i];
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                if (a->requires_grad) {
                    // Sum gradients for broadcast scalar
                    float grad_sum = 0;
                    #pragma omp parallel for simd reduction(+:grad_sum)
                    for (size_t i = 0; i < out->size(); ++i) {
                        grad_sum += out->grad[i];
                    }
                    a->grad[0] += grad_sum;
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
    else {
        throw std::runtime_error("Shape mismatch in sub: incompatible shapes for broadcasting");
    }
}

// Element-wise multiplication with broadcasting support for scalars
inline TensorPtr mul(TensorPtr a, TensorPtr b) {
    // Check if shapes match exactly
    if (a->shape() == b->shape()) {
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
    // Broadcasting: b is scalar, a is larger
    else if (b->size() == 1) {
        auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float b_val = b->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < a->size(); ++i) {
            out->data[i] = a->data[i] * b_val;
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                float b_val = b->data[0];
                if (a->requires_grad) {
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < a->size(); ++i) {
                        a->grad[i] += out->grad[i] * b_val;
                    }
                }
                if (b->requires_grad) {
                    // Sum of (grad * a) for broadcast scalar
                    float grad_sum = 0;
                    #pragma omp parallel for simd reduction(+:grad_sum)
                    for (size_t i = 0; i < out->size(); ++i) {
                        grad_sum += out->grad[i] * a->data[i];
                    }
                    b->grad[0] += grad_sum;
                }
            };
        }

        return out;
    }
    // Broadcasting: a is scalar, b is larger
    else if (a->size() == 1) {
        auto out = std::make_shared<Tensor>(b->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float a_val = a->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < b->size(); ++i) {
            out->data[i] = a_val * b->data[i];
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                float a_val = a->data[0];
                if (a->requires_grad) {
                    // Sum of (grad * b) for broadcast scalar
                    float grad_sum = 0;
                    #pragma omp parallel for simd reduction(+:grad_sum)
                    for (size_t i = 0; i < out->size(); ++i) {
                        grad_sum += out->grad[i] * b->data[i];
                    }
                    a->grad[0] += grad_sum;
                }
                if (b->requires_grad) {
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < b->size(); ++i) {
                        b->grad[i] += out->grad[i] * a_val;
                    }
                }
            };
        }

        return out;
    }
    else {
        throw std::runtime_error("Shape mismatch in mul: incompatible shapes for broadcasting");
    }
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

// Element-wise division with broadcasting support for scalars
inline TensorPtr div(TensorPtr a, TensorPtr b) {
    // Check if shapes match exactly
    if (a->shape() == b->shape()) {
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
    // Broadcasting: b is scalar, a is larger (a / scalar)
    else if (b->size() == 1) {
        auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float b_val = b->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < a->size(); ++i) {
            out->data[i] = a->data[i] / b_val;
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                float b_val = b->data[0];
                if (a->requires_grad) {
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < a->size(); ++i) {
                        a->grad[i] += out->grad[i] / b_val;
                    }
                }
                if (b->requires_grad) {
                    // d(a/b)/db = -a/b^2, summed over all elements
                    float grad_sum = 0;
                    float b_sq = b_val * b_val;
                    #pragma omp parallel for simd reduction(+:grad_sum)
                    for (size_t i = 0; i < out->size(); ++i) {
                        grad_sum += out->grad[i] * (-a->data[i] / b_sq);
                    }
                    b->grad[0] += grad_sum;
                }
            };
        }

        return out;
    }
    // Broadcasting: a is scalar, b is larger (scalar / b)
    else if (a->size() == 1) {
        auto out = std::make_shared<Tensor>(b->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float a_val = a->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < b->size(); ++i) {
            out->data[i] = a_val / b->data[i];
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                float a_val = a->data[0];
                if (a->requires_grad) {
                    // d(a/b)/da = 1/b, summed over all elements
                    float grad_sum = 0;
                    #pragma omp parallel for simd reduction(+:grad_sum)
                    for (size_t i = 0; i < out->size(); ++i) {
                        grad_sum += out->grad[i] / b->data[i];
                    }
                    a->grad[0] += grad_sum;
                }
                if (b->requires_grad) {
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < b->size(); ++i) {
                        b->grad[i] += out->grad[i] * (-a_val / (b->data[i] * b->data[i]));
                    }
                }
            };
        }

        return out;
    }
    else {
        throw std::runtime_error("Shape mismatch in div: incompatible shapes for broadcasting");
    }
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

// Cholesky decomposition
// Computes L where A = LL^T for a symmetric positive-definite matrix A
inline TensorPtr cholesky(TensorPtr A, bool lower = true) {
    if (A->ndim() != 2) {
        throw std::runtime_error("cholesky expects a 2D tensor");
    }

    size_t n = A->shape()[0];
    if (A->shape()[1] != n) {
        throw std::runtime_error("cholesky expects a square matrix");
    }

    auto L = std::make_shared<Tensor>(std::initializer_list<size_t>{n, n}, A->requires_grad);
    L->children = {A};
    L->is_leaf = false;

    // Forward pass: Compute Cholesky decomposition
    if (lower) {
        // Lower triangular: A = LL^T
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                float sum = A->data[i * n + j];

                // Subtract L[i,k] * L[j,k] for k < j
                for (size_t k = 0; k < j; ++k) {
                    sum -= L->data[i * n + k] * L->data[j * n + k];
                }

                if (i == j) {
                    // Diagonal element
                    if (sum <= 0.0f) {
                        throw std::runtime_error("Matrix is not positive definite");
                    }
                    L->data[i * n + j] = std::sqrt(sum);
                } else {
                    // Off-diagonal element
                    L->data[i * n + j] = sum / L->data[j * n + j];
                }
            }
            // Upper triangle is zero
            for (size_t j = i + 1; j < n; ++j) {
                L->data[i * n + j] = 0.0f;
            }
        }
    } else {
        // Upper triangular: A = U^TU
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i <= j; ++i) {
                float sum = A->data[i * n + j];

                for (size_t k = 0; k < i; ++k) {
                    sum -= L->data[k * n + i] * L->data[k * n + j];
                }

                if (i == j) {
                    if (sum <= 0.0f) {
                        throw std::runtime_error("Matrix is not positive definite");
                    }
                    L->data[i * n + j] = std::sqrt(sum);
                } else {
                    L->data[i * n + j] = sum / L->data[i * n + i];
                }
            }
            // Lower triangle is zero
            for (size_t i = j + 1; i < n; ++i) {
                L->data[i * n + j] = 0.0f;
            }
        }
    }

    // Backward pass: Compute gradient of A from gradient of L
    // This uses the implicit function theorem for Cholesky decomposition
    if (L->requires_grad) {
        L->backward_fn = [A, L, n, lower]() {
            if (!A->requires_grad) return;

            // Allocate temporary gradient for L
            std::vector<float> phi(n * n, 0.0f);

            if (lower) {
                // Backpropagation for lower triangular Cholesky
                // We solve: L * phi + phi^T * L = dL (symmetric part)
                // Working backwards from bottom-right to top-left

                for (int i = n - 1; i >= 0; --i) {
                    for (int j = i; j >= 0; --j) {
                        float val = L->grad[i * n + j];

                        // Add contributions from elements we've already processed
                        for (size_t k = i + 1; k < n; ++k) {
                            val -= phi[k * n + j] * L->data[k * n + i];
                        }

                        if (i == j) {
                            // Diagonal: dL_ii/dA contributes through sqrt
                            phi[i * n + j] = val / (2.0f * L->data[i * n + j]);
                        } else {
                            // Off-diagonal
                            float temp = val / L->data[j * n + j];
                            phi[i * n + j] = temp;

                            // Subtract contribution to L[j,j]
                            for (size_t k = j + 1; k < n; ++k) {
                                if (k == i) continue;
                                phi[j * n + j] -= temp * L->data[k * n + j];
                            }
                        }
                    }
                }

                // Accumulate gradient to A (symmetrize)
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        A->grad[i * n + j] += phi[i * n + j];
                        if (i != j) {
                            A->grad[j * n + i] += phi[i * n + j];
                        }
                    }
                }
            } else {
                // Backpropagation for upper triangular
                // Similar logic but for upper triangular structure
                for (int j = n - 1; j >= 0; --j) {
                    for (int i = j; i >= 0; --i) {
                        float val = L->grad[i * n + j];

                        for (size_t k = j + 1; k < n; ++k) {
                            val -= phi[i * n + k] * L->data[j * n + k];
                        }

                        if (i == j) {
                            phi[i * n + j] = val / (2.0f * L->data[i * n + j]);
                        } else {
                            float temp = val / L->data[i * n + i];
                            phi[i * n + j] = temp;

                            for (size_t k = i + 1; k < n; ++k) {
                                if (k == j) continue;
                                phi[i * n + i] -= temp * L->data[i * n + k];
                            }
                        }
                    }
                }

                // Accumulate gradient to A (symmetrize)
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = i; j < n; ++j) {
                        A->grad[i * n + j] += phi[i * n + j];
                        if (i != j) {
                            A->grad[j * n + i] += phi[i * n + j];
                        }
                    }
                }
            }
        };
    }

    return L;
}

// Triangular solve: Solves Ax = b where A is triangular
// Returns x such that Ax = b (if transpose=false) or A^Tx = b (if transpose=true)
inline TensorPtr solve_triangular(TensorPtr A, TensorPtr b, bool lower = true, bool transpose = false) {
    if (A->ndim() != 2) {
        throw std::runtime_error("solve_triangular expects A to be a 2D tensor");
    }

    size_t n = A->shape()[0];
    if (A->shape()[1] != n) {
        throw std::runtime_error("solve_triangular expects A to be square");
    }

    size_t m = 1;
    bool is_vector = (b->ndim() == 1);

    if (is_vector) {
        if (b->shape()[0] != n) {
            throw std::runtime_error("Shape mismatch in solve_triangular");
        }
    } else if (b->ndim() == 2) {
        if (b->shape()[0] != n) {
            throw std::runtime_error("Shape mismatch in solve_triangular");
        }
        m = b->shape()[1];
    } else {
        throw std::runtime_error("solve_triangular expects b to be 1D or 2D");
    }

    auto x = std::make_shared<Tensor>(b->shape(), A->requires_grad || b->requires_grad);
    x->children = {A, b};
    x->is_leaf = false;

    // Forward pass: Solve triangular system
    for (size_t col = 0; col < m; ++col) {
        if (lower && !transpose) {
            // Forward substitution: Lx = b
            for (size_t i = 0; i < n; ++i) {
                float sum = (is_vector ? b->data[i] : b->data[i * m + col]);
                for (size_t j = 0; j < i; ++j) {
                    sum -= A->data[i * n + j] * (is_vector ? x->data[j] : x->data[j * m + col]);
                }
                float result = sum / A->data[i * n + i];
                if (is_vector) {
                    x->data[i] = result;
                } else {
                    x->data[i * m + col] = result;
                }
            }
        } else if (!lower && !transpose) {
            // Back substitution: Ux = b
            for (int i = n - 1; i >= 0; --i) {
                float sum = (is_vector ? b->data[i] : b->data[i * m + col]);
                for (size_t j = i + 1; j < n; ++j) {
                    sum -= A->data[i * n + j] * (is_vector ? x->data[j] : x->data[j * m + col]);
                }
                float result = sum / A->data[i * n + i];
                if (is_vector) {
                    x->data[i] = result;
                } else {
                    x->data[i * m + col] = result;
                }
            }
        } else if (lower && transpose) {
            // Back substitution: L^Tx = b
            for (int i = n - 1; i >= 0; --i) {
                float sum = (is_vector ? b->data[i] : b->data[i * m + col]);
                for (size_t j = i + 1; j < n; ++j) {
                    sum -= A->data[j * n + i] * (is_vector ? x->data[j] : x->data[j * m + col]);
                }
                float result = sum / A->data[i * n + i];
                if (is_vector) {
                    x->data[i] = result;
                } else {
                    x->data[i * m + col] = result;
                }
            }
        } else {
            // Forward substitution: U^Tx = b
            for (size_t i = 0; i < n; ++i) {
                float sum = (is_vector ? b->data[i] : b->data[i * m + col]);
                for (size_t j = 0; j < i; ++j) {
                    sum -= A->data[j * n + i] * (is_vector ? x->data[j] : x->data[j * m + col]);
                }
                float result = sum / A->data[i * n + i];
                if (is_vector) {
                    x->data[i] = result;
                } else {
                    x->data[i * m + col] = result;
                }
            }
        }
    }

    // Backward pass: Compute gradients using implicit differentiation
    // For Ax = b, we have: dA·x + A·dx = db
    // Therefore: A·dx = db - dA·x
    // So: dx = A^{-1}(db - dA·x) = A^{-1}·db - x (by solving again)
    if (x->requires_grad) {
        x->backward_fn = [A, b, x, n, m, lower, transpose, is_vector]() {
            // Gradient w.r.t b: solve A^T λ = dx (where λ is the gradient)
            if (b->requires_grad) {
                // For the equation Ax = b, the gradient of b is: db = A^{-T} dx
                // We solve this by transposing the original solve
                for (size_t col = 0; col < m; ++col) {
                    std::vector<float> lambda(n);

                    // Solve A^T λ = x.grad
                    if (lower && !transpose) {
                        // Solve L^T λ = dx
                        for (int i = n - 1; i >= 0; --i) {
                            float sum = (is_vector ? x->grad[i] : x->grad[i * m + col]);
                            for (size_t j = i + 1; j < n; ++j) {
                                sum -= A->data[j * n + i] * lambda[j];
                            }
                            lambda[i] = sum / A->data[i * n + i];
                        }
                    } else if (!lower && !transpose) {
                        // Solve U^T λ = dx
                        for (size_t i = 0; i < n; ++i) {
                            float sum = (is_vector ? x->grad[i] : x->grad[i * m + col]);
                            for (size_t j = 0; j < i; ++j) {
                                sum -= A->data[j * n + i] * lambda[j];
                            }
                            lambda[i] = sum / A->data[i * n + i];
                        }
                    } else if (lower && transpose) {
                        // Solve L λ = dx
                        for (size_t i = 0; i < n; ++i) {
                            float sum = (is_vector ? x->grad[i] : x->grad[i * m + col]);
                            for (size_t j = 0; j < i; ++j) {
                                sum -= A->data[i * n + j] * lambda[j];
                            }
                            lambda[i] = sum / A->data[i * n + i];
                        }
                    } else {
                        // Solve U λ = dx
                        for (int i = n - 1; i >= 0; --i) {
                            float sum = (is_vector ? x->grad[i] : x->grad[i * m + col]);
                            for (size_t j = i + 1; j < n; ++j) {
                                sum -= A->data[i * n + j] * lambda[j];
                            }
                            lambda[i] = sum / A->data[i * n + i];
                        }
                    }

                    // Accumulate to b's gradient
                    for (size_t i = 0; i < n; ++i) {
                        if (is_vector) {
                            b->grad[i] += lambda[i];
                        } else {
                            b->grad[i * m + col] += lambda[i];
                        }
                    }
                }
            }

            // Gradient w.r.t A: dA = -λ x^T (outer product)
            // where λ = A^{-T} dx (computed above)
            if (A->requires_grad) {
                for (size_t col = 0; col < m; ++col) {
                    std::vector<float> lambda(n);

                    // Solve A^T λ = dx (same as for b's gradient)
                    if (lower && !transpose) {
                        for (int i = n - 1; i >= 0; --i) {
                            float sum = (is_vector ? x->grad[i] : x->grad[i * m + col]);
                            for (size_t j = i + 1; j < n; ++j) {
                                sum -= A->data[j * n + i] * lambda[j];
                            }
                            lambda[i] = sum / A->data[i * n + i];
                        }
                    } else if (!lower && !transpose) {
                        for (size_t i = 0; i < n; ++i) {
                            float sum = (is_vector ? x->grad[i] : x->grad[i * m + col]);
                            for (size_t j = 0; j < i; ++j) {
                                sum -= A->data[j * n + i] * lambda[j];
                            }
                            lambda[i] = sum / A->data[i * n + i];
                        }
                    } else if (lower && transpose) {
                        for (size_t i = 0; i < n; ++i) {
                            float sum = (is_vector ? x->grad[i] : x->grad[i * m + col]);
                            for (size_t j = 0; j < i; ++j) {
                                sum -= A->data[i * n + j] * lambda[j];
                            }
                            lambda[i] = sum / A->data[i * n + i];
                        }
                    } else {
                        for (int i = n - 1; i >= 0; --i) {
                            float sum = (is_vector ? x->grad[i] : x->grad[i * m + col]);
                            for (size_t j = i + 1; j < n; ++j) {
                                sum -= A->data[i * n + j] * lambda[j];
                            }
                            lambda[i] = sum / A->data[i * n + i];
                        }
                    }

                    // Compute outer product: dA -= λ ⊗ x
                    for (size_t i = 0; i < n; ++i) {
                        for (size_t j = 0; j < n; ++j) {
                            float x_val = (is_vector ? x->data[j] : x->data[j * m + col]);
                            A->grad[i * n + j] -= lambda[i] * x_val;
                        }
                    }
                }
            }
        };
    }

    return x;
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

// Absolute value
inline TensorPtr abs(TensorPtr x) {
    auto out = std::make_shared<Tensor>(x->shape(), x->requires_grad);
    out->children = {x};
    out->is_leaf = false;

    // Forward pass
    #pragma omp parallel for simd
    for (size_t i = 0; i < x->size(); ++i) {
        out->data[i] = std::abs(x->data[i]);
    }

    // Backward pass: d|x|/dx = sign(x)
    // At x=0, we use subgradient 0
    if (out->requires_grad) {
        out->backward_fn = [x, out]() {
            if (x->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < x->size(); ++i) {
                    float sign = (x->data[i] > 0) ? 1.0f : ((x->data[i] < 0) ? -1.0f : 0.0f);
                    x->grad[i] += out->grad[i] * sign;
                }
            }
        };
    }

    return out;
}

// Element-wise minimum with broadcasting support for scalars
inline TensorPtr minimum(TensorPtr a, TensorPtr b) {
    // Check if shapes match exactly
    if (a->shape() == b->shape()) {
        auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        #pragma omp parallel for simd
        for (size_t i = 0; i < a->size(); ++i) {
            out->data[i] = std::min(a->data[i], b->data[i]);
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                if (a->requires_grad || b->requires_grad) {
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < a->size(); ++i) {
                        if (a->data[i] < b->data[i]) {
                            if (a->requires_grad) a->grad[i] += out->grad[i];
                        } else if (b->data[i] < a->data[i]) {
                            if (b->requires_grad) b->grad[i] += out->grad[i];
                        } else {
                            if (a->requires_grad) a->grad[i] += 0.5f * out->grad[i];
                            if (b->requires_grad) b->grad[i] += 0.5f * out->grad[i];
                        }
                    }
                }
            };
        }

        return out;
    }
    // Broadcasting: b is scalar, a is larger
    else if (b->size() == 1) {
        auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float b_val = b->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < a->size(); ++i) {
            out->data[i] = std::min(a->data[i], b_val);
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                float b_val = b->data[0];
                float b_grad_sum = 0;
                #pragma omp parallel for simd reduction(+:b_grad_sum)
                for (size_t i = 0; i < a->size(); ++i) {
                    if (a->data[i] < b_val) {
                        if (a->requires_grad) a->grad[i] += out->grad[i];
                    } else if (b_val < a->data[i]) {
                        b_grad_sum += out->grad[i];
                    } else {
                        if (a->requires_grad) a->grad[i] += 0.5f * out->grad[i];
                        b_grad_sum += 0.5f * out->grad[i];
                    }
                }
                if (b->requires_grad) b->grad[0] += b_grad_sum;
            };
        }

        return out;
    }
    // Broadcasting: a is scalar, b is larger
    else if (a->size() == 1) {
        auto out = std::make_shared<Tensor>(b->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float a_val = a->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < b->size(); ++i) {
            out->data[i] = std::min(a_val, b->data[i]);
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                float a_val = a->data[0];
                float a_grad_sum = 0;
                #pragma omp parallel for simd reduction(+:a_grad_sum)
                for (size_t i = 0; i < b->size(); ++i) {
                    if (a_val < b->data[i]) {
                        a_grad_sum += out->grad[i];
                    } else if (b->data[i] < a_val) {
                        if (b->requires_grad) b->grad[i] += out->grad[i];
                    } else {
                        a_grad_sum += 0.5f * out->grad[i];
                        if (b->requires_grad) b->grad[i] += 0.5f * out->grad[i];
                    }
                }
                if (a->requires_grad) a->grad[0] += a_grad_sum;
            };
        }

        return out;
    }
    else {
        throw std::runtime_error("Shape mismatch in minimum: incompatible shapes for broadcasting");
    }
}

// Element-wise maximum with broadcasting support for scalars
inline TensorPtr maximum(TensorPtr a, TensorPtr b) {
    // Check if shapes match exactly
    if (a->shape() == b->shape()) {
        auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        #pragma omp parallel for simd
        for (size_t i = 0; i < a->size(); ++i) {
            out->data[i] = std::max(a->data[i], b->data[i]);
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                if (a->requires_grad || b->requires_grad) {
                    #pragma omp parallel for simd
                    for (size_t i = 0; i < a->size(); ++i) {
                        if (a->data[i] > b->data[i]) {
                            if (a->requires_grad) a->grad[i] += out->grad[i];
                        } else if (b->data[i] > a->data[i]) {
                            if (b->requires_grad) b->grad[i] += out->grad[i];
                        } else {
                            if (a->requires_grad) a->grad[i] += 0.5f * out->grad[i];
                            if (b->requires_grad) b->grad[i] += 0.5f * out->grad[i];
                        }
                    }
                }
            };
        }

        return out;
    }
    // Broadcasting: b is scalar, a is larger
    else if (b->size() == 1) {
        auto out = std::make_shared<Tensor>(a->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float b_val = b->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < a->size(); ++i) {
            out->data[i] = std::max(a->data[i], b_val);
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                float b_val = b->data[0];
                float b_grad_sum = 0;
                #pragma omp parallel for simd reduction(+:b_grad_sum)
                for (size_t i = 0; i < a->size(); ++i) {
                    if (a->data[i] > b_val) {
                        if (a->requires_grad) a->grad[i] += out->grad[i];
                    } else if (b_val > a->data[i]) {
                        b_grad_sum += out->grad[i];
                    } else {
                        if (a->requires_grad) a->grad[i] += 0.5f * out->grad[i];
                        b_grad_sum += 0.5f * out->grad[i];
                    }
                }
                if (b->requires_grad) b->grad[0] += b_grad_sum;
            };
        }

        return out;
    }
    // Broadcasting: a is scalar, b is larger
    else if (a->size() == 1) {
        auto out = std::make_shared<Tensor>(b->shape(), a->requires_grad || b->requires_grad);
        out->children = {a, b};
        out->is_leaf = false;

        float a_val = a->data[0];
        #pragma omp parallel for simd
        for (size_t i = 0; i < b->size(); ++i) {
            out->data[i] = std::max(a_val, b->data[i]);
        }

        if (out->requires_grad) {
            out->backward_fn = [a, b, out]() {
                float a_val = a->data[0];
                float a_grad_sum = 0;
                #pragma omp parallel for simd reduction(+:a_grad_sum)
                for (size_t i = 0; i < b->size(); ++i) {
                    if (a_val > b->data[i]) {
                        a_grad_sum += out->grad[i];
                    } else if (b->data[i] > a_val) {
                        if (b->requires_grad) b->grad[i] += out->grad[i];
                    } else {
                        a_grad_sum += 0.5f * out->grad[i];
                        if (b->requires_grad) b->grad[i] += 0.5f * out->grad[i];
                    }
                }
                if (a->requires_grad) a->grad[0] += a_grad_sum;
            };
        }

        return out;
    }
    else {
        throw std::runtime_error("Shape mismatch in maximum: incompatible shapes for broadcasting");
    }
}

} // namespace autograd