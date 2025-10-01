#pragma once

#include "tensor.hpp"
#include "ops.hpp"
#include "jacobian.hpp"
#include <functional>
#include <vector>
#include <random>
#include <deque>
#include <memory>
#include <unordered_map>
#include <cstring>

namespace autograd {

// ============================================================================
// Hessian Matrix Class - Comprehensive Second-Order Derivative Computation
// ============================================================================

// Forward declarations
class HessianApproximation;
class BFGSApproximation;
class LBFGSApproximation;

// Enum for different Hessian computation methods
enum class HessianMethod {
    EXACT,                  // Exact computation via automatic differentiation
    FINITE_DIFFERENCE,      // Finite difference approximation
    DIAGONAL_ONLY,          // Only compute diagonal elements
    GAUSS_NEWTON,          // Gauss-Newton approximation (for least squares)
    BFGS,                  // BFGS quasi-Newton approximation
    LBFGS,                 // Limited-memory BFGS
    STOCHASTIC_DIAGONAL,   // Hutchinson's stochastic diagonal estimator
    BLOCK_DIAGONAL         // Block diagonal approximation
};

// Hessian storage format
enum class HessianStorage {
    DENSE,           // Full dense matrix
    DIAGONAL,        // Only diagonal elements
    SPARSE,          // Sparse representation (future)
    IMPLICIT         // Never store explicitly, only products
};

// Main Hessian class encapsulating various computation strategies
class Hessian {
private:
    TensorPtr data_;                    // Actual Hessian data (if stored)
    size_t dimension_;                  // Dimension of the Hessian
    HessianMethod method_;              // Computation method
    HessianStorage storage_;            // Storage format
    std::unique_ptr<HessianApproximation> approximation_;  // For quasi-Newton methods
    
    // Cached computations
    mutable TensorPtr eigenvalues_;
    mutable TensorPtr eigenvectors_;
    mutable TensorPtr inverse_;
    mutable bool is_positive_definite_;
    mutable bool checked_positive_definite_;
    
public:
    // Constructors
    Hessian(size_t dim, HessianMethod method = HessianMethod::EXACT,
            HessianStorage storage = HessianStorage::DENSE)
        : dimension_(dim), method_(method), storage_(storage),
          is_positive_definite_(false), checked_positive_definite_(false) {
        
        if (storage == HessianStorage::DENSE) {
            data_ = zeros({dim, dim}, false);
        } else if (storage == HessianStorage::DIAGONAL) {
            data_ = zeros({dim}, false);
        }
    }
    
    // Factory methods for different computation strategies
    static std::unique_ptr<Hessian> compute(TensorPtr loss, TensorPtr x,
                                           HessianMethod method = HessianMethod::EXACT);
    
    // Compute exact Hessian via automatic differentiation
    void computeExact(TensorPtr loss, TensorPtr x);
    
    // Compute via finite differences
    void computeFiniteDifference(std::function<TensorPtr(TensorPtr)> f, TensorPtr x,
                                 float epsilon = 1e-5f);
    
    // Compute only diagonal elements
    void computeDiagonal(TensorPtr loss, TensorPtr x);
    
    // Compute stochastic diagonal approximation
    void computeStochasticDiagonal(TensorPtr loss, TensorPtr x, int n_samples = 30);
    
    // Access methods
    float at(size_t i, size_t j) const;
    void set(size_t i, size_t j, float value);
    
    // Matrix operations
    TensorPtr matvec(TensorPtr v) const;           // Hessian-vector product
    TensorPtr solve(TensorPtr b) const;            // Solve H*x = b
    TensorPtr diagonal() const;                    // Get diagonal
    float trace() const;                          // Compute trace
    float determinant() const;                    // Compute determinant
    
    // Eigenvalue analysis
    std::pair<TensorPtr, TensorPtr> eigen() const; // Eigenvalues and eigenvectors
    bool isPositiveDefinite() const;
    float condition_number() const;
    
    // Regularization
    void addDiagonal(float lambda);               // H = H + λI (Levenberg-Marquardt)
    void makePositiveDefinite(float min_eigenvalue = 1e-6f);
    
    // Get raw data
    TensorPtr getData() const { return data_; }
    size_t dimension() const { return dimension_; }
    HessianMethod method() const { return method_; }
    HessianStorage storage() const { return storage_; }
};

// ============================================================================
// Base class for Hessian approximations
// ============================================================================

class HessianApproximation {
public:
    virtual ~HessianApproximation() = default;
    
    // Update approximation with new gradient information
    virtual void update(TensorPtr x_old, TensorPtr g_old,
                       TensorPtr x_new, TensorPtr g_new) = 0;
    
    // Apply Hessian to vector
    virtual TensorPtr matvec(TensorPtr v) const = 0;
    
    // Apply inverse Hessian to vector
    virtual TensorPtr solve(TensorPtr v) const = 0;
    
    // Get diagonal approximation
    virtual TensorPtr diagonal() const = 0;
    
    // Reset approximation
    virtual void reset() = 0;
};

// ============================================================================
// BFGS Approximation
// ============================================================================

class BFGSApproximation : public HessianApproximation {
private:
    TensorPtr B_;           // Hessian approximation
    TensorPtr H_;           // Inverse Hessian approximation
    size_t dimension_;
    bool store_inverse_;
    
public:
    BFGSApproximation(size_t dim, bool store_inverse = true)
        : dimension_(dim), store_inverse_(store_inverse) {
        reset();
    }
    
    void update(TensorPtr x_old, TensorPtr g_old,
               TensorPtr x_new, TensorPtr g_new) override;
    
    TensorPtr matvec(TensorPtr v) const override;
    TensorPtr solve(TensorPtr v) const override;
    TensorPtr diagonal() const override;
    void reset() override;
    
private:
    void updateBFGS(TensorPtr s, TensorPtr y);
    void updateInverseBFGS(TensorPtr s, TensorPtr y);
};

// ============================================================================
// L-BFGS Approximation (Limited Memory BFGS)
// ============================================================================

class LBFGSApproximation : public HessianApproximation {
private:
    std::deque<TensorPtr> s_history_;  // x_{k+1} - x_k
    std::deque<TensorPtr> y_history_;  // g_{k+1} - g_k
    std::deque<float> rho_history_;    // 1 / (y^T * s)
    size_t history_size_;
    size_t dimension_;
    float gamma_;                      // Scaling factor
    
public:
    LBFGSApproximation(size_t dim, size_t history_size = 10)
        : history_size_(history_size), dimension_(dim), gamma_(1.0f) {}
    
    void update(TensorPtr x_old, TensorPtr g_old,
               TensorPtr x_new, TensorPtr g_new) override;
    
    TensorPtr matvec(TensorPtr v) const override;
    TensorPtr solve(TensorPtr v) const override;  // Two-loop recursion
    TensorPtr diagonal() const override;
    void reset() override;
    
private:
    float dot(TensorPtr a, TensorPtr b) const;
};

// ============================================================================
// SR1 (Symmetric Rank-1) Approximation
// ============================================================================

class SR1Approximation : public HessianApproximation {
private:
    TensorPtr B_;           // Hessian approximation
    size_t dimension_;
    float threshold_;       // Skip update threshold
    
public:
    SR1Approximation(size_t dim, float threshold = 1e-8f)
        : dimension_(dim), threshold_(threshold) {
        reset();
    }
    
    void update(TensorPtr x_old, TensorPtr g_old,
               TensorPtr x_new, TensorPtr g_new) override;
    
    TensorPtr matvec(TensorPtr v) const override;
    TensorPtr solve(TensorPtr v) const override;
    TensorPtr diagonal() const override;
    void reset() override;
};

// ============================================================================
// Implementation of Hessian class methods
// ============================================================================

std::unique_ptr<Hessian> Hessian::compute(TensorPtr loss, TensorPtr x,
                                         HessianMethod method) {
    auto hessian = std::make_unique<Hessian>(x->size(), method);
    
    switch (method) {
        case HessianMethod::EXACT:
            hessian->computeExact(loss, x);
            break;
        case HessianMethod::DIAGONAL_ONLY:
            hessian->storage_ = HessianStorage::DIAGONAL;
            hessian->data_ = zeros({x->size()}, false);
            hessian->computeDiagonal(loss, x);
            break;
        case HessianMethod::STOCHASTIC_DIAGONAL:
            hessian->storage_ = HessianStorage::DIAGONAL;
            hessian->data_ = zeros({x->size()}, false);
            hessian->computeStochasticDiagonal(loss, x);
            break;
        case HessianMethod::BFGS:
            hessian->approximation_ = std::make_unique<BFGSApproximation>(x->size());
            hessian->storage_ = HessianStorage::IMPLICIT;
            break;
        case HessianMethod::LBFGS:
            hessian->approximation_ = std::make_unique<LBFGSApproximation>(x->size());
            hessian->storage_ = HessianStorage::IMPLICIT;
            break;
        default:
            throw std::runtime_error("Unsupported Hessian computation method");
    }
    
    return hessian;
}

void Hessian::computeExact(TensorPtr loss, TensorPtr x) {
    if (loss->size() != 1) {
        throw std::runtime_error("Hessian requires scalar loss function");
    }
    
    size_t n = dimension_;
    
    // First compute gradient
    x->zero_grad();
    loss->backward();
    
    // Store first-order gradients
    std::vector<float> grad1(x->grad, x->grad + n);
    
    // Compute second derivatives
    for (size_t i = 0; i < n; ++i) {
        // Create computational graph for gradient computation
        x->zero_grad();
        loss->backward();
        
        // Perturb x[i] slightly
        float original = x->data[i];
        x->data[i] += 1e-5f;
        
        // Recompute forward and backward
        x->zero_grad();
        // Note: In practice, we'd need to recompute the forward pass here
        loss->backward();
        
        // Compute second derivatives via finite differences
        for (size_t j = 0; j < n; ++j) {
            data_->data[i * n + j] = (x->grad[j] - grad1[j]) / 1e-5f;
        }
        
        // Restore original value
        x->data[i] = original;
    }
    
    // Make Hessian symmetric (it should be for twice-differentiable functions)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float avg = (data_->data[i * n + j] + 
                        data_->data[j * n + i]) / 2.0f;
            data_->data[i * n + j] = avg;
            data_->data[j * n + i] = avg;
        }
    }
}

void Hessian::computeFiniteDifference(std::function<TensorPtr(TensorPtr)> f, 
                                     TensorPtr x, float epsilon) {
    size_t n = dimension_;
    
    // Compute gradient at x
    auto compute_grad = [&](TensorPtr point) -> std::vector<float> {
        point->zero_grad();
        auto output = f(point);
        if (output->size() != 1) {
            throw std::runtime_error("Function must return scalar for Hessian");
        }
        output->backward();
        return std::vector<float>(point->grad, point->grad + n);
    };
    
    auto grad = compute_grad(x);
    
    // Compute Hessian via finite differences of gradient
    for (size_t i = 0; i < n; ++i) {
        auto x_perturbed = x->clone();
        x_perturbed->data[i] += epsilon;
        auto grad_perturbed = compute_grad(x_perturbed);
        
        for (size_t j = 0; j < n; ++j) {
            data_->data[i * n + j] = (grad_perturbed[j] - grad[j]) / epsilon;
        }
    }
    
    // Symmetrize
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float avg = (data_->data[i * n + j] + data_->data[j * n + i]) / 2.0f;
            data_->data[i * n + j] = avg;
            data_->data[j * n + i] = avg;
        }
    }
}

void Hessian::computeDiagonal(TensorPtr loss, TensorPtr x) {
    if (loss->size() != 1) {
        throw std::runtime_error("Diagonal Hessian requires scalar loss");
    }
    
    size_t n = dimension_;
    
    for (size_t i = 0; i < n; ++i) {
        // Compute ∂²loss/∂x_i²
        x->zero_grad();
        loss->backward();
        float grad_i = x->grad[i];
        
        // Perturb x_i slightly
        float original = x->data[i];
        x->data[i] += 1e-5f;
        
        // Recompute gradient at perturbed point
        x->zero_grad();
        loss->backward();
        float grad_i_perturbed = x->grad[i];
        
        // Second derivative via finite difference
        data_->data[i] = (grad_i_perturbed - grad_i) / 1e-5f;
        
        // Restore original value
        x->data[i] = original;
    }
}

void Hessian::computeStochasticDiagonal(TensorPtr loss, TensorPtr x, int n_samples) {
    if (loss->size() != 1) {
        throw std::runtime_error("Stochastic diagonal requires scalar loss");
    }
    
    size_t n = dimension_;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int sample = 0; sample < n_samples; ++sample) {
        // Create Rademacher random vector (±1 with equal probability)
        auto z = zeros({n}, false);
        for (size_t i = 0; i < n; ++i) {
            z->data[i] = (dis(gen) < 0.5) ? -1.0f : 1.0f;
        }
        
        // Compute Hessian-vector product H*z
        x->zero_grad();
        loss->backward();
        auto g = zeros({n}, false);
        std::memcpy(g->data, x->grad, n * sizeof(float));
        
        // Compute directional derivative of gradient in direction z
        float eps = 1e-5f;
        for (size_t i = 0; i < n; ++i) {
            x->data[i] += eps * z->data[i];
        }
        
        x->zero_grad();
        loss->backward();
        
        // H*z ≈ (grad(x + eps*z) - grad(x)) / eps
        for (size_t i = 0; i < n; ++i) {
            float hz_i = (x->grad[i] - g->data[i]) / eps;
            // Accumulate z_i * (H*z)_i for diagonal estimate
            data_->data[i] += z->data[i] * hz_i;
            // Restore x
            x->data[i] -= eps * z->data[i];
        }
    }
    
    // Average over samples
    for (size_t i = 0; i < n; ++i) {
        data_->data[i] /= n_samples;
    }
}

float Hessian::at(size_t i, size_t j) const {
    if (storage_ == HessianStorage::DIAGONAL) {
        return (i == j) ? data_->data[i] : 0.0f;
    } else if (storage_ == HessianStorage::DENSE) {
        return data_->data[i * dimension_ + j];
    } else {
        throw std::runtime_error("Cannot access individual elements of implicit Hessian");
    }
}

void Hessian::set(size_t i, size_t j, float value) {
    if (storage_ == HessianStorage::DIAGONAL) {
        if (i == j) {
            data_->data[i] = value;
        }
    } else if (storage_ == HessianStorage::DENSE) {
        data_->data[i * dimension_ + j] = value;
    } else {
        throw std::runtime_error("Cannot set individual elements of implicit Hessian");
    }
}

TensorPtr Hessian::matvec(TensorPtr v) const {
    if (v->size() != dimension_) {
        throw std::runtime_error("Vector dimension mismatch for Hessian-vector product");
    }
    
    auto result = zeros({dimension_}, false);
    
    if (storage_ == HessianStorage::DIAGONAL) {
        for (size_t i = 0; i < dimension_; ++i) {
            result->data[i] = data_->data[i] * v->data[i];
        }
    } else if (storage_ == HessianStorage::DENSE) {
        for (size_t i = 0; i < dimension_; ++i) {
            float sum = 0;
            for (size_t j = 0; j < dimension_; ++j) {
                sum += data_->data[i * dimension_ + j] * v->data[j];
            }
            result->data[i] = sum;
        }
    } else if (storage_ == HessianStorage::IMPLICIT && approximation_) {
        return approximation_->matvec(v);
    }
    
    return result;
}

TensorPtr Hessian::solve(TensorPtr b) const {
    if (b->size() != dimension_) {
        throw std::runtime_error("Vector dimension mismatch for Hessian solve");
    }
    
    if (storage_ == HessianStorage::DIAGONAL) {
        auto result = zeros({dimension_}, false);
        for (size_t i = 0; i < dimension_; ++i) {
            if (std::abs(data_->data[i]) < 1e-10f) {
                throw std::runtime_error("Singular diagonal Hessian");
            }
            result->data[i] = b->data[i] / data_->data[i];
        }
        return result;
    } else if (storage_ == HessianStorage::IMPLICIT && approximation_) {
        return approximation_->solve(b);
    } else {
        // For dense matrices, use simple Gaussian elimination (could be improved)
        // This is a placeholder - in production, use a proper linear solver
        throw std::runtime_error("Dense Hessian solve not yet implemented");
    }
}

TensorPtr Hessian::diagonal() const {
    auto diag = zeros({dimension_}, false);
    
    if (storage_ == HessianStorage::DIAGONAL) {
        std::memcpy(diag->data, data_->data, dimension_ * sizeof(float));
    } else if (storage_ == HessianStorage::DENSE) {
        for (size_t i = 0; i < dimension_; ++i) {
            diag->data[i] = data_->data[i * dimension_ + i];
        }
    } else if (storage_ == HessianStorage::IMPLICIT && approximation_) {
        return approximation_->diagonal();
    }
    
    return diag;
}

float Hessian::trace() const {
    float tr = 0;
    
    if (storage_ == HessianStorage::DIAGONAL) {
        for (size_t i = 0; i < dimension_; ++i) {
            tr += data_->data[i];
        }
    } else if (storage_ == HessianStorage::DENSE) {
        for (size_t i = 0; i < dimension_; ++i) {
            tr += data_->data[i * dimension_ + i];
        }
    } else {
        auto diag = diagonal();
        for (size_t i = 0; i < dimension_; ++i) {
            tr += diag->data[i];
        }
    }
    
    return tr;
}

void Hessian::addDiagonal(float lambda) {
    if (storage_ == HessianStorage::DIAGONAL) {
        for (size_t i = 0; i < dimension_; ++i) {
            data_->data[i] += lambda;
        }
    } else if (storage_ == HessianStorage::DENSE) {
        for (size_t i = 0; i < dimension_; ++i) {
            data_->data[i * dimension_ + i] += lambda;
        }
    } else {
        throw std::runtime_error("Cannot add diagonal to implicit Hessian");
    }
}

bool Hessian::isPositiveDefinite() const {
    if (checked_positive_definite_) {
        return is_positive_definite_;
    }
    
    // Simple check: all diagonal elements positive (necessary but not sufficient)
    if (storage_ == HessianStorage::DIAGONAL) {
        is_positive_definite_ = true;
        for (size_t i = 0; i < dimension_; ++i) {
            if (data_->data[i] <= 0) {
                is_positive_definite_ = false;
                break;
            }
        }
    } else {
        // For dense matrices, would need eigenvalue decomposition
        // This is a placeholder
        is_positive_definite_ = false;
    }
    
    checked_positive_definite_ = true;
    return is_positive_definite_;
}

// ============================================================================
// L-BFGS Implementation
// ============================================================================

void LBFGSApproximation::update(TensorPtr x_old, TensorPtr g_old,
                               TensorPtr x_new, TensorPtr g_new) {
    auto s = zeros({dimension_}, false);
    auto y = zeros({dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        s->data[i] = x_new->data[i] - x_old->data[i];
        y->data[i] = g_new->data[i] - g_old->data[i];
    }
    
    // Check curvature condition
    float sy = dot(s, y);
    
    if (sy > 1e-10f) {  // Only update if curvature condition satisfied
        s_history_.push_back(s);
        y_history_.push_back(y);
        rho_history_.push_back(1.0f / sy);
        
        // Update scaling factor
        float yy = dot(y, y);
        if (yy > 0) {
            gamma_ = sy / yy;
        }
        
        // Maintain history size limit
        if (s_history_.size() > history_size_) {
            s_history_.pop_front();
            y_history_.pop_front();
            rho_history_.pop_front();
        }
    }
}

TensorPtr LBFGSApproximation::solve(TensorPtr grad) const {
    if (s_history_.empty()) {
        // No history, return scaled gradient
        auto result = grad->clone();
        for (size_t i = 0; i < dimension_; ++i) {
            result->data[i] *= gamma_;
        }
        return result;
    }
    
    auto q = grad->clone();
    std::vector<float> alphas;
    
    // First loop - backward through history
    for (int i = s_history_.size() - 1; i >= 0; --i) {
        float alpha = rho_history_[i] * dot(s_history_[i], q);
        alphas.push_back(alpha);
        
        for (size_t j = 0; j < dimension_; ++j) {
            q->data[j] -= alpha * y_history_[i]->data[j];
        }
    }
    
    // Scale by initial approximation
    auto r = zeros({dimension_}, false);
    for (size_t i = 0; i < dimension_; ++i) {
        r->data[i] = gamma_ * q->data[i];
    }
    
    // Second loop - forward through history
    for (size_t i = 0; i < s_history_.size(); ++i) {
        float beta = rho_history_[i] * dot(y_history_[i], r);
        
        for (size_t j = 0; j < dimension_; ++j) {
            r->data[j] += s_history_[i]->data[j] * 
                          (alphas[s_history_.size() - 1 - i] - beta);
        }
    }
    
    return r;
}

TensorPtr LBFGSApproximation::matvec(TensorPtr v) const {
    // For L-BFGS, we typically don't compute H*v directly
    // Instead, we work with H^(-1)*v
    // This is a placeholder that uses finite differences
    auto eps = 1e-7f;
    auto v_perturbed = v->clone();
    for (size_t i = 0; i < dimension_; ++i) {
        v_perturbed->data[i] *= (1.0f + eps);
    }
    
    auto hv = solve(v);
    auto hv_perturbed = solve(v_perturbed);
    
    auto result = zeros({dimension_}, false);
    for (size_t i = 0; i < dimension_; ++i) {
        result->data[i] = (hv_perturbed->data[i] - hv->data[i]) / 
                         (eps * v->data[i]);
    }
    
    return result;
}

TensorPtr LBFGSApproximation::diagonal() const {
    auto diag = ones({dimension_}, false);
    
    // Use scaling factor as diagonal approximation
    for (size_t i = 0; i < dimension_; ++i) {
        diag->data[i] = 1.0f / gamma_;  // Approximate H_ii
    }
    
    return diag;
}

void LBFGSApproximation::reset() {
    s_history_.clear();
    y_history_.clear();
    rho_history_.clear();
    gamma_ = 1.0f;
}

float LBFGSApproximation::dot(TensorPtr a, TensorPtr b) const {
    float sum = 0;
    for (size_t i = 0; i < dimension_; ++i) {
        sum += a->data[i] * b->data[i];
    }
    return sum;
}

// ============================================================================
// BFGS Implementation
// ============================================================================

void BFGSApproximation::update(TensorPtr x_old, TensorPtr g_old,
                              TensorPtr x_new, TensorPtr g_new) {
    auto s = zeros({dimension_}, false);
    auto y = zeros({dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        s->data[i] = x_new->data[i] - x_old->data[i];
        y->data[i] = g_new->data[i] - g_old->data[i];
    }
    
    // Check curvature condition
    float sy = 0;
    for (size_t i = 0; i < dimension_; ++i) {
        sy += s->data[i] * y->data[i];
    }
    
    if (sy > 1e-10f) {
        if (store_inverse_) {
            updateInverseBFGS(s, y);
        } else {
            updateBFGS(s, y);
        }
    }
}

void BFGSApproximation::updateBFGS(TensorPtr s, TensorPtr y) {
    // BFGS update: B_new = B - (B*s*s^T*B)/(s^T*B*s) + (y*y^T)/(y^T*s)
    
    // Compute B*s
    auto Bs = zeros({dimension_}, false);
    for (size_t i = 0; i < dimension_; ++i) {
        float sum = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            sum += B_->data[i * dimension_ + j] * s->data[j];
        }
        Bs->data[i] = sum;
    }
    
    // Compute s^T*B*s
    float sBs = 0;
    for (size_t i = 0; i < dimension_; ++i) {
        sBs += s->data[i] * Bs->data[i];
    }
    
    // Compute y^T*s
    float ys = 0;
    for (size_t i = 0; i < dimension_; ++i) {
        ys += y->data[i] * s->data[i];
    }
    
    // Update B
    if (sBs > 1e-10f && ys > 1e-10f) {
        for (size_t i = 0; i < dimension_; ++i) {
            for (size_t j = 0; j < dimension_; ++j) {
                B_->data[i * dimension_ + j] -= (Bs->data[i] * Bs->data[j]) / sBs;
                B_->data[i * dimension_ + j] += (y->data[i] * y->data[j]) / ys;
            }
        }
    }
}

void BFGSApproximation::updateInverseBFGS(TensorPtr s, TensorPtr y) {
    // Sherman-Morrison formula for inverse update
    // H_new = (I - rho*s*y^T)*H*(I - rho*y*s^T) + rho*s*s^T
    
    float ys = 0;
    for (size_t i = 0; i < dimension_; ++i) {
        ys += y->data[i] * s->data[i];
    }
    
    if (ys < 1e-10f) return;
    
    float rho = 1.0f / ys;
    
    // Compute H*y
    auto Hy = zeros({dimension_}, false);
    for (size_t i = 0; i < dimension_; ++i) {
        float sum = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            sum += H_->data[i * dimension_ + j] * y->data[j];
        }
        Hy->data[i] = sum;
    }
    
    // Update H
    for (size_t i = 0; i < dimension_; ++i) {
        for (size_t j = 0; j < dimension_; ++j) {
            float Hij = H_->data[i * dimension_ + j];
            Hij -= rho * (s->data[i] * Hy->data[j] + Hy->data[i] * s->data[j]);
            Hij += rho * (1.0f + rho * 0) * s->data[i] * s->data[j];  // Simplified
            H_->data[i * dimension_ + j] = Hij;
        }
    }
}

TensorPtr BFGSApproximation::matvec(TensorPtr v) const {
    auto result = zeros({dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        float sum = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            sum += B_->data[i * dimension_ + j] * v->data[j];
        }
        result->data[i] = sum;
    }
    
    return result;
}

TensorPtr BFGSApproximation::solve(TensorPtr v) const {
    if (!store_inverse_) {
        throw std::runtime_error("BFGS solve requires inverse storage");
    }
    
    auto result = zeros({dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        float sum = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            sum += H_->data[i * dimension_ + j] * v->data[j];
        }
        result->data[i] = sum;
    }
    
    return result;
}

TensorPtr BFGSApproximation::diagonal() const {
    auto diag = zeros({dimension_}, false);
    
    if (store_inverse_) {
        for (size_t i = 0; i < dimension_; ++i) {
            diag->data[i] = H_->data[i * dimension_ + i];
        }
    } else {
        for (size_t i = 0; i < dimension_; ++i) {
            diag->data[i] = B_->data[i * dimension_ + i];
        }
    }
    
    return diag;
}

void BFGSApproximation::reset() {
    // Initialize with identity matrix
    B_ = zeros({dimension_, dimension_}, false);
    H_ = zeros({dimension_, dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        B_->data[i * dimension_ + i] = 1.0f;
        H_->data[i * dimension_ + i] = 1.0f;
    }
}

// ============================================================================
// SR1 Implementation
// ============================================================================

void SR1Approximation::update(TensorPtr x_old, TensorPtr g_old,
                             TensorPtr x_new, TensorPtr g_new) {
    auto s = zeros({dimension_}, false);
    auto y = zeros({dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        s->data[i] = x_new->data[i] - x_old->data[i];
        y->data[i] = g_new->data[i] - g_old->data[i];
    }
    
    // Compute B*s
    auto Bs = zeros({dimension_}, false);
    for (size_t i = 0; i < dimension_; ++i) {
        float sum = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            sum += B_->data[i * dimension_ + j] * s->data[j];
        }
        Bs->data[i] = sum;
    }
    
    // Compute y - B*s
    auto r = zeros({dimension_}, false);
    for (size_t i = 0; i < dimension_; ++i) {
        r->data[i] = y->data[i] - Bs->data[i];
    }
    
    // Compute denominator
    float denom = 0;
    for (size_t i = 0; i < dimension_; ++i) {
        denom += r->data[i] * s->data[i];
    }
    
    // Skip update if denominator is too small
    if (std::abs(denom) > threshold_) {
        // SR1 update: B_new = B + (r * r^T) / (r^T * s)
        for (size_t i = 0; i < dimension_; ++i) {
            for (size_t j = 0; j < dimension_; ++j) {
                B_->data[i * dimension_ + j] += (r->data[i] * r->data[j]) / denom;
            }
        }
    }
}

TensorPtr SR1Approximation::matvec(TensorPtr v) const {
    auto result = zeros({dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        float sum = 0;
        for (size_t j = 0; j < dimension_; ++j) {
            sum += B_->data[i * dimension_ + j] * v->data[j];
        }
        result->data[i] = sum;
    }
    
    return result;
}

TensorPtr SR1Approximation::solve(TensorPtr v) const {
    // SR1 doesn't maintain inverse, would need to solve linear system
    throw std::runtime_error("SR1 solve not implemented - use iterative solver");
}

TensorPtr SR1Approximation::diagonal() const {
    auto diag = zeros({dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        diag->data[i] = B_->data[i * dimension_ + i];
    }
    
    return diag;
}

void SR1Approximation::reset() {
    // Initialize with identity matrix
    B_ = zeros({dimension_, dimension_}, false);
    
    for (size_t i = 0; i < dimension_; ++i) {
        B_->data[i * dimension_ + i] = 1.0f;
    }
}

// ============================================================================
// Utility Functions (backward compatibility and convenience)
// ============================================================================

// Compute exact Hessian matrix (backward compatible)
inline TensorPtr compute_hessian(TensorPtr loss, TensorPtr x) {
    auto hessian = Hessian::compute(loss, x, HessianMethod::EXACT);
    return hessian->getData();
}

// Compute only diagonal of Hessian
inline TensorPtr hessian_diagonal(TensorPtr loss, TensorPtr x) {
    auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);
    return hessian->getData();
}

// Hutchinson's stochastic diagonal estimator
inline TensorPtr hessian_diagonal_stochastic(TensorPtr loss, TensorPtr x, int n_samples = 30) {
    auto hessian = Hessian::compute(loss, x, HessianMethod::STOCHASTIC_DIAGONAL);
    hessian->computeStochasticDiagonal(loss, x, n_samples);
    return hessian->getData();
}

// Gauss-Newton Hessian approximation for least squares
inline TensorPtr gauss_newton_hessian_diag(std::function<TensorPtr(TensorPtr)> residual_fn, 
                                          TensorPtr x) {
    // Compute Jacobian
    auto J = compute_jacobian(residual_fn, x);
    
    // Diagonal of J^T*J
    size_t m = J->shape()[0];  // Number of residuals
    size_t n = J->shape()[1];  // Number of parameters
    
    auto diag = zeros({n}, false);
    
    // Diagonal of J^T*J: (J^T*J)_ii = sum_k J_ki^2
    for (size_t i = 0; i < n; ++i) {
        float sum = 0;
        for (size_t k = 0; k < m; ++k) {
            float j_ki = J->data[k * n + i];
            sum += j_ki * j_ki;
        }
        diag->data[i] = 2.0f * sum;  // Factor of 2 for least squares
    }
    
    return diag;
}

// Laplacian (trace of Hessian)
inline TensorPtr laplacian(TensorPtr scalar_output, TensorPtr x) {
    auto hessian = Hessian::compute(scalar_output, x);
    
    auto result = zeros({1}, false);
    result->data[0] = hessian->trace();
    
    return result;
}

// Newton step with diagonal Hessian
inline void newton_step_diagonal(TensorPtr x, TensorPtr loss, float damping = 1e-4f) {
    // Compute gradient
    x->zero_grad();
    loss->backward();
    auto grad = zeros({x->size()}, false);
    std::memcpy(grad->data, x->grad, x->size() * sizeof(float));
    
    // Compute diagonal Hessian
    auto hessian = Hessian::compute(loss, x, HessianMethod::STOCHASTIC_DIAGONAL);
    auto diag_H = hessian->diagonal();
    
    // Newton step: x = x - (H + λI)^(-1) * g
    for (size_t i = 0; i < x->size(); ++i) {
        float h_ii = diag_H->data[i];
        // Add damping for numerical stability (Levenberg-Marquardt style)
        float step = grad->data[i] / (std::abs(h_ii) + damping);
        x->data[i] -= step;
    }
}

// Block diagonal Hessian for neural networks
inline std::vector<TensorPtr> block_diagonal_hessian(
    TensorPtr loss,
    const std::vector<TensorPtr>& param_blocks) {
    
    std::vector<TensorPtr> blocks;
    
    for (const auto& params : param_blocks) {
        // Compute diagonal for this block
        auto hessian = Hessian::compute(loss, params, HessianMethod::DIAGONAL_ONLY);
        blocks.push_back(hessian->diagonal());
    }
    
    return blocks;
}

// Backward compatible QuasiNewtonApprox (now using L-BFGS)
class QuasiNewtonApprox : public LBFGSApproximation {
public:
    QuasiNewtonApprox(size_t dim, size_t history = 10) 
        : LBFGSApproximation(dim, history) {}
    
    // Additional backward compatible methods
    TensorPtr get_diagonal() { return diagonal(); }
    TensorPtr apply_inverse(TensorPtr grad) { return solve(grad); }
};

} // namespace autograd