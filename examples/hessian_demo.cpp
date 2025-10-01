// Hessian computation and approximation demonstration
// Shows various methods for computing and using the Hessian matrix

#include "../include/autograd/autograd.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <cmath>

using namespace autograd;
using namespace std::chrono;

// Timer utility
class Timer {
    high_resolution_clock::time_point start;
public:
    Timer() : start(high_resolution_clock::now()) {}
    double elapsed() {
        auto end = high_resolution_clock::now();
        return duration_cast<microseconds>(end - start).count() / 1000.0;
    }
};

// Rosenbrock function - a classic test for optimization algorithms
// f(x, y) = (a - x)^2 + b*(y - x^2)^2
// Has a global minimum at (a, a^2)
TensorPtr rosenbrock(TensorPtr params, float a = 1.0f, float b = 100.0f) {
    // Extract x and y from params vector
    auto x = from_vector({params->data[0]}, {1}, true);
    auto y = from_vector({params->data[1]}, {1}, true);
    auto a_tensor = from_vector({a}, {1}, false);
    auto b_tensor = from_vector({b}, {1}, false);
    
    auto term1 = mul(sub(a_tensor, x), sub(a_tensor, x));
    auto x_squared = mul(x, x);
    auto diff = sub(y, x_squared);
    auto term2 = mul(b_tensor, mul(diff, diff));
    
    return add(term1, term2);
}

// Quadratic function for testing
// f(x) = 0.5 * x^T * A * x + b^T * x + c
TensorPtr quadratic(TensorPtr x) {
    // Simple 3D quadratic with diagonal A matrix
    auto x0 = from_vector({x->data[0]}, {1}, true);
    auto x1 = from_vector({x->data[1]}, {1}, true);
    auto x2 = from_vector({x->data[2]}, {1}, true);
    
    auto coef2 = from_vector({2.0f}, {1}, false);
    auto coef3 = from_vector({3.0f}, {1}, false);
    auto coef4 = from_vector({4.0f}, {1}, false);
    auto coef15 = from_vector({1.5f}, {1}, false);
    
    auto term1 = mul(coef2, mul(x0, x0));
    auto term2 = mul(coef3, mul(x1, x1));
    auto term3 = mul(coef4, mul(x2, x2));
    auto term4 = mul(coef15, mul(x0, x1));  // Cross term
    
    auto quad_terms = add(add(add(term1, term2), term3), term4);
    auto linear_terms = add(add(x0, mul(coef2, x1)), mul(coef3, x2));
    
    return sub(quad_terms, linear_terms);
}

// Logistic regression loss for ML example
TensorPtr logistic_loss(TensorPtr theta, TensorPtr X, TensorPtr y) {
    // L(theta) = -sum(y * log(sigmoid(X * theta)) + (1-y) * log(1 - sigmoid(X * theta)))
    auto z = matmul(X, theta);
    auto probs = sigmoid(z);
    
    auto eps = from_vector({1e-7f}, {1}, false);  // For numerical stability
    auto one = from_vector({1.0f}, {1}, false);
    auto probs_safe = add(probs, eps);
    auto one_minus_probs_safe = add(sub(one, probs), eps);
    
    auto term1 = mul(y, log(probs_safe));
    auto term2 = mul(sub(one, y), log(one_minus_probs_safe));
    auto neg_loss = mean(add(term1, term2));
    
    return mul(from_vector({-1.0f}, {1}, false), neg_loss);
}

void print_matrix(const std::string& name, TensorPtr mat, size_t rows, size_t cols) {
    std::cout << name << ":\n";
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) 
                     << std::setw(10) << mat->data[i * cols + j];
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void print_vector(const std::string& name, TensorPtr vec) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec->size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec->data[i];
        if (i < vec->size() - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "    Hessian Computation Demonstration   \n";
    std::cout << "========================================\n\n";
    
    // Test 1: Exact Hessian computation on Rosenbrock function
    std::cout << "1. EXACT HESSIAN - Rosenbrock Function\n";
    std::cout << "---------------------------------------\n";
    {
        auto params = from_vector({0.5f, 0.5f}, {2}, true);
        auto loss = rosenbrock(params);
        
        Timer timer;
        auto hessian = Hessian::compute(loss, params, HessianMethod::EXACT);
        double elapsed = timer.elapsed();
        
        std::cout << "Point: [" << params->data[0] << ", " << params->data[1] << "]\n";
        std::cout << "Function value: " << loss->data[0] << "\n";
        print_matrix("Hessian", hessian->getData(), 2, 2);
        std::cout << "Trace: " << hessian->trace() << "\n";
        std::cout << "Time: " << elapsed << " ms\n\n";
    }
    
    // Test 2: Diagonal Hessian approximation
    std::cout << "2. DIAGONAL HESSIAN - Quadratic Function\n";
    std::cout << "----------------------------------------\n";
    {
        auto x = from_vector({1.0f, -0.5f, 2.0f}, {3}, true);
        auto loss = quadratic(x);
        
        // Exact diagonal
        Timer timer1;
        auto hessian_diag = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);
        double elapsed1 = timer1.elapsed();
        
        // Stochastic diagonal
        Timer timer2;
        auto hessian_stoch = Hessian::compute(loss, x, HessianMethod::STOCHASTIC_DIAGONAL);
        hessian_stoch->computeStochasticDiagonal(loss, x, 50);
        double elapsed2 = timer2.elapsed();
        
        print_vector("Exact diagonal", hessian_diag->getData());
        print_vector("Stochastic diagonal (50 samples)", hessian_stoch->getData());
        std::cout << "Exact time: " << elapsed1 << " ms\n";
        std::cout << "Stochastic time: " << elapsed2 << " ms\n\n";
    }
    
    // Test 3: L-BFGS approximation
    std::cout << "3. L-BFGS APPROXIMATION - Optimization\n";
    std::cout << "---------------------------------------\n";
    {
        auto x = from_vector({-1.5f, 2.5f}, {2}, true);
        auto lbfgs_approx = std::make_unique<LBFGSApproximation>(2, 5);
        
        std::cout << "Optimizing Rosenbrock function with L-BFGS...\n\n";
        
        float learning_rate = 0.01f;
        for (int iter = 0; iter < 10; ++iter) {
            // Compute loss and gradient
            auto loss = rosenbrock(x);
            x->zero_grad();
            loss->backward();
            
            auto grad = zeros({2}, false);
            std::memcpy(grad->data, x->grad, 2 * sizeof(float));
            
            // Apply L-BFGS direction
            TensorPtr direction;
            if (iter > 0) {
                direction = lbfgs_approx->solve(grad);
            } else {
                direction = grad->clone();
            }
            
            // Store old values for update
            auto x_old = x->clone();
            auto g_old = grad->clone();
            
            // Take step
            for (size_t i = 0; i < 2; ++i) {
                x->data[i] -= learning_rate * direction->data[i];
            }
            
            // Compute new gradient
            loss = rosenbrock(x);
            x->zero_grad();
            loss->backward();
            auto g_new = zeros({2}, false);
            std::memcpy(g_new->data, x->grad, 2 * sizeof(float));
            
            // Update L-BFGS approximation
            lbfgs_approx->update(x_old, g_old, x, g_new);
            
            if (iter % 2 == 0) {
                std::cout << "Iter " << iter << ": x = [" 
                         << std::fixed << std::setprecision(4)
                         << x->data[0] << ", " << x->data[1] 
                         << "], f(x) = " << loss->data[0] 
                         << ", ||grad|| = " << std::sqrt(g_new->data[0]*g_new->data[0] + 
                                                         g_new->data[1]*g_new->data[1]) << "\n";
            }
        }
        std::cout << "\nOptimal point should be near [1, 1]\n\n";
    }
    
    // Test 4: BFGS approximation
    std::cout << "4. BFGS APPROXIMATION\n";
    std::cout << "---------------------\n";
    {
        auto x = from_vector({0.0f, 0.0f, 0.0f}, {3}, true);
        auto bfgs_approx = std::make_unique<BFGSApproximation>(3, true);
        
        // Initial gradient
        auto loss = quadratic(x);
        x->zero_grad();
        loss->backward();
        auto g_old = zeros({3}, false);
        std::memcpy(g_old->data, x->grad, 3 * sizeof(float));
        
        // Take a step
        auto x_old = x->clone();
        for (size_t i = 0; i < 3; ++i) {
            x->data[i] -= 0.1f * g_old->data[i];
        }
        
        // New gradient
        loss = quadratic(x);
        x->zero_grad();
        loss->backward();
        auto g_new = zeros({3}, false);
        std::memcpy(g_new->data, x->grad, 3 * sizeof(float));
        
        // Update BFGS
        bfgs_approx->update(x_old, g_old, x, g_new);
        
        // Test Hessian-vector product
        auto test_vec = from_vector({1.0f, 0.5f, -0.3f}, {3}, false);
        auto hv = bfgs_approx->matvec(test_vec);
        
        print_vector("Test vector", test_vec);
        print_vector("BFGS * vector", hv);
        
        // Test solve (inverse Hessian application)
        auto inv_hv = bfgs_approx->solve(test_vec);
        print_vector("BFGS^(-1) * vector", inv_hv);
    }
    
    // Test 5: Newton's method with different Hessian approximations
    std::cout << "5. NEWTON'S METHOD COMPARISON\n";
    std::cout << "------------------------------\n";
    {
        std::cout << "Minimizing quadratic function...\n\n";
        
        // Method 1: Exact Hessian Newton
        {
            auto x = from_vector({2.0f, -1.0f, 3.0f}, {3}, true);
            std::cout << "Exact Hessian Newton:\n";
            
            for (int iter = 0; iter < 3; ++iter) {
                auto loss = quadratic(x);
                
                // Compute gradient
                x->zero_grad();
                loss->backward();
                auto grad = zeros({3}, false);
                std::memcpy(grad->data, x->grad, 3 * sizeof(float));
                
                // Compute Hessian
                auto hessian = Hessian::compute(loss, x, HessianMethod::EXACT);
                
                // Add regularization for stability
                hessian->addDiagonal(0.01f);
                
                // Newton direction: d = -H^(-1) * g
                // For now, use diagonal approximation for simplicity
                auto diag = hessian->diagonal();
                for (size_t i = 0; i < 3; ++i) {
                    x->data[i] -= grad->data[i] / (diag->data[i] + 0.01f);
                }
                
                loss = quadratic(x);
                std::cout << "  Iter " << iter << ": f(x) = " << loss->data[0] << "\n";
            }
            print_vector("  Final x", x);
        }
        
        // Method 2: Diagonal Hessian Newton
        {
            auto x = from_vector({2.0f, -1.0f, 3.0f}, {3}, true);
            std::cout << "\nDiagonal Hessian Newton:\n";
            
            for (int iter = 0; iter < 3; ++iter) {
                auto loss = quadratic(x);
                newton_step_diagonal(x, loss, 0.01f);
                
                loss = quadratic(x);
                std::cout << "  Iter " << iter << ": f(x) = " << loss->data[0] << "\n";
            }
            print_vector("  Final x", x);
        }
    }
    
    // Test 6: Condition number and positive definiteness
    std::cout << "6. HESSIAN PROPERTIES\n";
    std::cout << "---------------------\n";
    {
        // Test at minimum of Rosenbrock (should be positive definite)
        auto x_min = from_vector({1.0f, 1.0f}, {2}, true);
        auto loss_min = rosenbrock(x_min);
        auto hessian_min = Hessian::compute(loss_min, x_min, HessianMethod::EXACT);
        
        std::cout << "At minimum [1, 1]:\n";
        print_matrix("Hessian", hessian_min->getData(), 2, 2);
        std::cout << "Positive definite: " << (hessian_min->isPositiveDefinite() ? "Yes" : "No") << "\n\n";
        
        // Test at saddle point
        auto x_saddle = from_vector({0.0f, 0.0f}, {2}, true);
        auto loss_saddle = rosenbrock(x_saddle);
        auto hessian_saddle = Hessian::compute(loss_saddle, x_saddle, HessianMethod::EXACT);
        
        std::cout << "At point [0, 0]:\n";
        print_matrix("Hessian", hessian_saddle->getData(), 2, 2);
        std::cout << "Positive definite: " << (hessian_saddle->isPositiveDefinite() ? "Yes" : "No") << "\n\n";
    }
    
    // Test 7: Gauss-Newton for least squares
    std::cout << "7. GAUSS-NEWTON APPROXIMATION\n";
    std::cout << "------------------------------\n";
    {
        // Simple linear least squares: ||Ax - b||^2
        auto A = from_vector({2.0f, 1.0f, 1.0f, 3.0f, 1.5f, 0.5f}, {3, 2}, false);
        auto b = from_vector({1.0f, 2.0f, 1.5f}, {3}, false);
        auto x = from_vector({0.0f, 0.0f}, {2}, true);
        
        auto residual_fn = [&](TensorPtr params) -> TensorPtr {
            return sub(matmul(A, params), b);
        };
        
        auto loss_fn = [&](TensorPtr params) -> TensorPtr {
            auto residual = residual_fn(params);
            auto half = from_vector({0.5f}, {1}, false);
            return mul(half, sum(mul(residual, residual)));
        };
        
        // Compute Gauss-Newton Hessian diagonal
        auto gn_diag = gauss_newton_hessian_diag(residual_fn, x);
        
        // Compute exact Hessian for comparison
        auto loss = loss_fn(x);
        auto exact_hessian = Hessian::compute(loss, x, HessianMethod::EXACT);
        auto exact_diag = exact_hessian->diagonal();
        
        std::cout << "For least squares problem ||Ax - b||^2:\n";
        print_vector("Gauss-Newton diagonal", gn_diag);
        print_vector("Exact Hessian diagonal", exact_diag);
        std::cout << "Note: Gauss-Newton ignores second-order terms\n\n";
    }
    
    // Test 8: Performance comparison
    std::cout << "8. PERFORMANCE COMPARISON\n";
    std::cout << "-------------------------\n";
    {
        size_t dim = 20;
        auto x = randn({dim}, true);
        
        // Create a random quadratic function
        auto loss_fn = [dim](TensorPtr params) -> TensorPtr {
            auto result = from_vector({0.0f}, {1}, false);
            for (size_t i = 0; i < dim; ++i) {
                auto xi = from_vector({params->data[i]}, {1}, true);
                auto coeff = from_vector({float(i + 1)}, {1}, false);
                result = add(result, mul(coeff, mul(xi, xi)));
                if (i > 0) {
                    auto xi_prev = from_vector({params->data[i - 1]}, {1}, true);
                    auto half = from_vector({0.5f}, {1}, false);
                    result = add(result, mul(half, mul(xi, xi_prev)));
                }
            }
            return result;
        };
        
        auto loss = loss_fn(x);
        
        std::cout << "Dimension: " << dim << "\n\n";
        
        // Exact Hessian
        {
            Timer timer;
            auto hessian = Hessian::compute(loss, x, HessianMethod::EXACT);
            double elapsed = timer.elapsed();
            std::cout << "Exact Hessian: " << elapsed << " ms\n";
        }
        
        // Diagonal only
        {
            Timer timer;
            auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);
            double elapsed = timer.elapsed();
            std::cout << "Diagonal only: " << elapsed << " ms\n";
        }
        
        // Stochastic diagonal
        {
            Timer timer;
            auto hessian = Hessian::compute(loss, x, HessianMethod::STOCHASTIC_DIAGONAL);
            hessian->computeStochasticDiagonal(loss, x, 30);
            double elapsed = timer.elapsed();
            std::cout << "Stochastic diagonal (30 samples): " << elapsed << " ms\n";
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "        Demonstration Complete          \n";
    std::cout << "========================================\n";
    
    return 0;
}