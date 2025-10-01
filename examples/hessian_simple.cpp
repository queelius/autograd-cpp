// Simple demonstration of Hessian functionality
#include "../include/autograd/autograd.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

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
    
    // Test 1: Simple quadratic function
    std::cout << "1. EXACT HESSIAN - Simple Quadratic\n";
    std::cout << "-----------------------------------\n";
    std::cout << "f(x,y) = x^2 + 2*y^2 + x*y\n\n";
    {
        auto params = from_vector({1.0f, 2.0f}, {2}, true);
        
        // Define function inline for simplicity
        auto x = from_vector({params->data[0]}, {1}, true);
        auto y = from_vector({params->data[1]}, {1}, true);
        
        // f = x^2 + 2*y^2 + x*y
        auto loss = add(add(mul(x, x), mul(from_vector({2.0f}, {1}, false), mul(y, y))), 
                       mul(x, y));
        
        std::cout << "Point: [" << params->data[0] << ", " << params->data[1] << "]\n";
        std::cout << "Function value: " << loss->data[0] << "\n";
        
        Timer timer;
        auto hessian = Hessian::compute(loss, params, HessianMethod::EXACT);
        double elapsed = timer.elapsed();
        
        print_matrix("Hessian", hessian->getData(), 2, 2);
        std::cout << "Expected Hessian: [[2, 1], [1, 4]]\n";
        std::cout << "Trace: " << hessian->trace() << " (expected: 6)\n";
        std::cout << "Time: " << elapsed << " ms\n\n";
    }
    
    // Test 2: Diagonal Hessian approximation
    std::cout << "2. DIAGONAL HESSIAN METHODS\n";
    std::cout << "---------------------------\n";
    {
        auto x = from_vector({1.0f, -0.5f, 2.0f}, {3}, true);
        
        // f = 2*x0^2 + 3*x1^2 + 4*x2^2
        auto x0 = from_vector({x->data[0]}, {1}, true);
        auto x1 = from_vector({x->data[1]}, {1}, true);
        auto x2 = from_vector({x->data[2]}, {1}, true);
        
        auto loss = add(add(
            mul(from_vector({2.0f}, {1}, false), mul(x0, x0)),
            mul(from_vector({3.0f}, {1}, false), mul(x1, x1))),
            mul(from_vector({4.0f}, {1}, false), mul(x2, x2)));
        
        // Exact diagonal
        Timer timer1;
        auto hessian_diag = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);
        double elapsed1 = timer1.elapsed();
        
        // Stochastic diagonal
        Timer timer2;
        auto hessian_stoch = Hessian::compute(loss, x, HessianMethod::STOCHASTIC_DIAGONAL);
        hessian_stoch->computeStochasticDiagonal(loss, x, 50);
        double elapsed2 = timer2.elapsed();
        
        std::cout << "Function: f = 2*x0^2 + 3*x1^2 + 4*x2^2\n";
        std::cout << "Expected diagonal: [4, 6, 8]\n\n";
        print_vector("Exact diagonal", hessian_diag->getData());
        print_vector("Stochastic diagonal", hessian_stoch->getData());
        std::cout << "Exact time: " << elapsed1 << " ms\n";
        std::cout << "Stochastic time: " << elapsed2 << " ms\n\n";
    }
    
    // Test 3: L-BFGS approximation
    std::cout << "3. L-BFGS APPROXIMATION\n";
    std::cout << "-----------------------\n";
    {
        auto x = from_vector({2.0f, 1.0f}, {2}, true);
        auto lbfgs_approx = std::make_unique<LBFGSApproximation>(2, 5);
        
        std::cout << "Optimizing f(x,y) = (x-1)^2 + (y-2)^2\n\n";
        
        float learning_rate = 0.1f;
        for (int iter = 0; iter < 5; ++iter) {
            // Compute loss: f = (x-1)^2 + (y-2)^2
            auto x_val = from_vector({x->data[0]}, {1}, true);
            auto y_val = from_vector({x->data[1]}, {1}, true);
            auto one = from_vector({1.0f}, {1}, false);
            auto two = from_vector({2.0f}, {1}, false);
            
            auto loss = add(mul(sub(x_val, one), sub(x_val, one)),
                          mul(sub(y_val, two), sub(y_val, two)));
            
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
            auto x_val_new = from_vector({x->data[0]}, {1}, true);
            auto y_val_new = from_vector({x->data[1]}, {1}, true);
            auto loss_new = add(mul(sub(x_val_new, one), sub(x_val_new, one)),
                               mul(sub(y_val_new, two), sub(y_val_new, two)));
            x->zero_grad();
            loss_new->backward();
            auto g_new = zeros({2}, false);
            std::memcpy(g_new->data, x->grad, 2 * sizeof(float));
            
            // Update L-BFGS approximation
            lbfgs_approx->update(x_old, g_old, x, g_new);
            
            std::cout << "Iter " << iter << ": x = [" 
                     << std::fixed << std::setprecision(4)
                     << x->data[0] << ", " << x->data[1] 
                     << "], f(x) = " << loss_new->data[0] 
                     << ", ||grad|| = " << std::sqrt(g_new->data[0]*g_new->data[0] + 
                                                     g_new->data[1]*g_new->data[1]) << "\n";
        }
        std::cout << "\nOptimal point should be [1, 2]\n\n";
    }
    
    // Test 4: BFGS approximation
    std::cout << "4. BFGS APPROXIMATION\n";
    std::cout << "--------------------\n";
    {
        auto x = from_vector({0.0f, 0.0f}, {2}, true);
        auto bfgs_approx = std::make_unique<BFGSApproximation>(2, true);
        
        // Initial gradient for f = x^2 + y^2
        auto x_val = from_vector({x->data[0]}, {1}, true);
        auto y_val = from_vector({x->data[1]}, {1}, true);
        auto loss = add(mul(x_val, x_val), mul(y_val, y_val));
        
        x->zero_grad();
        loss->backward();
        auto g_old = zeros({2}, false);
        std::memcpy(g_old->data, x->grad, 2 * sizeof(float));
        
        // Take a step
        auto x_old = x->clone();
        x->data[0] = 1.0f;
        x->data[1] = 1.0f;
        
        // New gradient
        x_val = from_vector({x->data[0]}, {1}, true);
        y_val = from_vector({x->data[1]}, {1}, true);
        loss = add(mul(x_val, x_val), mul(y_val, y_val));
        x->zero_grad();
        loss->backward();
        auto g_new = zeros({2}, false);
        std::memcpy(g_new->data, x->grad, 2 * sizeof(float));
        
        // Update BFGS
        bfgs_approx->update(x_old, g_old, x, g_new);
        
        // Test Hessian-vector product
        auto test_vec = from_vector({1.0f, 0.5f}, {2}, false);
        auto hv = bfgs_approx->matvec(test_vec);
        
        std::cout << "After one BFGS update:\n";
        print_vector("Test vector", test_vec);
        print_vector("BFGS * vector", hv);
        
        // Test solve (inverse Hessian application)
        auto inv_hv = bfgs_approx->solve(test_vec);
        print_vector("BFGS^(-1) * vector", inv_hv);
    }
    
    // Test 5: Newton's method with diagonal Hessian
    std::cout << "5. NEWTON'S METHOD WITH DIAGONAL HESSIAN\n";
    std::cout << "----------------------------------------\n";
    {
        auto x = from_vector({3.0f, -2.0f}, {2}, true);
        std::cout << "Minimizing f(x,y) = x^2 + 2*y^2\n\n";
        
        for (int iter = 0; iter < 3; ++iter) {
            auto x_val = from_vector({x->data[0]}, {1}, true);
            auto y_val = from_vector({x->data[1]}, {1}, true);
            auto loss = add(mul(x_val, x_val), 
                          mul(from_vector({2.0f}, {1}, false), mul(y_val, y_val)));
            
            newton_step_diagonal(x, loss, 0.01f);
            
            // Recompute loss for display
            x_val = from_vector({x->data[0]}, {1}, true);
            y_val = from_vector({x->data[1]}, {1}, true);
            loss = add(mul(x_val, x_val), 
                      mul(from_vector({2.0f}, {1}, false), mul(y_val, y_val)));
            
            std::cout << "Iter " << iter << ": x = [" << x->data[0] << ", " << x->data[1] 
                     << "], f(x) = " << loss->data[0] << "\n";
        }
        std::cout << "\nOptimal point should be [0, 0]\n\n";
    }
    
    // Test 6: Performance comparison
    std::cout << "6. PERFORMANCE COMPARISON\n";
    std::cout << "------------------------\n";
    {
        size_t dim = 10;
        auto x = randn({dim}, true);
        
        // Create a simple diagonal quadratic
        auto loss = from_vector({0.0f}, {1}, false);
        for (size_t i = 0; i < dim; ++i) {
            auto xi = from_vector({x->data[i]}, {1}, true);
            auto coeff = from_vector({float(i + 1)}, {1}, false);
            loss = add(loss, mul(coeff, mul(xi, xi)));
        }
        
        std::cout << "Dimension: " << dim << "\n\n";
        
        // Exact Hessian (would be slow for large dim)
        if (dim <= 10) {
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