#pragma once

#include "../base/regression.hpp"
#include "../../autograd/autograd.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

namespace statmodels {

using namespace autograd;

// Linear Regression Model
// y = X @ beta + intercept + epsilon
class LinearRegression : public RegressionModel {
private:
    TensorPtr beta_;      // Coefficients
    TensorPtr intercept_; // Intercept term
    bool fit_intercept_;  // Whether to fit an intercept
    
public:
    LinearRegression(bool fit_intercept = true) 
        : fit_intercept_(fit_intercept) {}
    
    // Initialize parameters based on input dimensions
    void initialize_parameters(size_t n_features) {
        // Initialize with small random values
        std::vector<float> beta_init(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            beta_init[i] = 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        }
        beta_ = from_vector(beta_init, {static_cast<int>(n_features), 1}, true);
        
        if (fit_intercept_) {
            intercept_ = from_vector({0.0f}, {1}, true);
        }
    }
    
    // Predict output for given input
    TensorPtr predict(TensorPtr X) override {
        if (!beta_) {
            throw std::runtime_error("Model not fitted. Call fit() first.");
        }
        
        // y = X @ beta
        auto y_pred = matmul(X, beta_);
        
        // Add intercept if fitted
        if (fit_intercept_ && intercept_) {
            y_pred = add(y_pred, intercept_);
        }
        
        return y_pred;
    }
    
    // Fit the model using gradient descent
    void fit(TensorPtr X, TensorPtr y,
            int max_iterations = 1000,
            float learning_rate = 0.01f,
            float tolerance = 1e-5,
            bool verbose = false) override {
        
        // Get dimensions
        size_t n_samples = X->shape[0];
        size_t n_features = X->shape[1];
        
        // Initialize parameters if not already done
        if (!beta_) {
            initialize_parameters(n_features);
        }
        
        // Training loop
        float prev_loss = std::numeric_limits<float>::max();
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Forward pass
            auto y_pred = predict(X);
            
            // Compute loss (MSE)
            auto loss = mse_loss(y, y_pred);
            
            // Add regularization if specified
            if (regularization_alpha_ > 0) {
                std::vector<TensorPtr> params = {beta_};
                loss = add_regularization(loss, params);
            }
            
            // Backward pass
            loss->backward();
            
            float loss_value = loss->data[0];
            float grad_norm = compute_gradient_norm();
            
            // Check convergence
            if (std::abs(loss_value - prev_loss) < tolerance) {
                if (verbose) {
                    std::cout << "\nConverged at iteration " << iter << std::endl;
                    std::cout << "Final loss: " << loss_value << std::endl;
                }
                break;
            }
            
            // Gradient descent step
            apply_gradient_step(learning_rate);
            
            // Clear gradients
            zero_grad();
            
            // Update previous loss
            prev_loss = loss_value;
            
            // Print progress
            if (verbose && (iter % 100 == 0 || iter == max_iterations - 1)) {
                print_iteration_info(iter, loss_value, grad_norm, learning_rate);
            }
        }
    }
    
    // Get model parameters
    std::vector<TensorPtr> get_parameters() override {
        std::vector<TensorPtr> params;
        if (beta_) params.push_back(beta_);
        if (fit_intercept_ && intercept_) params.push_back(intercept_);
        return params;
    }
    
    // Zero gradients
    void zero_grad() override {
        if (beta_) beta_->zero_grad();
        if (intercept_) intercept_->zero_grad();
    }
    
    // Get model name
    std::string get_name() const override {
        return "LinearRegression";
    }
    
    // Get coefficients
    TensorPtr get_coefficients() override {
        return beta_;
    }
    
    // Get intercept
    TensorPtr get_intercept() override {
        if (!fit_intercept_) {
            return from_vector({0.0f}, {1}, false);
        }
        return intercept_;
    }
    
    // Analytical solution using normal equations (for comparison)
    void fit_normal_equation(TensorPtr X, TensorPtr y) {
        size_t n_samples = X->shape[0];
        size_t n_features = X->shape[1];
        
        TensorPtr X_with_intercept = X;
        
        // Add intercept column if needed
        if (fit_intercept_) {
            // Create ones column
            std::vector<float> ones(n_samples, 1.0f);
            auto ones_tensor = from_vector(ones, {static_cast<int>(n_samples), 1}, false);
            
            // Concatenate X with ones column
            // This would require a concatenate function in autograd
            // For now, we'll use gradient descent
            throw std::runtime_error("Normal equation not fully implemented - use gradient descent");
        }
        
        // beta = (X^T @ X)^(-1) @ X^T @ y
        // This would require matrix inverse which may not be available
    }
    
    // Print model summary
    void summary() const {
        std::cout << "\n=== Linear Regression Model ===" << std::endl;
        if (beta_) {
            std::cout << "Coefficients shape: [" << beta_->shape[0] << ", " << beta_->shape[1] << "]" << std::endl;
            std::cout << "Coefficients: ";
            for (size_t i = 0; i < beta_->size(); ++i) {
                std::cout << std::fixed << std::setprecision(4) << beta_->data[i];
                if (i < beta_->size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        if (fit_intercept_ && intercept_) {
            std::cout << "Intercept: " << std::fixed << std::setprecision(4) << intercept_->data[0] << std::endl;
        }
        if (regularization_alpha_ > 0) {
            std::cout << "Regularization: " << regularization_penalty_ 
                     << " (alpha=" << regularization_alpha_ << ")" << std::endl;
        }
    }
};

} // namespace statmodels