#pragma once

#include "../base/regression.hpp"
#include "../../autograd/autograd.hpp"
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace statmodels {

using namespace autograd;

// Logistic Regression Model
// p(y=1|X) = sigmoid(X @ beta + intercept)
class LogisticRegression : public RegressionModel {
private:
    TensorPtr beta_;      // Coefficients
    TensorPtr intercept_; // Intercept term
    bool fit_intercept_;  // Whether to fit an intercept
    
public:
    LogisticRegression(bool fit_intercept = true) 
        : fit_intercept_(fit_intercept) {}
    
    // Initialize parameters based on input dimensions
    void initialize_parameters(size_t n_features) {
        // Initialize with small random values (Xavier initialization)
        float scale = std::sqrt(2.0f / n_features);
        std::vector<float> beta_init(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            beta_init[i] = scale * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        }
        beta_ = from_vector(beta_init, {static_cast<int>(n_features), 1}, true);
        
        if (fit_intercept_) {
            intercept_ = from_vector({0.0f}, {1}, true);
        }
    }
    
    // Compute logits (linear combination)
    TensorPtr compute_logits(TensorPtr X) {
        if (!beta_) {
            throw std::runtime_error("Model not fitted. Call fit() first.");
        }
        
        // logits = X @ beta
        auto logits = matmul(X, beta_);
        
        // Add intercept if fitted
        if (fit_intercept_ && intercept_) {
            logits = add(logits, intercept_);
        }
        
        return logits;
    }
    
    // Predict probabilities
    TensorPtr predict_proba(TensorPtr X) {
        auto logits = compute_logits(X);
        return sigmoid(logits);
    }
    
    // Predict binary classes (threshold at 0.5)
    TensorPtr predict(TensorPtr X) override {
        auto proba = predict_proba(X);
        
        // Apply threshold of 0.5
        std::vector<float> predictions;
        for (size_t i = 0; i < proba->size(); ++i) {
            predictions.push_back(proba->data[i] >= 0.5f ? 1.0f : 0.0f);
        }
        
        return from_vector(predictions, proba->shape, false);
    }
    
    // Binary cross-entropy loss
    TensorPtr binary_cross_entropy_loss(TensorPtr y_true, TensorPtr y_pred_proba) {
        // Loss = -mean(y * log(p) + (1-y) * log(1-p))
        // Add small epsilon to avoid log(0)
        float epsilon = 1e-7f;
        auto eps_tensor = from_vector({epsilon}, {1}, false);
        auto one = from_vector({1.0f}, {1}, false);
        
        // Clip probabilities to avoid numerical issues
        auto p_clipped = maximum(minimum(y_pred_proba, sub(one, eps_tensor)), eps_tensor);
        
        // Compute loss terms
        auto term1 = mul(y_true, log(p_clipped));
        auto term2 = mul(sub(one, y_true), log(sub(one, p_clipped)));
        
        // Negative mean
        auto loss = neg(mean(add(term1, term2)));
        
        return loss;
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
            auto y_pred_proba = predict_proba(X);
            
            // Compute loss (binary cross-entropy)
            auto loss = binary_cross_entropy_loss(y, y_pred_proba);
            
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
    
    // Override score to use accuracy for classification
    float score(TensorPtr X, TensorPtr y) override {
        auto predictions = predict(X);
        
        // Calculate accuracy
        int correct = 0;
        for (size_t i = 0; i < predictions->size(); ++i) {
            if (std::abs(predictions->data[i] - y->data[i]) < 0.5f) {
                correct++;
            }
        }
        
        return static_cast<float>(correct) / predictions->size();
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
        return "LogisticRegression";
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
    
    // Print model summary
    void summary() const {
        std::cout << "\n=== Logistic Regression Model ===" << std::endl;
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
    
    // Get decision boundary equation (for 2D problems)
    std::string get_decision_boundary() const {
        if (!beta_ || beta_->size() != 2) {
            return "Decision boundary only available for 2D problems";
        }
        
        float w1 = beta_->data[0];
        float w2 = beta_->data[1];
        float b = fit_intercept_ && intercept_ ? intercept_->data[0] : 0.0f;
        
        // w1*x1 + w2*x2 + b = 0
        // x2 = -(w1*x1 + b) / w2
        
        std::stringstream ss;
        ss << std::fixed << std::setprecision(3);
        ss << "x2 = " << (-w1/w2) << " * x1";
        if (b != 0) {
            ss << " + " << (-b/w2);
        }
        
        return ss.str();
    }
};

} // namespace statmodels