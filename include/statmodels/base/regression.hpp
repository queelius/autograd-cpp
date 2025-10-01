#pragma once

#include "../../autograd/autograd.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace statmodels {

using namespace autograd;

// Base class for regression models
class RegressionModel {
public:
    virtual ~RegressionModel() = default;
    
    // Core interface for regression models
    virtual TensorPtr predict(TensorPtr X) = 0;
    
    // Fit the model to data
    virtual void fit(TensorPtr X, TensorPtr y, 
                    int max_iterations = 1000,
                    float learning_rate = 0.01f,
                    float tolerance = 1e-5,
                    bool verbose = false) = 0;
    
    // Score the model (default: R-squared for regression)
    virtual float score(TensorPtr X, TensorPtr y) {
        auto predictions = predict(X);
        return r_squared(y, predictions);
    }
    
    // Get model parameters
    virtual std::vector<TensorPtr> get_parameters() = 0;
    virtual void zero_grad() = 0;
    virtual std::string get_name() const = 0;
    
    // Parameter value access
    virtual std::vector<float> get_parameter_values() {
        std::vector<float> values;
        for (auto& param : get_parameters()) {
            for (size_t i = 0; i < param->size(); ++i) {
                values.push_back(param->data[i]);
            }
        }
        return values;
    }
    
    virtual void set_parameter_values(const std::vector<float>& values) {
        size_t idx = 0;
        for (auto& param : get_parameters()) {
            for (size_t i = 0; i < param->size(); ++i) {
                param->data[i] = values[idx++];
            }
        }
    }
    
    // Get coefficients and intercept (for linear models)
    virtual TensorPtr get_coefficients() {
        throw std::runtime_error("get_coefficients not implemented for this model");
    }
    
    virtual TensorPtr get_intercept() {
        throw std::runtime_error("get_intercept not implemented for this model");
    }
    
    // Regularization parameters
    virtual void set_regularization(float alpha, const std::string& penalty = "l2") {
        regularization_alpha_ = alpha;
        regularization_penalty_ = penalty;
    }
    
protected:
    float regularization_alpha_ = 0.0f;
    std::string regularization_penalty_ = "none";
    
    // Common loss functions
    virtual TensorPtr mse_loss(TensorPtr y_true, TensorPtr y_pred) {
        auto diff = sub(y_true, y_pred);
        auto squared = mul(diff, diff);
        return mean(squared);
    }
    
    virtual TensorPtr mae_loss(TensorPtr y_true, TensorPtr y_pred) {
        auto diff = sub(y_true, y_pred);
        auto abs_diff = abs(diff);
        return mean(abs_diff);
    }
    
    // Add regularization term to loss
    virtual TensorPtr add_regularization(TensorPtr loss, const std::vector<TensorPtr>& params) {
        if (regularization_alpha_ <= 0.0f || regularization_penalty_ == "none") {
            return loss;
        }
        
        auto reg_term = zeros({1}, false);
        
        for (const auto& param : params) {
            if (regularization_penalty_ == "l2") {
                // L2 regularization: alpha * sum(param^2)
                auto squared = mul(param, param);
                reg_term = add(reg_term, sum(squared));
            } else if (regularization_penalty_ == "l1") {
                // L1 regularization: alpha * sum(|param|)
                auto abs_param = abs(param);
                reg_term = add(reg_term, sum(abs_param));
            }
        }
        
        auto alpha_tensor = from_vector({regularization_alpha_}, {1}, false);
        reg_term = mul(alpha_tensor, reg_term);
        
        return add(loss, reg_term);
    }
    
    // Model evaluation metrics
    float r_squared(TensorPtr y_true, TensorPtr y_pred) {
        // R² = 1 - (SS_res / SS_tot)
        auto y_mean = mean(y_true);
        
        // SS_res = sum((y_true - y_pred)^2)
        auto residuals = sub(y_true, y_pred);
        auto ss_res = sum(mul(residuals, residuals));
        
        // SS_tot = sum((y_true - y_mean)^2)
        auto total = sub(y_true, y_mean);
        auto ss_tot = sum(mul(total, total));
        
        // Avoid division by zero
        if (ss_tot->data[0] == 0.0f) {
            return 0.0f;
        }
        
        return 1.0f - (ss_res->data[0] / ss_tot->data[0]);
    }
    
    float mean_squared_error(TensorPtr y_true, TensorPtr y_pred) {
        auto loss = mse_loss(y_true, y_pred);
        return loss->data[0];
    }
    
    float mean_absolute_error(TensorPtr y_true, TensorPtr y_pred) {
        auto loss = mae_loss(y_true, y_pred);
        return loss->data[0];
    }
    
    // Gradient computation helpers
    float compute_gradient_norm() {
        float norm = 0.0f;
        for (auto& param : get_parameters()) {
            for (size_t i = 0; i < param->size(); ++i) {
                norm += param->grad[i] * param->grad[i];
            }
        }
        return std::sqrt(norm);
    }
    
    void apply_gradient_step(float learning_rate) {
        for (auto& param : get_parameters()) {
            for (size_t i = 0; i < param->size(); ++i) {
                param->data[i] -= learning_rate * param->grad[i];
            }
        }
    }
    
    // Print utilities
    void print_iteration_info(int iter, float loss, float grad_norm, float learning_rate) {
        std::cout << "Iteration " << iter 
                  << ": loss = " << std::fixed << std::setprecision(6) << loss
                  << ", |grad| = " << std::scientific << std::setprecision(4) << grad_norm
                  << ", lr = " << std::fixed << std::setprecision(6) << learning_rate
                  << std::endl;
    }
};

} // namespace statmodels