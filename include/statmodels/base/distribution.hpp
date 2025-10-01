#pragma once

#include "../../autograd/autograd.hpp"
#include "../optimization/optimizer.hpp"
#include <vector>
#include <functional>
#include <iostream>
#include <limits>
#include <cmath>
#include <string>
#include <iomanip>
#include <algorithm>

namespace statmodels {

using namespace autograd;

// Common convergence criteria for optimization
enum class ConvergenceCriterion {
    GRADIENT,       // |gradient| < tolerance
    PARAMETER,      // |Δparameter| < tolerance  
    LOSS,          // |ΔNLL| < tolerance
    TARGET         // |parameter - target| < tolerance (if available)
};

// Base class for probability distributions
class Distribution {
public:
    virtual ~Distribution() = default;
    
    // Core interface that derived classes must implement
    virtual TensorPtr log_pdf(TensorPtr x) = 0;
    virtual TensorPtr pdf(TensorPtr x) {
        return exp(log_pdf(x));
    }
    
    // CDF and quantile functions (optional to implement)
    virtual TensorPtr cdf(TensorPtr x) {
        throw std::runtime_error("CDF not implemented for this distribution");
    }
    virtual TensorPtr quantile(TensorPtr p) {
        throw std::runtime_error("Quantile function not implemented for this distribution");
    }
    
    // Sampling interface
    virtual std::vector<float> sample(size_t n_samples, uint32_t seed = 42) = 0;
    
    // Parameter access
    virtual std::vector<TensorPtr> get_parameters() = 0;
    virtual void zero_grad() = 0;
    virtual std::string get_name() const = 0;
    
    // Default NLL implementation using log_pdf
    virtual TensorPtr negative_log_likelihood(const std::vector<float>& data) {
        auto nll = zeros({1}, false);
        for (float x_val : data) {
            auto x = from_vector({x_val}, {1}, false);
            auto log_p = log_pdf(x);
            nll = sub(nll, log_p);
        }
        return nll;
    }
    
    // Parameter value access (for bootstrap and diagnostics)
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
    
    // Statistical properties
    virtual float mean() const {
        throw std::runtime_error("Mean not implemented for this distribution");
    }
    virtual float variance() const {
        throw std::runtime_error("Variance not implemented for this distribution");
    }
    virtual float std_dev() const {
        return std::sqrt(variance());
    }
    
    // Fitting methods
    int fit_mle(const std::vector<float>& data,
                float tolerance = 1e-5,
                float initial_lr = 0.01f,
                int max_iterations = 10000,
                ConvergenceCriterion criterion = ConvergenceCriterion::GRADIENT,
                bool verbose = false) {
        return fit_adaptive(data, tolerance, initial_lr, max_iterations, criterion, verbose);
    }
    
    // Generic adaptive optimizer with convergence criteria
    int fit_adaptive(const std::vector<float>& data,
                     float tolerance = 1e-5,
                     float initial_lr = 0.01f,
                     int max_iterations = 10000,
                     ConvergenceCriterion criterion = ConvergenceCriterion::GRADIENT,
                     bool verbose = false) {
        
        float learning_rate = initial_lr;
        float prev_nll = std::numeric_limits<float>::max();
        std::vector<float> prev_params = get_parameter_values();
        int patience = 0;
        const int max_patience = 20;
        int stall_count = 0;
        const int max_stall = 50;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            auto nll = negative_log_likelihood(data);
            float nll_value = nll->data[0];
            
            nll->backward();
            float grad_norm = compute_gradient_norm();
            
            std::vector<float> current_params = get_parameter_values();
            
            bool converged = check_convergence(criterion, tolerance, iter, 
                                              grad_norm, current_params, prev_params,
                                              nll_value, prev_nll, verbose);
            
            if (converged) {
                if (verbose) {
                    print_convergence_info(iter, grad_norm, nll_value);
                }
                return iter;
            }
            
            if (std::isnan(nll_value) || std::isinf(nll_value)) {
                handle_numerical_issues(learning_rate, prev_params, verbose, iter);
                continue;
            }
            
            bool params_updated = adaptive_learning_rate_update(
                nll_value, prev_nll, learning_rate, initial_lr, 
                patience, max_patience, stall_count, max_stall,
                current_params, prev_params
            );
            
            if (params_updated) {
                apply_gradient_step(learning_rate);
                prev_params = current_params;
                prev_nll = nll_value;
            }
            
            zero_grad();
            
            if (verbose && should_print(iter)) {
                print_iteration_info(iter, grad_norm, nll_value, learning_rate, criterion);
            }
        }
        
        if (verbose) {
            print_max_iterations_reached();
        }
        
        return max_iterations;
    }
    
protected:
    virtual float compute_gradient_norm() {
        float norm = 0.0f;
        for (auto& param : get_parameters()) {
            for (size_t i = 0; i < param->size(); ++i) {
                norm += param->grad[i] * param->grad[i];
            }
        }
        return std::sqrt(norm);
    }
    
    virtual void apply_gradient_step(float learning_rate) {
        const float max_step = 0.1f;
        for (auto& param : get_parameters()) {
            for (size_t i = 0; i < param->size(); ++i) {
                float step = learning_rate * param->grad[i];
                step = std::max(-max_step, std::min(max_step, step));
                param->data[i] -= step;
            }
        }
    }
    
    virtual void print_parameter_values() {
        auto values = get_parameter_values();
        for (size_t i = 0; i < values.size(); ++i) {
            std::cout << "θ_" << i << " = " << values[i];
            if (i < values.size() - 1) std::cout << ", ";
        }
    }
    
private:
    bool check_convergence(ConvergenceCriterion criterion, float tolerance, int iter,
                          float grad_norm, const std::vector<float>& current_params,
                          const std::vector<float>& prev_params, float nll_value, float prev_nll,
                          bool verbose) {
        
        bool converged = false;
        std::string convergence_reason;
        
        switch (criterion) {
            case ConvergenceCriterion::GRADIENT: {
                if (grad_norm < tolerance) {
                    converged = true;
                    convergence_reason = "gradient norm < " + std::to_string(tolerance);
                }
                break;
            }
            case ConvergenceCriterion::PARAMETER: {
                float param_change = compute_param_change(current_params, prev_params);
                if (iter > 0 && param_change < tolerance) {
                    converged = true;
                    convergence_reason = "parameter change < " + std::to_string(tolerance);
                }
                break;
            }
            case ConvergenceCriterion::LOSS: {
                float loss_change = std::abs(nll_value - prev_nll);
                if (iter > 0 && loss_change < tolerance) {
                    converged = true;
                    convergence_reason = "loss change < " + std::to_string(tolerance);
                }
                break;
            }
            case ConvergenceCriterion::TARGET: {
                break;
            }
        }
        
        if (converged && verbose) {
            std::cout << "\nConverged! Reason: " << convergence_reason << std::endl;
        }
        
        return converged;
    }
    
    float compute_param_change(const std::vector<float>& current, const std::vector<float>& prev) {
        float change = 0.0f;
        for (size_t i = 0; i < current.size(); ++i) {
            float diff = current[i] - prev[i];
            change += diff * diff;
        }
        return std::sqrt(change);
    }
    
    void handle_numerical_issues(float& learning_rate, const std::vector<float>& prev_params,
                                bool verbose, int iter) {
        if (verbose) {
            std::cout << "Warning: NaN or Inf detected at iteration " << iter << std::endl;
        }
        set_parameter_values(prev_params);
        learning_rate *= 0.1f;
        zero_grad();
    }
    
    bool adaptive_learning_rate_update(float nll_value, float prev_nll, float& learning_rate,
                                      float initial_lr, int& patience, int max_patience,
                                      int& stall_count, int max_stall,
                                      const std::vector<float>& current_params,
                                      const std::vector<float>& prev_params) {
        
        if (nll_value > prev_nll + 1e-8f) {
            learning_rate *= 0.7f;
            set_parameter_values(prev_params);
            patience++;
            
            if (patience > max_patience) {
                learning_rate = initial_lr * 0.01f;
                patience = 0;
            }
            return false;
        } else if (std::abs(nll_value - prev_nll) < 1e-10f) {
            stall_count++;
            if (stall_count > max_stall) {
                learning_rate *= 2.0f;
                learning_rate = std::min(learning_rate, initial_lr);
                stall_count = 0;
            }
            return true;
        } else {
            stall_count = 0;
            patience = 0;
            return true;
        }
    }
    
    bool should_print(int iter) {
        return iter < 20 || iter % 100 == 0;
    }
    
    void print_convergence_info(int iter, float grad_norm, float nll_value) {
        std::cout << "\nConverged at iteration " << iter << std::endl;
        std::cout << "Model: " << get_name() << std::endl;
        std::cout << "Final parameters: ";
        print_parameter_values();
        std::cout << std::endl;
        std::cout << "Final NLL = " << nll_value << std::endl;
        std::cout << "Final gradient norm = " << grad_norm << std::endl;
    }
    
    void print_iteration_info(int iter, float grad_norm, float nll_value, float learning_rate,
                             ConvergenceCriterion criterion) {
        std::cout << "Iteration " << iter << ": ";
        print_parameter_values();
        std::cout << ", NLL = " << nll_value
                  << ", |grad| = " << std::scientific << std::setprecision(4) << grad_norm
                  << std::fixed << std::setprecision(4)
                  << ", lr = " << learning_rate;
        std::cout << std::endl;
    }
    
    void print_max_iterations_reached() {
        std::cout << "\nReached maximum iterations without convergence" << std::endl;
        std::cout << "Final parameters: ";
        print_parameter_values();
        std::cout << std::endl;
    }
};

// Alias for backward compatibility
using StatisticalModel = Distribution;

} // namespace statmodels