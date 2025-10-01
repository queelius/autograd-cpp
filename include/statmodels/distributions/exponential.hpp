#pragma once

#include "../base/distribution.hpp"
#include "../../autograd/autograd.hpp"
#include <vector>
#include <random>
#include <iostream>
#include <limits>
#include <cmath>

namespace statmodels {

using namespace autograd;

// Exponential distribution model
// PDF: f(x|λ) = λ * exp(-λx) for x ≥ 0
// Log-likelihood: log L(λ|x) = n*log(λ) - λ*Σx_i
class ExponentialDistribution : public Distribution {
private:
    TensorPtr log_lambda;  // Log-parameterization to ensure λ > 0
    
public:
    // Initialize with a starting value for lambda
    ExponentialDistribution(float initial_lambda = 1.0f) {
        float init_log = std::log(initial_lambda);
        log_lambda = from_vector({init_log}, {1}, true);
    }
    
    // Get current lambda value
    float get_lambda() const {
        return std::exp(log_lambda->data[0]);
    }
    
    // Set lambda value
    void set_lambda(float lambda) {
        log_lambda->data[0] = std::log(lambda);
        log_lambda->zero_grad();
    }
    
    // Override to handle log-parameterization correctly
    std::vector<float> get_parameter_values() override {
        return {get_lambda()};
    }
    
    void set_parameter_values(const std::vector<float>& values) override {
        if (!values.empty()) {
            set_lambda(values[0]);
        }
    }
    
    // Compute log probability density for a single point
    TensorPtr log_pdf(TensorPtr x) override {
        auto lambda = exp(log_lambda);
        // log f(x|λ) = log(λ) - λ*x
        return sub(log(lambda), mul(lambda, x));
    }
    
    // CDF: F(x|λ) = 1 - exp(-λx)
    TensorPtr cdf(TensorPtr x) override {
        auto lambda = exp(log_lambda);
        auto one = from_vector({1.0f}, {1}, false);
        auto minus_one = from_vector({-1.0f}, {1}, false);
        return sub(one, exp(mul(minus_one, mul(lambda, x))));
    }
    
    // Quantile function: F^-1(p) = -log(1-p) / λ
    TensorPtr quantile(TensorPtr p) override {
        auto lambda = exp(log_lambda);
        auto one = from_vector({1.0f}, {1}, false);
        auto minus_one = from_vector({-1.0f}, {1}, false);
        return div(mul(minus_one, log(sub(one, p))), lambda);
    }
    
    // Generate samples from the current distribution
    std::vector<float> sample(size_t n_samples, uint32_t seed = 42) override {
        std::mt19937 gen(seed);
        std::exponential_distribution<float> dist(get_lambda());
        
        std::vector<float> samples;
        samples.reserve(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            samples.push_back(dist(gen));
        }
        return samples;
    }
    
    // Statistical properties
    float mean() const override {
        return 1.0f / get_lambda();
    }
    
    float variance() const override {
        float lambda = get_lambda();
        return 1.0f / (lambda * lambda);
    }
    
    // Analytical MLE (for comparison)
    static float analytical_mle(const std::vector<float>& data) {
        float sum = 0.0f;
        for (float x : data) {
            sum += x;
        }
        return data.size() / sum;
    }
    
    // Get parameter gradient (useful for diagnostics)
    float get_gradient() const {
        return log_lambda->grad[0];
    }
    
    // Get the parameter tensors (for optimization)
    std::vector<TensorPtr> get_parameters() override {
        return {log_lambda};
    }
    
    // Zero gradients
    void zero_grad() override {
        log_lambda->zero_grad();
    }
    
    // Get model name
    std::string get_name() const override {
        return "Exponential";
    }
    
protected:
    // Override to provide custom parameter printing
    void print_parameter_values() override {
        std::cout << "λ = " << get_lambda();
    }
};

// Helper function to generate exponential data
std::vector<float> generate_exponential_data(float lambda, size_t n_samples, uint32_t seed = 42) {
    ExponentialDistribution exp_dist(lambda);
    return exp_dist.sample(n_samples, seed);
}

// Alias for backward compatibility
using ExponentialModel = ExponentialDistribution;

} // namespace statmodels