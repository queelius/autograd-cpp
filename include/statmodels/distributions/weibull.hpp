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

// Weibull distribution model
// PDF: f(x|k,λ) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k) for x ≥ 0
// Log-likelihood: log L(k,λ|x) = n*log(k) - n*k*log(λ) + (k-1)*Σlog(x_i) - Σ(x_i/λ)^k
class WeibullDistribution : public Distribution {
private:
    TensorPtr log_k;      // Log-parameterization of shape parameter k > 0
    TensorPtr log_lambda; // Log-parameterization of scale parameter λ > 0
    
public:
    // Initialize with starting values
    WeibullDistribution(float initial_k = 1.0f, float initial_lambda = 1.0f) {
        float init_log_k = std::log(initial_k);
        float init_log_lambda = std::log(initial_lambda);
        log_k = from_vector({init_log_k}, {1}, true);
        log_lambda = from_vector({init_log_lambda}, {1}, true);
    }
    
    // Get current parameter values
    float get_k() const {
        return std::exp(log_k->data[0]);
    }
    
    float get_lambda() const {
        return std::exp(log_lambda->data[0]);
    }
    
    // Set parameter values
    void set_parameters(float k, float lambda) {
        log_k->data[0] = std::log(k);
        log_lambda->data[0] = std::log(lambda);
        log_k->zero_grad();
        log_lambda->zero_grad();
    }
    
    // Override to handle log-parameterization correctly
    std::vector<float> get_parameter_values() override {
        return {get_k(), get_lambda()};
    }
    
    void set_parameter_values(const std::vector<float>& values) override {
        if (values.size() >= 2) {
            set_parameters(values[0], values[1]);
        }
    }
    
    // Override with efficient batch computation
    TensorPtr negative_log_likelihood(const std::vector<float>& data) override {
        size_t n = data.size();
        
        // Compute required statistics
        float log_sum = 0.0f;
        for (float x : data) {
            if (x > 0) {
                log_sum += std::log(x);
            }
        }
        
        // Get parameters
        auto k = exp(log_k);
        auto lambda = exp(log_lambda);
        
        // Create tensors for constants
        auto n_tensor = from_vector({static_cast<float>(n)}, {1}, false);
        auto log_sum_tensor = from_vector({log_sum}, {1}, false);
        
        // Compute Σ(x_i/λ)^k more efficiently
        float x_k_sum = 0.0f;
        float k_val = std::exp(log_k->data[0]);
        for (float x : data) {
            x_k_sum += std::pow(x, k_val);
        }
        auto x_k_sum_tensor = from_vector({x_k_sum}, {1}, false);
        
        // Compute (1/λ^k) * sum(x_i^k)
        auto lambda_k = exp(mul(k, log(lambda)));
        auto power_sum = div(x_k_sum_tensor, lambda_k);
        
        // NLL = -n*log(k) + n*k*log(λ) - (k-1)*Σlog(x_i) + Σ(x_i/λ)^k
        auto term1 = mul(n_tensor, log(k));
        auto term2 = mul(mul(n_tensor, k), log(lambda));
        auto one = from_vector({1.0f}, {1}, false);
        auto k_minus_1 = sub(k, one);
        auto term3 = mul(k_minus_1, log_sum_tensor);
        
        auto nll = sub(add(sub(term2, term1), power_sum), term3);
        
        return nll;
    }
    
    // Compute log probability density for a single point
    TensorPtr log_pdf(TensorPtr x) override {
        auto k = exp(log_k);
        auto lambda = exp(log_lambda);
        
        // log f(x|k,λ) = log(k) - log(λ) + (k-1)*log(x/λ) - (x/λ)^k
        auto x_over_lambda = div(x, lambda);
        auto one = from_vector({1.0f}, {1}, false);
        auto k_minus_1 = sub(k, one);
        
        auto term1 = log(k);
        auto term2 = log(lambda);
        auto term3 = mul(k_minus_1, log(x_over_lambda));
        auto term4 = exp(mul(k, log(x_over_lambda)));
        
        return sub(sub(add(term1, term3), term2), term4);
    }
    
    // CDF: F(x|k,λ) = 1 - exp(-(x/λ)^k)
    TensorPtr cdf(TensorPtr x) override {
        auto k = exp(log_k);
        auto lambda = exp(log_lambda);
        auto one = from_vector({1.0f}, {1}, false);
        auto minus_one = from_vector({-1.0f}, {1}, false);
        
        auto x_over_lambda = div(x, lambda);
        auto power = exp(mul(k, log(x_over_lambda)));
        
        return sub(one, exp(mul(minus_one, power)));
    }
    
    // Quantile function: F^-1(p) = λ * (-log(1-p))^(1/k)
    TensorPtr quantile(TensorPtr p) override {
        auto k = exp(log_k);
        auto lambda = exp(log_lambda);
        auto one = from_vector({1.0f}, {1}, false);
        auto minus_one = from_vector({-1.0f}, {1}, false);
        
        auto neg_log_1_minus_p = mul(minus_one, log(sub(one, p)));
        auto power = exp(div(log(neg_log_1_minus_p), k));
        
        return mul(lambda, power);
    }
    
    // Generate samples from the current distribution
    std::vector<float> sample(size_t n_samples, uint32_t seed = 42) override {
        std::mt19937 gen(seed);
        std::weibull_distribution<float> dist(get_k(), get_lambda());
        
        std::vector<float> samples;
        samples.reserve(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            samples.push_back(dist(gen));
        }
        return samples;
    }
    
    // Statistical properties
    float mean() const override {
        float k = get_k();
        float lambda = get_lambda();
        // Mean = λ * Γ(1 + 1/k)
        // Using approximation for gamma function
        return lambda * std::tgamma(1.0f + 1.0f/k);
    }
    
    float variance() const override {
        float k = get_k();
        float lambda = get_lambda();
        // Var = λ^2 * [Γ(1 + 2/k) - Γ(1 + 1/k)^2]
        float gamma1 = std::tgamma(1.0f + 1.0f/k);
        float gamma2 = std::tgamma(1.0f + 2.0f/k);
        return lambda * lambda * (gamma2 - gamma1 * gamma1);
    }
    
    // Get parameter tensors
    std::vector<TensorPtr> get_parameters() override {
        return {log_k, log_lambda};
    }
    
    // Zero gradients
    void zero_grad() override {
        log_k->zero_grad();
        log_lambda->zero_grad();
    }
    
    // Get model name
    std::string get_name() const override {
        return "Weibull";
    }
    
protected:
    // Override to provide custom parameter printing
    void print_parameter_values() override {
        std::cout << "k = " << get_k() << ", λ = " << get_lambda();
    }
};

// Helper function to generate Weibull data
std::vector<float> generate_weibull_data(float k, float lambda, size_t n_samples, uint32_t seed = 42) {
    WeibullDistribution weibull_dist(k, lambda);
    return weibull_dist.sample(n_samples, seed);
}

// Alias for backward compatibility
using WeibullModel = WeibullDistribution;

} // namespace statmodels