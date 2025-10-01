#pragma once

#include "../base/distribution.hpp"
#include "../../autograd/autograd.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <stdexcept>

namespace statmodels {

using namespace autograd;

// Empirical distribution class representing a discrete distribution from samples
class EmpiricalDistribution : public Distribution {
private:
    std::vector<std::vector<float>> samples_;  // Stored samples (each inner vector is a parameter set)
    size_t n_params_;
    
public:
    // Constructor from bootstrap parameter samples
    EmpiricalDistribution(const std::vector<std::vector<float>>& samples) 
        : samples_(samples) {
        if (samples.empty()) {
            throw std::invalid_argument("EmpiricalDistribution requires non-empty samples");
        }
        n_params_ = samples[0].size();
    }
    
    // Get all samples
    const std::vector<std::vector<float>>& get_samples() const {
        return samples_;
    }
    
    // Get mean of each parameter
    std::vector<float> get_parameter_values() override {
        std::vector<float> means(n_params_, 0.0f);
        for (const auto& sample : samples_) {
            for (size_t i = 0; i < n_params_; ++i) {
                means[i] += sample[i];
            }
        }
        for (size_t i = 0; i < n_params_; ++i) {
            means[i] /= samples_.size();
        }
        return means;
    }
    
    // Set parameter values (not applicable for empirical distribution)
    void set_parameter_values(const std::vector<float>& values) override {
        // No-op for empirical distribution
    }
    
    // Get standard deviation of each parameter
    std::vector<float> get_standard_deviations() {
        auto means = get_parameter_values();
        std::vector<float> stds(n_params_, 0.0f);
        
        for (const auto& sample : samples_) {
            for (size_t i = 0; i < n_params_; ++i) {
                float diff = sample[i] - means[i];
                stds[i] += diff * diff;
            }
        }
        
        for (size_t i = 0; i < n_params_; ++i) {
            stds[i] = std::sqrt(stds[i] / (samples_.size() - 1));
        }
        
        return stds;
    }
    
    // Get percentile for each parameter
    std::vector<float> get_percentiles(float percentile) {
        if (percentile < 0.0f || percentile > 100.0f) {
            throw std::invalid_argument("Percentile must be between 0 and 100");
        }
        
        std::vector<float> result(n_params_);
        
        for (size_t p = 0; p < n_params_; ++p) {
            std::vector<float> param_values;
            for (const auto& sample : samples_) {
                param_values.push_back(sample[p]);
            }
            
            std::sort(param_values.begin(), param_values.end());
            
            size_t idx = static_cast<size_t>((percentile / 100.0f) * (param_values.size() - 1));
            result[p] = param_values[idx];
        }
        
        return result;
    }
    
    // Get confidence intervals
    std::vector<std::pair<float, float>> get_confidence_intervals(float confidence_level = 0.95f) {
        float alpha = 1.0f - confidence_level;
        auto lower = get_percentiles(100.0f * alpha / 2.0f);
        auto upper = get_percentiles(100.0f * (1.0f - alpha / 2.0f));
        
        std::vector<std::pair<float, float>> intervals;
        for (size_t i = 0; i < n_params_; ++i) {
            intervals.push_back({lower[i], upper[i]});
        }
        
        return intervals;
    }
    
    // Get median of each parameter
    std::vector<float> get_medians() {
        return get_percentiles(50.0f);
    }
    
    // Get interquartile range
    std::vector<std::pair<float, float>> get_iqr() {
        auto q1 = get_percentiles(25.0f);
        auto q3 = get_percentiles(75.0f);
        
        std::vector<std::pair<float, float>> iqr;
        for (size_t i = 0; i < n_params_; ++i) {
            iqr.push_back({q1[i], q3[i]});
        }
        
        return iqr;
    }
    
    // Sample from the empirical distribution (resample from stored samples)
    std::vector<float> sample(size_t n_samples, uint32_t seed = 42) override {
        std::mt19937 gen(seed);
        std::uniform_int_distribution<size_t> dist(0, samples_.size() - 1);
        
        // For now, return samples of the first parameter
        // This could be extended to return multivariate samples
        std::vector<float> result;
        for (size_t i = 0; i < n_samples; ++i) {
            size_t idx = dist(gen);
            result.push_back(samples_[idx][0]);
        }
        
        return result;
    }
    
    // Get covariance matrix (for multivariate analysis)
    std::vector<std::vector<float>> get_covariance_matrix() {
        auto means = get_parameter_values();
        std::vector<std::vector<float>> cov(n_params_, std::vector<float>(n_params_, 0.0f));
        
        for (const auto& sample : samples_) {
            for (size_t i = 0; i < n_params_; ++i) {
                for (size_t j = 0; j < n_params_; ++j) {
                    cov[i][j] += (sample[i] - means[i]) * (sample[j] - means[j]);
                }
            }
        }
        
        // Normalize
        float n = static_cast<float>(samples_.size() - 1);
        for (size_t i = 0; i < n_params_; ++i) {
            for (size_t j = 0; j < n_params_; ++j) {
                cov[i][j] /= n;
            }
        }
        
        return cov;
    }
    
    // Get correlation matrix
    std::vector<std::vector<float>> get_correlation_matrix() {
        auto cov = get_covariance_matrix();
        auto stds = get_standard_deviations();
        std::vector<std::vector<float>> corr(n_params_, std::vector<float>(n_params_));
        
        for (size_t i = 0; i < n_params_; ++i) {
            for (size_t j = 0; j < n_params_; ++j) {
                if (stds[i] > 0 && stds[j] > 0) {
                    corr[i][j] = cov[i][j] / (stds[i] * stds[j]);
                } else {
                    corr[i][j] = (i == j) ? 1.0f : 0.0f;
                }
            }
        }
        
        return corr;
    }
    
    // Statistical properties (for first parameter by default)
    float mean() const override {
        float sum = 0.0f;
        for (const auto& sample : samples_) {
            sum += sample[0];
        }
        return sum / samples_.size();
    }
    
    float variance() const override {
        float m = mean();
        float var = 0.0f;
        for (const auto& sample : samples_) {
            float diff = sample[0] - m;
            var += diff * diff;
        }
        return var / (samples_.size() - 1);
    }
    
    // Required overrides (not meaningful for empirical distribution)
    TensorPtr log_pdf(TensorPtr x) override {
        throw std::runtime_error("log_pdf not implemented for EmpiricalDistribution");
    }
    
    TensorPtr cdf(TensorPtr x) override {
        throw std::runtime_error("cdf not implemented for EmpiricalDistribution");
    }
    
    TensorPtr quantile(TensorPtr p) override {
        throw std::runtime_error("quantile not implemented for EmpiricalDistribution");
    }
    
    std::vector<TensorPtr> get_parameters() override {
        return {};  // No trainable parameters
    }
    
    void zero_grad() override {
        // No-op
    }
    
    std::string get_name() const override {
        return "Empirical";
    }
    
protected:
    void print_parameter_values() override {
        auto means = get_parameter_values();
        std::cout << "means = [";
        for (size_t i = 0; i < means.size(); ++i) {
            std::cout << means[i];
            if (i < means.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
    }
};

// Specialized bootstrap distribution that extends EmpiricalDistribution
class BootstrapDistribution : public EmpiricalDistribution {
private:
    std::string model_name_;
    float confidence_level_;
    
public:
    BootstrapDistribution(const std::vector<std::vector<float>>& samples,
                         const std::string& model_name,
                         float confidence_level = 0.95f)
        : EmpiricalDistribution(samples), 
          model_name_(model_name),
          confidence_level_(confidence_level) {}
    
    std::string get_name() const override {
        return "Bootstrap(" + model_name_ + ")";
    }
    
    // Get bootstrap estimate (mean of the sampling distribution)
    std::vector<float> get_bootstrap_estimate() {
        return get_parameter_values();
    }
    
    // Get bootstrap standard error
    std::vector<float> get_bootstrap_standard_error() {
        return get_standard_deviations();
    }
    
    // Get bootstrap confidence intervals using the stored confidence level
    std::vector<std::pair<float, float>> get_bootstrap_ci() {
        return get_confidence_intervals(confidence_level_);
    }
    
    // Get bias estimate (if true parameters are known)
    std::vector<float> get_bias(const std::vector<float>& true_params) {
        auto estimates = get_bootstrap_estimate();
        std::vector<float> bias;
        
        for (size_t i = 0; i < estimates.size() && i < true_params.size(); ++i) {
            bias.push_back(estimates[i] - true_params[i]);
        }
        
        return bias;
    }
};

} // namespace statmodels