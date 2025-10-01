#pragma once

#include "../base/distribution.hpp"
#include "empirical_distribution.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <memory>

namespace statmodels {

// Structure to hold bootstrap results
struct BootstrapResults {
    std::vector<std::vector<float>> parameter_samples;  // Each inner vector is one bootstrap sample's parameters
    std::vector<float> parameter_means;
    std::vector<float> parameter_stds;
    std::vector<float> parameter_ci_lower;  // e.g., 2.5th percentile
    std::vector<float> parameter_ci_upper;  // e.g., 97.5th percentile
    float confidence_level;
    std::shared_ptr<BootstrapDistribution> bootstrap_distribution;  // Optional bootstrap distribution
};

// Bootstrap class for parameter estimation with confidence intervals
class Bootstrap {
public:
    // Perform bootstrap resampling and parameter estimation
    static BootstrapResults run(Distribution& model,
                               const std::vector<float>& data,
                               int n_bootstrap = 100,
                               float confidence_level = 0.95f,
                               float tolerance = 1e-3,
                               float initial_lr = 0.001f,
                               int max_iterations = 1000,
                               bool verbose = false,
                               uint32_t seed = 42,
                               bool return_distribution = false) {
        
        if (verbose) {
            std::cout << "\n=== Bootstrap Analysis ===" << std::endl;
            std::cout << "Number of bootstrap samples: " << n_bootstrap << std::endl;
            std::cout << "Confidence level: " << (confidence_level * 100) << "%" << std::endl;
            std::cout << "Original sample size: " << data.size() << std::endl;
            std::cout << std::endl;
        }
        
        BootstrapResults results;
        results.confidence_level = confidence_level;
        
        std::mt19937 gen(seed);
        std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
        
        // Store original parameters to reset after each bootstrap
        auto original_params = model.get_parameter_values();
        
        // Perform bootstrap iterations
        for (int b = 0; b < n_bootstrap; ++b) {
            if (verbose && (b % 10 == 0 || b == n_bootstrap - 1)) {
                std::cout << "Bootstrap iteration " << (b + 1) << "/" << n_bootstrap << std::endl;
            }
            
            // Create bootstrap sample (sampling with replacement)
            std::vector<float> bootstrap_sample;
            bootstrap_sample.reserve(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                bootstrap_sample.push_back(data[dist(gen)]);
            }
            
            // Reset model to original parameters
            model.set_parameter_values(original_params);
            
            // Fit model to bootstrap sample
            model.fit_adaptive(bootstrap_sample, tolerance, initial_lr, max_iterations,
                             ConvergenceCriterion::GRADIENT, false);
            
            // Store fitted parameters
            auto params = model.get_parameter_values();
            results.parameter_samples.push_back(params);
        }
        
        // Calculate statistics
        calculate_statistics(results, confidence_level);
        
        // Create bootstrap distribution if requested
        if (return_distribution && !results.parameter_samples.empty()) {
            results.bootstrap_distribution = std::make_shared<BootstrapDistribution>(
                results.parameter_samples, model.get_name(), confidence_level);
        }
        
        // Reset model to original parameters
        model.set_parameter_values(original_params);
        
        if (verbose) {
            print_results(results, model);
        }
        
        return results;
    }
    
    // Perform parametric bootstrap (generate new data from fitted model)
    static BootstrapResults run_parametric(Distribution& model,
                                          size_t sample_size,
                                          int n_bootstrap = 100,
                                          float confidence_level = 0.95f,
                                          float tolerance = 1e-3,
                                          float initial_lr = 0.001f,
                                          int max_iterations = 1000,
                                          bool verbose = false,
                                          uint32_t seed = 42,
                                          bool return_distribution = false) {
        
        if (verbose) {
            std::cout << "\n=== Parametric Bootstrap Analysis ===" << std::endl;
            std::cout << "Number of bootstrap samples: " << n_bootstrap << std::endl;
            std::cout << "Confidence level: " << (confidence_level * 100) << "%" << std::endl;
            std::cout << "Sample size per bootstrap: " << sample_size << std::endl;
            std::cout << std::endl;
        }
        
        BootstrapResults results;
        results.confidence_level = confidence_level;
        
        // Store original (fitted) parameters for data generation
        auto fitted_params = model.get_parameter_values();
        
        // Define neutral starting parameters
        std::vector<float> neutral_params = fitted_params;
        for (size_t i = 0; i < neutral_params.size(); ++i) {
            neutral_params[i] = fitted_params[i] * (0.7f + 0.6f * (float)rand() / RAND_MAX);
        }
        
        // Perform bootstrap iterations
        for (int b = 0; b < n_bootstrap; ++b) {
            if (verbose && (b % 10 == 0 || b == n_bootstrap - 1)) {
                std::cout << "Bootstrap iteration " << (b + 1) << "/" << n_bootstrap << std::endl;
            }
            
            // Ensure model has fitted parameters for data generation
            model.set_parameter_values(fitted_params);
            
            // Generate new data from fitted model parameters
            auto bootstrap_sample = model.sample(sample_size, seed + b);
            
            // Reset model to neutral starting point for fitting
            model.set_parameter_values(neutral_params);
            
            // Fit model to bootstrap sample
            model.fit_adaptive(bootstrap_sample, tolerance, initial_lr, max_iterations,
                             ConvergenceCriterion::GRADIENT, false);
            
            // Store fitted parameters
            auto params = model.get_parameter_values();
            results.parameter_samples.push_back(params);
        }
        
        // Calculate statistics
        calculate_statistics(results, confidence_level);
        
        // Create bootstrap distribution if requested
        if (return_distribution && !results.parameter_samples.empty()) {
            results.bootstrap_distribution = std::make_shared<BootstrapDistribution>(
                results.parameter_samples, model.get_name(), confidence_level);
        }
        
        // Reset model to original fitted parameters
        model.set_parameter_values(fitted_params);
        
        if (verbose) {
            print_results(results, model);
        }
        
        return results;
    }
    
private:
    static void calculate_statistics(BootstrapResults& results, float confidence_level) {
        if (results.parameter_samples.empty()) return;
        
        size_t n_params = results.parameter_samples[0].size();
        size_t n_samples = results.parameter_samples.size();
        
        // Initialize vectors
        results.parameter_means.resize(n_params, 0.0f);
        results.parameter_stds.resize(n_params, 0.0f);
        results.parameter_ci_lower.resize(n_params);
        results.parameter_ci_upper.resize(n_params);
        
        // Calculate percentiles for confidence intervals
        float alpha = 1.0f - confidence_level;
        size_t lower_idx = static_cast<size_t>(n_samples * (alpha / 2.0f));
        size_t upper_idx = static_cast<size_t>(n_samples * (1.0f - alpha / 2.0f));
        
        // Process each parameter
        for (size_t p = 0; p < n_params; ++p) {
            // Extract values for this parameter across all bootstrap samples
            std::vector<float> param_values;
            for (const auto& sample : results.parameter_samples) {
                param_values.push_back(sample[p]);
            }
            
            // Calculate mean
            float mean = std::accumulate(param_values.begin(), param_values.end(), 0.0f) / n_samples;
            results.parameter_means[p] = mean;
            
            // Calculate standard deviation
            float variance = 0.0f;
            for (float val : param_values) {
                float diff = val - mean;
                variance += diff * diff;
            }
            results.parameter_stds[p] = std::sqrt(variance / (n_samples - 1));
            
            // Sort for percentiles
            std::sort(param_values.begin(), param_values.end());
            
            // Get confidence interval bounds
            results.parameter_ci_lower[p] = param_values[lower_idx];
            results.parameter_ci_upper[p] = param_values[upper_idx];
        }
    }
    
    static void print_results(const BootstrapResults& results, const Distribution& model) {
        std::cout << "\n=== Bootstrap Results for " << model.get_name() << " Distribution ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        
        for (size_t i = 0; i < results.parameter_means.size(); ++i) {
            std::cout << "Parameter " << i << ":" << std::endl;
            std::cout << "  Mean:     " << results.parameter_means[i] << std::endl;
            std::cout << "  Std Dev:  " << results.parameter_stds[i] << std::endl;
            std::cout << "  " << (results.confidence_level * 100) << "% CI: ["
                     << results.parameter_ci_lower[i] << ", "
                     << results.parameter_ci_upper[i] << "]" << std::endl;
        }
        std::cout << std::endl;
    }
};

} // namespace statmodels