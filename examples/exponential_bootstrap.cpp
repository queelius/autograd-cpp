#include <autograd/autograd.hpp>
#include <statmodels/distributions/exponential.hpp>
#include <statmodels/inference/bootstrap.hpp>
#include <iostream>
#include <iomanip>

using namespace autograd;
using namespace statmodels;

int main() {
    std::cout << "=== Exponential Distribution: Bootstrap Confidence Intervals ===" << std::endl;
    std::cout << std::endl;
    
    // True parameter
    const float true_lambda = 2.0f;
    const size_t n_samples = 100;
    
    // Generate synthetic data
    std::cout << "Generating " << n_samples << " samples from Exponential(λ=" 
              << true_lambda << ")" << std::endl;
    auto data = generate_exponential_data(true_lambda, n_samples, 42);
    
    // Compute sample mean (1/λ for exponential)
    float sample_mean = 0.0f;
    for (float x : data) {
        sample_mean += x;
    }
    sample_mean /= n_samples;
    std::cout << "Sample mean: " << std::fixed << std::setprecision(4) << sample_mean << std::endl;
    std::cout << "Expected mean (1/λ): " << (1.0f / true_lambda) << std::endl;
    std::cout << std::endl;
    
    // Fit the model first
    std::cout << "=== Initial Model Fitting ===" << std::endl;
    ExponentialModel model(1.5f);  // Start with λ=1.5
    std::cout << "Initial λ = " << model.get_lambda() << std::endl;
    
    model.fit_adaptive(data, 1e-4, 0.01f, 1000, ConvergenceCriterion::GRADIENT, false);
    
    std::cout << "Fitted λ = " << model.get_lambda() << std::endl;
    std::cout << "True λ = " << true_lambda << std::endl;
    std::cout << std::endl;
    
    // Perform bootstrap analysis
    std::cout << "=== Bootstrap Analysis ===" << std::endl;
    std::cout << "Performing non-parametric bootstrap with 200 resamples..." << std::endl;
    
    auto bootstrap_results = Bootstrap::run(
        model,           // Model to use
        data,           // Original data
        200,            // Number of bootstrap samples
        0.95f,          // Confidence level (95%)
        1e-4,           // Convergence tolerance
        0.01f,          // Initial learning rate
        500,            // Max iterations per bootstrap
        true,           // Verbose
        123,            // Random seed
        true            // Return bootstrap distribution
    );
    
    // Print detailed results
    std::cout << "\n=== Summary Statistics ===" << std::endl;
    std::cout << "True λ:                " << true_lambda << std::endl;
    std::cout << "Point estimate λ:      " << model.get_lambda() << std::endl;
    std::cout << "Bootstrap mean λ:      " << bootstrap_results.parameter_means[0] << std::endl;
    std::cout << "Bootstrap std λ:       " << bootstrap_results.parameter_stds[0] << std::endl;
    std::cout << "95% CI for λ:          [" 
              << bootstrap_results.parameter_ci_lower[0] << ", "
              << bootstrap_results.parameter_ci_upper[0] << "]" << std::endl;
    
    // Check if true parameter is in confidence interval
    bool in_ci = (true_lambda >= bootstrap_results.parameter_ci_lower[0] && 
                  true_lambda <= bootstrap_results.parameter_ci_upper[0]);
    std::cout << "True λ in 95% CI:      " << (in_ci ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    // Demonstrate using the bootstrap distribution
    if (bootstrap_results.bootstrap_distribution) {
        std::cout << "=== Bootstrap Distribution Analysis ===" << std::endl;
        auto dist = bootstrap_results.bootstrap_distribution;
        
        // Get various statistics from the empirical distribution
        auto medians = dist->get_medians();
        auto iqr = dist->get_iqr();
        
        std::cout << "Median λ:              " << medians[0] << std::endl;
        std::cout << "IQR for λ:             [" << iqr[0].first << ", " << iqr[0].second << "]" << std::endl;
        
        // Get percentiles
        auto p5 = dist->get_percentiles(5.0f);
        auto p95 = dist->get_percentiles(95.0f);
        std::cout << "5th percentile:        " << p5[0] << std::endl;
        std::cout << "95th percentile:       " << p95[0] << std::endl;
        
        // Get bootstrap standard error
        auto se = dist->get_bootstrap_standard_error();
        std::cout << "Bootstrap SE:          " << se[0] << std::endl;
        std::cout << std::endl;
    }
    
    // Try parametric bootstrap
    std::cout << "=== Parametric Bootstrap ===" << std::endl;
    std::cout << "Performing parametric bootstrap with 200 resamples..." << std::endl;
    
    auto parametric_results = Bootstrap::run_parametric(
        model,          // Model to use
        n_samples,      // Sample size to generate
        200,            // Number of bootstrap samples
        0.95f,          // Confidence level
        1e-4,           // Convergence tolerance
        0.01f,          // Initial learning rate
        500,            // Max iterations
        true,           // Verbose
        456,            // Random seed
        false           // Don't return distribution (to demonstrate different usage)
    );
    
    std::cout << "\n=== Parametric Bootstrap Summary ===" << std::endl;
    std::cout << "Bootstrap mean λ:      " << parametric_results.parameter_means[0] << std::endl;
    std::cout << "Bootstrap std λ:       " << parametric_results.parameter_stds[0] << std::endl;
    std::cout << "95% CI for λ:          [" 
              << parametric_results.parameter_ci_lower[0] << ", "
              << parametric_results.parameter_ci_upper[0] << "]" << std::endl;
    
    // Check if true parameter is in confidence interval
    in_ci = (true_lambda >= parametric_results.parameter_ci_lower[0] && 
             true_lambda <= parametric_results.parameter_ci_upper[0]);
    std::cout << "True λ in 95% CI:      " << (in_ci ? "Yes" : "No") << std::endl;
    
    return 0;
}