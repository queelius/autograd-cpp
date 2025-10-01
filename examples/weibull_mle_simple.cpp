#include <autograd/autograd.hpp>
#include <statmodels/weibull.hpp>
#include <iostream>
#include <iomanip>

using namespace autograd;
using namespace statmodels;

int main() {
    std::cout << "=== Learning Weibull Distribution Parameters from Data ===" << std::endl;
    std::cout << std::endl;
    
    // True parameters
    const float true_k = 2.5f;      // shape parameter
    const float true_lambda = 3.0f;  // scale parameter
    const size_t n_samples = 100;  // Small sample size test
    
    // Generate synthetic data
    std::cout << "Generating " << n_samples << " samples from Weibull(k=" 
              << true_k << ", λ=" << true_lambda << ")" << std::endl;
    auto data = generate_weibull_data(true_k, true_lambda, n_samples, 123);
    
    // Compute sample statistics
    float sample_mean = 0.0f;
    float sample_min = data[0];
    float sample_max = data[0];
    for (float x : data) {
        sample_mean += x;
        sample_min = std::min(sample_min, x);
        sample_max = std::max(sample_max, x);
    }
    sample_mean /= n_samples;
    
    std::cout << "Sample statistics:" << std::endl;
    std::cout << "  Mean:     " << std::fixed << std::setprecision(4) << sample_mean << std::endl;
    std::cout << "  Min:      " << sample_min << std::endl;
    std::cout << "  Max:      " << sample_max << std::endl;
    std::cout << std::endl;
    
    // Fit model
    std::cout << "=== Fitting Model with Gradient Descent ===" << std::endl;
    WeibullModel model(1.5f, 2.0f);  // Start with k=1.5, λ=2.0
    std::cout << "Initial k = " << model.get_k() << ", λ = " << model.get_lambda() << std::endl;
    std::cout << "True k = " << true_k << ", λ = " << true_lambda << std::endl;
    std::cout << "Convergence tolerance: 1e-3 on gradient norm" << std::endl;
    std::cout << std::endl;
    
    int iterations = model.fit_adaptive(data, 1e-3, 0.001f, 5000, 
                                       ConvergenceCriterion::GRADIENT, true);
    
    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Converged in " << iterations << " iterations" << std::endl;
    std::cout << "True k:      " << true_k << std::endl;
    std::cout << "Estimated k: " << model.get_k() << std::endl;
    std::cout << "True λ:      " << true_lambda << std::endl;
    std::cout << "Estimated λ: " << model.get_lambda() << std::endl;
    std::cout << std::endl;
    
    // Test the learned model
    std::cout << "=== Testing Learned Model ===" << std::endl;
    
    // Generate new data from learned distribution
    auto new_samples = model.sample(20, 123);
    
    std::cout << "First 5 samples from learned distribution:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << new_samples[i] << std::endl;
    }
    
    // Compute likelihood of a test point
    auto test_point = from_vector({2.5f}, {1}, false);
    auto pdf_val = model.pdf(test_point);
    std::cout << std::endl;
    std::cout << "PDF at x=2.5: " << pdf_val->data[0] << std::endl;
    
    // Compute true PDF for comparison
    float x = 2.5f;
    float true_pdf = (true_k / true_lambda) * std::pow(x / true_lambda, true_k - 1) * 
                     std::exp(-std::pow(x / true_lambda, true_k));
    std::cout << "True PDF:     " << true_pdf << std::endl;
    
    return 0;
}