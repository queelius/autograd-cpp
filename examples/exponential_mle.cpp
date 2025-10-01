#include <autograd/autograd.hpp>
#include <statmodels/exponential.hpp>
#include <iostream>
#include <iomanip>

using namespace autograd;
using namespace statmodels;

int main() {
    std::cout << "=== Learning Exponential Distribution Parameters from Data ===" << std::endl;
    std::cout << std::endl;
    
    // True parameter
    const float true_lambda = 2.0f;
    const size_t n_samples = 100;
    
    // Generate synthetic data
    std::cout << "Generating " << n_samples << " samples from Exponential(λ=" << true_lambda << ")" << std::endl;
    auto data = generate_exponential_data(true_lambda, n_samples, 42);
    
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
    std::cout << "  Mean:     " << std::fixed << std::setprecision(4) << sample_mean 
              << " (theoretical: " << 1.0f/true_lambda << ")" << std::endl;
    std::cout << "  Min:      " << sample_min << std::endl;
    std::cout << "  Max:      " << sample_max << std::endl;
    std::cout << std::endl;
    
    // Analytical MLE
    float analytical_lambda = ExponentialModel::analytical_mle(data);
    std::cout << "Analytical MLE: λ = " << analytical_lambda << std::endl;
    std::cout << std::endl;
    
    // Fit model using adaptive gradient descent
    std::cout << "=== Fitting Model with Adaptive Gradient Descent ===" << std::endl;
    ExponentialModel model(0.01f);  // Start far from optimum to trigger adaptation
    
    std::cout << "Initial λ = " << model.get_lambda() << std::endl;
    std::cout << "Target λ (analytical MLE) = " << analytical_lambda << std::endl;
    std::cout << "Convergence tolerance: 1e-5" << std::endl;
    std::cout << std::endl;
    
    // Fit with adaptive learning rate until convergence (high initial LR to trigger adaptation)
    int iterations = model.fit_adaptive(data, 1e-5, 0.5f, 10000, 
                                       ConvergenceCriterion::GRADIENT, true);
    
    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "Converged in " << iterations << " iterations" << std::endl;
    std::cout << "True λ:           " << true_lambda << std::endl;
    std::cout << "Analytical MLE:   " << analytical_lambda << std::endl;
    std::cout << "Gradient-based:   " << model.get_lambda() << std::endl;
    std::cout << "Final gradient:   " << model.get_gradient() << " (should be ~0)" << std::endl;
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
    auto test_point = from_vector({0.5f}, {1}, false);
    auto pdf_val = model.pdf(test_point);
    std::cout << std::endl;
    std::cout << "PDF at x=0.5: " << pdf_val->data[0] << std::endl;
    std::cout << "True PDF:     " << true_lambda * std::exp(-true_lambda * 0.5f) << std::endl;
    
    return 0;
}