#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>

// Manual computation of Weibull NLL for verification
double compute_weibull_nll_manual(const std::vector<float>& data, float k, float lambda) {
    size_t n = data.size();
    double nll = 0.0;
    
    // Term 1: -n*log(k)
    nll -= n * std::log(k);
    
    // Term 2: n*k*log(λ)
    nll += n * k * std::log(lambda);
    
    // Term 3: -(k-1)*Σlog(x_i)
    double log_sum = 0.0;
    for (float x : data) {
        if (x > 0) {
            log_sum += std::log(x);
        }
    }
    nll -= (k - 1) * log_sum;
    
    // Term 4: Σ(x_i/λ)^k
    double power_sum = 0.0;
    for (float x : data) {
        power_sum += std::pow(x / lambda, k);
    }
    nll += power_sum;
    
    return nll;
}

// Alternative using log-likelihood then negating
double compute_weibull_ll_manual(const std::vector<float>& data, float k, float lambda) {
    size_t n = data.size();
    double ll = 0.0;
    
    // log L = n*log(k) - n*k*log(λ) + (k-1)*Σlog(x_i) - Σ(x_i/λ)^k
    ll += n * std::log(k);
    ll -= n * k * std::log(lambda);
    
    double log_sum = 0.0;
    for (float x : data) {
        if (x > 0) {
            log_sum += std::log(x);
        }
    }
    ll += (k - 1) * log_sum;
    
    double power_sum = 0.0;
    for (float x : data) {
        power_sum += std::pow(x / lambda, k);
    }
    ll -= power_sum;
    
    return -ll;  // Return negative log-likelihood
}

int main() {
    // Test data
    std::vector<float> data = {1.5, 2.3, 0.8, 3.1, 2.7};
    
    // Test parameters
    float k = 2.5;
    float lambda = 3.0;
    
    std::cout << std::fixed << std::setprecision(8);
    
    double nll1 = compute_weibull_nll_manual(data, k, lambda);
    double nll2 = compute_weibull_ll_manual(data, k, lambda);
    
    std::cout << "Test data: ";
    for (float x : data) std::cout << x << " ";
    std::cout << "\n";
    std::cout << "Parameters: k=" << k << ", λ=" << lambda << "\n\n";
    
    std::cout << "Method 1 (direct NLL): " << nll1 << "\n";
    std::cout << "Method 2 (via LL):     " << nll2 << "\n";
    std::cout << "Difference:            " << std::abs(nll1 - nll2) << "\n\n";
    
    // Check individual terms
    size_t n = data.size();
    double term1 = -(double)n * std::log(k);
    double term2 = (double)n * k * std::log(lambda);
    double log_sum = 0.0;
    for (float x : data) {
        if (x > 0) log_sum += std::log(x);
    }
    double term3 = -(k - 1) * log_sum;
    double power_sum = 0.0;
    for (float x : data) {
        power_sum += std::pow(x / lambda, k);
    }
    
    std::cout << "Individual terms:\n";
    std::cout << "  -n*log(k)         = " << term1 << "\n";
    std::cout << "  n*k*log(λ)        = " << term2 << "\n";
    std::cout << "  -(k-1)*Σlog(x_i)  = " << term3 << "\n";
    std::cout << "  Σ(x_i/λ)^k        = " << power_sum << "\n";
    std::cout << "  Total NLL         = " << (term1 + term2 + term3 + power_sum) << "\n";
    
    return 0;
}