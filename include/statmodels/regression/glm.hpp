#pragma once

#include "../base/regression.hpp"
#include "../../autograd/autograd.hpp"
#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <stdexcept>

namespace statmodels {

using namespace autograd;

// Link functions for GLMs
class LinkFunction {
public:
    virtual ~LinkFunction() = default;
    virtual TensorPtr link(TensorPtr mu) = 0;        // g(mu)
    virtual TensorPtr inverse(TensorPtr eta) = 0;    // g^-1(eta)
    virtual std::string name() const = 0;
};

// Identity link: g(mu) = mu
class IdentityLink : public LinkFunction {
public:
    TensorPtr link(TensorPtr mu) override {
        return mu;
    }
    
    TensorPtr inverse(TensorPtr eta) override {
        return eta;
    }
    
    std::string name() const override {
        return "identity";
    }
};

// Logit link: g(mu) = log(mu / (1 - mu))
class LogitLink : public LinkFunction {
public:
    TensorPtr link(TensorPtr mu) override {
        // logit = log(mu / (1 - mu))
        auto one = from_vector({1.0f}, {1}, false);
        auto odds = div(mu, sub(one, mu));
        return log(odds);
    }
    
    TensorPtr inverse(TensorPtr eta) override {
        // sigmoid = 1 / (1 + exp(-eta))
        return sigmoid(eta);
    }
    
    std::string name() const override {
        return "logit";
    }
};

// Log link: g(mu) = log(mu)
class LogLink : public LinkFunction {
public:
    TensorPtr link(TensorPtr mu) override {
        return log(mu);
    }
    
    TensorPtr inverse(TensorPtr eta) override {
        return exp(eta);
    }
    
    std::string name() const override {
        return "log";
    }
};

// Inverse link: g(mu) = 1/mu
class InverseLink : public LinkFunction {
public:
    TensorPtr link(TensorPtr mu) override {
        auto one = from_vector({1.0f}, {1}, false);
        return div(one, mu);
    }
    
    TensorPtr inverse(TensorPtr eta) override {
        auto one = from_vector({1.0f}, {1}, false);
        return div(one, eta);
    }
    
    std::string name() const override {
        return "inverse";
    }
};

// Distribution families for GLMs
class Family {
public:
    virtual ~Family() = default;
    virtual TensorPtr variance(TensorPtr mu) = 0;           // V(mu)
    virtual TensorPtr deviance(TensorPtr y, TensorPtr mu) = 0;  // Unit deviance
    virtual LinkFunction* canonical_link() = 0;             // Default link function
    virtual std::string name() const = 0;
};

// Gaussian family
class GaussianFamily : public Family {
private:
    IdentityLink identity_link_;
    
public:
    TensorPtr variance(TensorPtr mu) override {
        // V(mu) = 1 for Gaussian
        return ones_like(mu);
    }
    
    TensorPtr deviance(TensorPtr y, TensorPtr mu) override {
        // Deviance = (y - mu)^2
        auto diff = sub(y, mu);
        return mul(diff, diff);
    }
    
    LinkFunction* canonical_link() override {
        return &identity_link_;
    }
    
    std::string name() const override {
        return "Gaussian";
    }
};

// Binomial family
class BinomialFamily : public Family {
private:
    LogitLink logit_link_;
    
public:
    TensorPtr variance(TensorPtr mu) override {
        // V(mu) = mu * (1 - mu)
        auto one = from_vector({1.0f}, {1}, false);
        return mul(mu, sub(one, mu));
    }
    
    TensorPtr deviance(TensorPtr y, TensorPtr mu) override {
        // Deviance = 2 * (y * log(y/mu) + (1-y) * log((1-y)/(1-mu)))
        float epsilon = 1e-7f;
        auto eps = from_vector({epsilon}, {1}, false);
        auto one = from_vector({1.0f}, {1}, false);
        auto two = from_vector({2.0f}, {1}, false);
        
        // Clip mu to avoid numerical issues
        auto mu_clipped = maximum(minimum(mu, sub(one, eps)), eps);
        
        // Handle y = 0 and y = 1 cases
        auto term1 = mul(y, log(maximum(div(y, mu_clipped), eps)));
        auto term2 = mul(sub(one, y), log(maximum(div(sub(one, y), sub(one, mu_clipped)), eps)));
        
        return mul(two, add(term1, term2));
    }
    
    LinkFunction* canonical_link() override {
        return &logit_link_;
    }
    
    std::string name() const override {
        return "Binomial";
    }
};

// Poisson family
class PoissonFamily : public Family {
private:
    LogLink log_link_;
    
public:
    TensorPtr variance(TensorPtr mu) override {
        // V(mu) = mu
        return mu;
    }
    
    TensorPtr deviance(TensorPtr y, TensorPtr mu) override {
        // Deviance = 2 * (y * log(y/mu) - (y - mu))
        float epsilon = 1e-7f;
        auto eps = from_vector({epsilon}, {1}, false);
        auto two = from_vector({2.0f}, {1}, false);
        
        auto mu_clipped = maximum(mu, eps);
        auto y_clipped = maximum(y, eps);
        
        auto term1 = mul(y, log(div(y_clipped, mu_clipped)));
        auto term2 = sub(y, mu);
        
        return mul(two, sub(term1, term2));
    }
    
    LinkFunction* canonical_link() override {
        return &log_link_;
    }
    
    std::string name() const override {
        return "Poisson";
    }
};

// Generalized Linear Model
class GLM : public RegressionModel {
private:
    TensorPtr beta_;           // Coefficients
    TensorPtr intercept_;      // Intercept term
    bool fit_intercept_;       // Whether to fit an intercept
    Family* family_;           // Distribution family
    LinkFunction* link_;       // Link function
    
    // Family instances (owned by GLM)
    std::unique_ptr<GaussianFamily> gaussian_family_;
    std::unique_ptr<BinomialFamily> binomial_family_;
    std::unique_ptr<PoissonFamily> poisson_family_;
    
    // Link instances (owned by GLM)
    std::unique_ptr<IdentityLink> identity_link_;
    std::unique_ptr<LogitLink> logit_link_;
    std::unique_ptr<LogLink> log_link_;
    std::unique_ptr<InverseLink> inverse_link_;
    
public:
    GLM(const std::string& family = "gaussian", 
        const std::string& link = "canonical",
        bool fit_intercept = true) 
        : fit_intercept_(fit_intercept) {
        
        // Initialize families
        gaussian_family_ = std::make_unique<GaussianFamily>();
        binomial_family_ = std::make_unique<BinomialFamily>();
        poisson_family_ = std::make_unique<PoissonFamily>();
        
        // Initialize links
        identity_link_ = std::make_unique<IdentityLink>();
        logit_link_ = std::make_unique<LogitLink>();
        log_link_ = std::make_unique<LogLink>();
        inverse_link_ = std::make_unique<InverseLink>();
        
        // Set family
        if (family == "gaussian") {
            family_ = gaussian_family_.get();
        } else if (family == "binomial") {
            family_ = binomial_family_.get();
        } else if (family == "poisson") {
            family_ = poisson_family_.get();
        } else {
            throw std::invalid_argument("Unknown family: " + family);
        }
        
        // Set link function
        if (link == "canonical") {
            link_ = family_->canonical_link();
        } else if (link == "identity") {
            link_ = identity_link_.get();
        } else if (link == "logit") {
            link_ = logit_link_.get();
        } else if (link == "log") {
            link_ = log_link_.get();
        } else if (link == "inverse") {
            link_ = inverse_link_.get();
        } else {
            throw std::invalid_argument("Unknown link: " + link);
        }
    }
    
    // Initialize parameters based on input dimensions
    void initialize_parameters(size_t n_features) {
        // Initialize with small random values
        std::vector<float> beta_init(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            beta_init[i] = 0.01f * (static_cast<float>(rand()) / RAND_MAX - 0.5f);
        }
        beta_ = from_vector(beta_init, {static_cast<int>(n_features), 1}, true);
        
        if (fit_intercept_) {
            intercept_ = from_vector({0.0f}, {1}, true);
        }
    }
    
    // Compute linear predictor (eta = X @ beta + intercept)
    TensorPtr compute_linear_predictor(TensorPtr X) {
        if (!beta_) {
            throw std::runtime_error("Model not fitted. Call fit() first.");
        }
        
        auto eta = matmul(X, beta_);
        
        if (fit_intercept_ && intercept_) {
            eta = add(eta, intercept_);
        }
        
        return eta;
    }
    
    // Predict mean response
    TensorPtr predict(TensorPtr X) override {
        auto eta = compute_linear_predictor(X);
        return link_->inverse(eta);
    }
    
    // Fit using Iteratively Reweighted Least Squares (IRLS)
    void fit(TensorPtr X, TensorPtr y,
            int max_iterations = 25,
            float learning_rate = 1.0f,  // Not used in IRLS
            float tolerance = 1e-5,
            bool verbose = false) override {
        
        size_t n_samples = X->shape[0];
        size_t n_features = X->shape[1];
        
        // Initialize parameters if not already done
        if (!beta_) {
            initialize_parameters(n_features);
        }
        
        float prev_deviance = std::numeric_limits<float>::max();
        
        // IRLS iterations
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Compute linear predictor
            auto eta = compute_linear_predictor(X);
            
            // Compute mean response
            auto mu = link_->inverse(eta);
            
            // Compute variance
            auto variance = family_->variance(mu);
            
            // Compute working weights: W = 1 / (V(mu) * (g'(mu))^2)
            // For simplicity, using W = 1 / V(mu) (assumes canonical link)
            auto weights = div(ones_like(variance), variance);
            
            // Compute working response: z = eta + (y - mu) * g'(mu)
            // For simplicity, using z = eta + (y - mu) / V(mu)
            auto residuals = sub(y, mu);
            auto weighted_residuals = div(residuals, variance);
            auto z = add(eta, weighted_residuals);
            
            // Weighted least squares step
            // This would require solving (X^T W X) beta = X^T W z
            // For gradient descent approximation:
            
            // Compute gradient: X^T @ (weighted_residuals)
            auto gradient = matmul(transpose(X), weighted_residuals);
            
            // Update parameters
            gradient->backward();
            float grad_norm = compute_gradient_norm();
            
            // Simple gradient step (approximation of IRLS)
            apply_gradient_step(0.1f);  // Using fixed small learning rate
            
            // Compute deviance
            auto unit_deviances = family_->deviance(y, mu);
            float deviance = sum(unit_deviances)->data[0];
            
            // Check convergence
            if (std::abs(deviance - prev_deviance) < tolerance) {
                if (verbose) {
                    std::cout << "\nConverged at iteration " << iter << std::endl;
                    std::cout << "Final deviance: " << deviance << std::endl;
                }
                break;
            }
            
            prev_deviance = deviance;
            zero_grad();
            
            if (verbose && (iter % 5 == 0 || iter == max_iterations - 1)) {
                std::cout << "Iteration " << iter 
                         << ": deviance = " << std::fixed << std::setprecision(6) << deviance
                         << ", |grad| = " << std::scientific << std::setprecision(4) << grad_norm
                         << std::endl;
            }
        }
    }
    
    // Get model parameters
    std::vector<TensorPtr> get_parameters() override {
        std::vector<TensorPtr> params;
        if (beta_) params.push_back(beta_);
        if (fit_intercept_ && intercept_) params.push_back(intercept_);
        return params;
    }
    
    // Zero gradients
    void zero_grad() override {
        if (beta_) beta_->zero_grad();
        if (intercept_) intercept_->zero_grad();
    }
    
    // Get model name
    std::string get_name() const override {
        return "GLM(" + family_->name() + ", " + link_->name() + ")";
    }
    
    // Get coefficients
    TensorPtr get_coefficients() override {
        return beta_;
    }
    
    // Get intercept
    TensorPtr get_intercept() override {
        if (!fit_intercept_) {
            return from_vector({0.0f}, {1}, false);
        }
        return intercept_;
    }
    
    // Print model summary
    void summary() const {
        std::cout << "\n=== Generalized Linear Model ===" << std::endl;
        std::cout << "Family: " << family_->name() << std::endl;
        std::cout << "Link: " << link_->name() << std::endl;
        
        if (beta_) {
            std::cout << "Coefficients shape: [" << beta_->shape[0] << ", " << beta_->shape[1] << "]" << std::endl;
            std::cout << "Coefficients: ";
            for (size_t i = 0; i < beta_->size(); ++i) {
                std::cout << std::fixed << std::setprecision(4) << beta_->data[i];
                if (i < beta_->size() - 1) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        if (fit_intercept_ && intercept_) {
            std::cout << "Intercept: " << std::fixed << std::setprecision(4) << intercept_->data[0] << std::endl;
        }
        
        if (regularization_alpha_ > 0) {
            std::cout << "Regularization: " << regularization_penalty_ 
                     << " (alpha=" << regularization_alpha_ << ")" << std::endl;
        }
    }
};

} // namespace statmodels