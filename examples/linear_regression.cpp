#include "../include/autograd/autograd.hpp"
#include "../include/statmodels/regression/linear_regression.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace autograd;
using namespace statmodels;

// Generate synthetic linear data
void generate_linear_data(std::vector<std::vector<float>>& X_data,
                          std::vector<float>& y_data,
                          const std::vector<float>& true_coeffs,
                          float true_intercept,
                          size_t n_samples,
                          float noise_std = 0.1f) {
    std::mt19937 gen(42);
    std::normal_distribution<float> noise_dist(0.0f, noise_std);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    
    size_t n_features = true_coeffs.size();
    X_data.clear();
    y_data.clear();
    
    for (size_t i = 0; i < n_samples; ++i) {
        std::vector<float> x_row;
        float y = true_intercept;
        
        for (size_t j = 0; j < n_features; ++j) {
            float x = x_dist(gen);
            x_row.push_back(x);
            y += true_coeffs[j] * x;
        }
        
        y += noise_dist(gen);  // Add noise
        
        X_data.push_back(x_row);
        y_data.push_back(y);
    }
}

int main() {
    std::cout << "\n=== Linear Regression Example ===" << std::endl;
    std::cout << "Demonstrating linear regression with autograd\n" << std::endl;
    
    // True parameters
    std::vector<float> true_coeffs = {2.5f, -1.3f, 0.8f};
    float true_intercept = 1.0f;
    size_t n_samples = 100;
    size_t n_features = true_coeffs.size();
    
    std::cout << "True parameters:" << std::endl;
    std::cout << "  Coefficients: [";
    for (size_t i = 0; i < true_coeffs.size(); ++i) {
        std::cout << true_coeffs[i];
        if (i < true_coeffs.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Intercept: " << true_intercept << std::endl;
    std::cout << std::endl;
    
    // Generate synthetic data
    std::vector<std::vector<float>> X_data;
    std::vector<float> y_data;
    generate_linear_data(X_data, y_data, true_coeffs, true_intercept, n_samples);
    
    // Convert to tensors
    std::vector<float> X_flat;
    for (const auto& row : X_data) {
        for (float val : row) {
            X_flat.push_back(val);
        }
    }
    
    auto X = from_vector(X_flat, {static_cast<int>(n_samples), static_cast<int>(n_features)}, false);
    auto y = from_vector(y_data, {static_cast<int>(n_samples), 1}, false);
    
    // Create and fit model
    std::cout << "Fitting linear regression model..." << std::endl;
    LinearRegression model(true);  // fit_intercept = true
    
    // Fit the model
    model.fit(X, y, 1000, 0.01f, 1e-6, true);
    
    // Get fitted parameters
    auto coeffs = model.get_coefficients();
    auto intercept = model.get_intercept();
    
    std::cout << "\nFitted parameters:" << std::endl;
    std::cout << "  Coefficients: [";
    for (size_t i = 0; i < coeffs->size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << coeffs->data[i];
        if (i < coeffs->size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Intercept: " << std::fixed << std::setprecision(4) 
              << intercept->data[0] << std::endl;
    
    // Calculate R-squared score
    float r2_score = model.score(X, y);
    std::cout << "\nR-squared score: " << std::fixed << std::setprecision(4) 
              << r2_score << std::endl;
    
    // Test with regularization
    std::cout << "\n=== Testing with L2 Regularization ===" << std::endl;
    LinearRegression model_reg(true);
    model_reg.set_regularization(0.1f, "l2");
    
    model_reg.fit(X, y, 1000, 0.01f, 1e-6, false);
    
    auto coeffs_reg = model_reg.get_coefficients();
    auto intercept_reg = model_reg.get_intercept();
    
    std::cout << "Regularized coefficients: [";
    for (size_t i = 0; i < coeffs_reg->size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << coeffs_reg->data[i];
        if (i < coeffs_reg->size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Regularized intercept: " << std::fixed << std::setprecision(4) 
              << intercept_reg->data[0] << std::endl;
    
    float r2_score_reg = model_reg.score(X, y);
    std::cout << "R-squared score (regularized): " << std::fixed << std::setprecision(4) 
              << r2_score_reg << std::endl;
    
    // Make predictions on new data
    std::cout << "\n=== Making Predictions ===" << std::endl;
    std::vector<float> new_X_data = {1.0f, -0.5f, 0.3f};
    auto new_X = from_vector(new_X_data, {1, static_cast<int>(n_features)}, false);
    
    auto prediction = model.predict(new_X);
    
    // Calculate true value
    float true_value = true_intercept;
    for (size_t i = 0; i < n_features; ++i) {
        true_value += true_coeffs[i] * new_X_data[i];
    }
    
    std::cout << "Test input: [";
    for (size_t i = 0; i < new_X_data.size(); ++i) {
        std::cout << new_X_data[i];
        if (i < new_X_data.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "True value: " << std::fixed << std::setprecision(4) << true_value << std::endl;
    std::cout << "Predicted value: " << std::fixed << std::setprecision(4) 
              << prediction->data[0] << std::endl;
    std::cout << "Absolute error: " << std::fixed << std::setprecision(4) 
              << std::abs(prediction->data[0] - true_value) << std::endl;
    
    std::cout << "\nLinear regression example completed successfully!" << std::endl;
    
    return 0;
}