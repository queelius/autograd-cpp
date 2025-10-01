#include "../include/autograd/autograd.hpp"
#include "../include/statmodels/regression/logistic_regression.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

using namespace autograd;
using namespace statmodels;

// Generate synthetic binary classification data
void generate_binary_data(std::vector<std::vector<float>>& X_data,
                          std::vector<float>& y_data,
                          size_t n_samples,
                          size_t n_features = 2) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    X_data.clear();
    y_data.clear();
    
    // Generate two clusters
    for (size_t i = 0; i < n_samples; ++i) {
        std::vector<float> x_row;
        float label;
        
        if (i < n_samples / 2) {
            // Class 0: centered around (-1, -1)
            for (size_t j = 0; j < n_features; ++j) {
                x_row.push_back(dist(gen) - 1.0f);
            }
            label = 0.0f;
        } else {
            // Class 1: centered around (1, 1)
            for (size_t j = 0; j < n_features; ++j) {
                x_row.push_back(dist(gen) + 1.0f);
            }
            label = 1.0f;
        }
        
        X_data.push_back(x_row);
        y_data.push_back(label);
    }
}

// Generate linearly separable data with known coefficients
void generate_linear_separable_data(std::vector<std::vector<float>>& X_data,
                                   std::vector<float>& y_data,
                                   const std::vector<float>& true_coeffs,
                                   float true_intercept,
                                   size_t n_samples) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> x_dist(-3.0f, 3.0f);
    std::normal_distribution<float> noise_dist(0.0f, 0.1f);
    
    size_t n_features = true_coeffs.size();
    X_data.clear();
    y_data.clear();
    
    for (size_t i = 0; i < n_samples; ++i) {
        std::vector<float> x_row;
        float logit = true_intercept;
        
        for (size_t j = 0; j < n_features; ++j) {
            float x = x_dist(gen);
            x_row.push_back(x);
            logit += true_coeffs[j] * x;
        }
        
        // Add small noise to logit
        logit += noise_dist(gen);
        
        // Convert to probability and sample
        float prob = 1.0f / (1.0f + std::exp(-logit));
        std::bernoulli_distribution bern(prob);
        float label = bern(gen) ? 1.0f : 0.0f;
        
        X_data.push_back(x_row);
        y_data.push_back(label);
    }
}

int main() {
    std::cout << "\n=== Logistic Regression Example ===" << std::endl;
    std::cout << "Demonstrating logistic regression for binary classification\n" << std::endl;
    
    // Example 1: Simple two-cluster data
    std::cout << "Example 1: Two-cluster classification" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    
    size_t n_samples = 200;
    size_t n_features = 2;
    
    std::vector<std::vector<float>> X_data;
    std::vector<float> y_data;
    generate_binary_data(X_data, y_data, n_samples, n_features);
    
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
    std::cout << "Fitting logistic regression model..." << std::endl;
    LogisticRegression model(true);  // fit_intercept = true
    
    // Fit the model
    model.fit(X, y, 500, 0.1f, 1e-5, true);
    
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
    
    // Calculate accuracy
    float accuracy = model.score(X, y);
    std::cout << "Training accuracy: " << std::fixed << std::setprecision(2) 
              << (accuracy * 100) << "%" << std::endl;
    
    // Print decision boundary (for 2D case)
    if (n_features == 2) {
        std::cout << "Decision boundary: " << model.get_decision_boundary() << std::endl;
    }
    
    // Example 2: Data with known coefficients
    std::cout << "\n\nExample 2: Data with known coefficients" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    
    std::vector<float> true_coeffs = {1.5f, -2.0f, 0.8f};
    float true_intercept = 0.5f;
    n_features = true_coeffs.size();
    n_samples = 300;
    
    std::cout << "True parameters:" << std::endl;
    std::cout << "  Coefficients: [";
    for (size_t i = 0; i < true_coeffs.size(); ++i) {
        std::cout << true_coeffs[i];
        if (i < true_coeffs.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Intercept: " << true_intercept << std::endl;
    
    // Generate data
    generate_linear_separable_data(X_data, y_data, true_coeffs, true_intercept, n_samples);
    
    // Convert to tensors
    X_flat.clear();
    for (const auto& row : X_data) {
        for (float val : row) {
            X_flat.push_back(val);
        }
    }
    
    auto X2 = from_vector(X_flat, {static_cast<int>(n_samples), static_cast<int>(n_features)}, false);
    auto y2 = from_vector(y_data, {static_cast<int>(n_samples), 1}, false);
    
    // Create and fit model
    LogisticRegression model2(true);
    std::cout << "\nFitting model..." << std::endl;
    model2.fit(X2, y2, 1000, 0.1f, 1e-6, false);
    
    // Get fitted parameters
    auto coeffs2 = model2.get_coefficients();
    auto intercept2 = model2.get_intercept();
    
    std::cout << "Fitted parameters:" << std::endl;
    std::cout << "  Coefficients: [";
    for (size_t i = 0; i < coeffs2->size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << coeffs2->data[i];
        if (i < coeffs2->size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Intercept: " << std::fixed << std::setprecision(4) 
              << intercept2->data[0] << std::endl;
    
    float accuracy2 = model2.score(X2, y2);
    std::cout << "Training accuracy: " << std::fixed << std::setprecision(2) 
              << (accuracy2 * 100) << "%" << std::endl;
    
    // Test with regularization
    std::cout << "\n=== Testing with L2 Regularization ===" << std::endl;
    LogisticRegression model_reg(true);
    model_reg.set_regularization(0.1f, "l2");
    
    model_reg.fit(X2, y2, 1000, 0.1f, 1e-6, false);
    
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
    
    float accuracy_reg = model_reg.score(X2, y2);
    std::cout << "Training accuracy (regularized): " << std::fixed << std::setprecision(2) 
              << (accuracy_reg * 100) << "%" << std::endl;
    
    // Make predictions on new data
    std::cout << "\n=== Making Predictions ===" << std::endl;
    std::vector<std::vector<float>> test_points = {
        {1.0f, -0.5f, 0.3f},
        {-1.0f, 2.0f, -0.5f},
        {0.0f, 0.0f, 0.0f}
    };
    
    for (const auto& point : test_points) {
        auto test_X = from_vector(point, {1, static_cast<int>(n_features)}, false);
        auto prob = model2.predict_proba(test_X);
        auto pred = model2.predict(test_X);
        
        std::cout << "Input: [";
        for (size_t i = 0; i < point.size(); ++i) {
            std::cout << point[i];
            if (i < point.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "  P(y=1): " << std::fixed << std::setprecision(4) << prob->data[0] << std::endl;
        std::cout << "  Prediction: " << (pred->data[0] > 0.5f ? "Class 1" : "Class 0") << std::endl;
    }
    
    std::cout << "\nLogistic regression example completed successfully!" << std::endl;
    
    return 0;
}