#pragma once

#include "tensor.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace autograd {
namespace optim {

// Base optimizer class
class Optimizer {
protected:
    std::vector<TensorPtr> parameters;
    
public:
    Optimizer(const std::vector<TensorPtr>& params) : parameters(params) {}
    virtual ~Optimizer() = default;
    
    virtual void step() = 0;
    virtual float get_lr() const = 0;
    virtual void set_lr(float new_lr) = 0;
    
    void zero_grad() {
        for (auto& p : parameters) {
            p->zero_grad();
        }
    }
};

// Stochastic Gradient Descent with Momentum
class SGD : public Optimizer {
private:
    float lr;
    float momentum;
    float weight_decay;
    std::vector<std::vector<float>> velocities;
    
public:
    SGD(const std::vector<TensorPtr>& params, 
        float learning_rate = 0.01f, 
        float momentum = 0.9f,
        float weight_decay = 0.0f)
        : Optimizer(params), lr(learning_rate), momentum(momentum), weight_decay(weight_decay) {
        
        for (const auto& p : parameters) {
            velocities.push_back(std::vector<float>(p->size(), 0));
        }
    }
    
    void step() override {
        #pragma omp parallel for
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& p = parameters[i];
            auto& v = velocities[i];
            
            #pragma omp simd
            for (size_t j = 0; j < p->size(); ++j) {
                float grad = p->grad[j];
                
                // Add weight decay (L2 regularization)
                if (weight_decay > 0) {
                    grad += weight_decay * p->data[j];
                }
                
                // Momentum update
                v[j] = momentum * v[j] - lr * grad;
                p->data[j] += v[j];
            }
        }
    }
    
    void set_lr(float new_lr) { lr = new_lr; }
    float get_lr() const { return lr; }
};

// Adam optimizer
class Adam : public Optimizer {
private:
    float lr;
    float beta1, beta2;
    float eps;
    float weight_decay;
    size_t t;  // timestep
    std::vector<std::vector<float>> m;  // First moment
    std::vector<std::vector<float>> v;  // Second moment
    
public:
    Adam(const std::vector<TensorPtr>& params,
         float learning_rate = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float eps = 1e-8f,
         float weight_decay = 0.0f)
        : Optimizer(params), lr(learning_rate), beta1(beta1), beta2(beta2), 
          eps(eps), weight_decay(weight_decay), t(0) {
        
        for (const auto& p : parameters) {
            m.push_back(std::vector<float>(p->size(), 0));
            v.push_back(std::vector<float>(p->size(), 0));
        }
    }
    
    void step() override {
        t++;
        
        // Bias correction
        float lr_t = lr * std::sqrt(1 - std::pow(beta2, t)) / (1 - std::pow(beta1, t));
        
        #pragma omp parallel for
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& p = parameters[i];
            auto& m_i = m[i];
            auto& v_i = v[i];
            
            #pragma omp simd
            for (size_t j = 0; j < p->size(); ++j) {
                float grad = p->grad[j];
                
                // Add weight decay
                if (weight_decay > 0) {
                    grad += weight_decay * p->data[j];
                }
                
                // Update biased first moment estimate
                m_i[j] = beta1 * m_i[j] + (1 - beta1) * grad;
                
                // Update biased second raw moment estimate
                v_i[j] = beta2 * v_i[j] + (1 - beta2) * grad * grad;
                
                // Update parameters
                p->data[j] -= lr_t * m_i[j] / (std::sqrt(v_i[j]) + eps);
            }
        }
    }
    
    void set_lr(float new_lr) { lr = new_lr; }
    float get_lr() const { return lr; }
};

// AdamW optimizer (Adam with decoupled weight decay)
class AdamW : public Optimizer {
private:
    float lr;
    float beta1, beta2;
    float eps;
    float weight_decay;
    size_t t;
    std::vector<std::vector<float>> m;
    std::vector<std::vector<float>> v;
    
public:
    AdamW(const std::vector<TensorPtr>& params,
          float learning_rate = 0.001f,
          float beta1 = 0.9f,
          float beta2 = 0.999f,
          float eps = 1e-8f,
          float weight_decay = 0.01f)
        : Optimizer(params), lr(learning_rate), beta1(beta1), beta2(beta2),
          eps(eps), weight_decay(weight_decay), t(0) {
        
        for (const auto& p : parameters) {
            m.push_back(std::vector<float>(p->size(), 0));
            v.push_back(std::vector<float>(p->size(), 0));
        }
    }
    
    void step() override {
        t++;
        
        // Bias correction
        float bias_correction1 = 1 - std::pow(beta1, t);
        float bias_correction2 = 1 - std::pow(beta2, t);
        float lr_t = lr * std::sqrt(bias_correction2) / bias_correction1;
        
        #pragma omp parallel for
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& p = parameters[i];
            auto& m_i = m[i];
            auto& v_i = v[i];
            
            #pragma omp simd
            for (size_t j = 0; j < p->size(); ++j) {
                float grad = p->grad[j];
                
                // Update moments
                m_i[j] = beta1 * m_i[j] + (1 - beta1) * grad;
                v_i[j] = beta2 * v_i[j] + (1 - beta2) * grad * grad;
                
                // Update parameters with decoupled weight decay
                p->data[j] -= lr_t * m_i[j] / (std::sqrt(v_i[j]) + eps) + lr * weight_decay * p->data[j];
            }
        }
    }
    
    void set_lr(float new_lr) { lr = new_lr; }
    float get_lr() const { return lr; }
};

// RMSprop optimizer
class RMSprop : public Optimizer {
private:
    float lr;
    float alpha;  // smoothing constant
    float eps;
    float weight_decay;
    std::vector<std::vector<float>> square_avg;
    
public:
    RMSprop(const std::vector<TensorPtr>& params,
            float learning_rate = 0.01f,
            float alpha = 0.99f,
            float eps = 1e-8f,
            float weight_decay = 0.0f)
        : Optimizer(params), lr(learning_rate), alpha(alpha), 
          eps(eps), weight_decay(weight_decay) {
        
        for (const auto& p : parameters) {
            square_avg.push_back(std::vector<float>(p->size(), 0));
        }
    }
    
    void step() override {
        #pragma omp parallel for
        for (size_t i = 0; i < parameters.size(); ++i) {
            auto& p = parameters[i];
            auto& sq_avg = square_avg[i];
            
            #pragma omp simd
            for (size_t j = 0; j < p->size(); ++j) {
                float grad = p->grad[j];
                
                // Add weight decay
                if (weight_decay > 0) {
                    grad += weight_decay * p->data[j];
                }
                
                // Update square average
                sq_avg[j] = alpha * sq_avg[j] + (1 - alpha) * grad * grad;
                
                // Update parameters
                p->data[j] -= lr * grad / (std::sqrt(sq_avg[j]) + eps);
            }
        }
    }
    
    void set_lr(float new_lr) { lr = new_lr; }
    float get_lr() const { return lr; }
};

// Learning rate schedulers
class LRScheduler {
protected:
    Optimizer* optimizer;
    float base_lr;
    
public:
    LRScheduler(Optimizer* opt, float base_learning_rate)
        : optimizer(opt), base_lr(base_learning_rate) {}
    
    virtual void step() = 0;
};

// Exponential decay scheduler
class ExponentialLR : public LRScheduler {
private:
    float gamma;
    size_t epoch;
    
public:
    ExponentialLR(Optimizer* opt, float gamma = 0.9f)
        : LRScheduler(opt, opt->get_lr()), gamma(gamma), epoch(0) {}
    
    void step() override {
        epoch++;
        float new_lr = base_lr * std::pow(gamma, epoch);
        static_cast<SGD*>(optimizer)->set_lr(new_lr);
    }
};

// Cosine annealing scheduler
class CosineAnnealingLR : public LRScheduler {
private:
    size_t T_max;
    float eta_min;
    size_t t;
    
public:
    CosineAnnealingLR(Optimizer* opt, size_t T_max, float eta_min = 0)
        : LRScheduler(opt, opt->get_lr()), T_max(T_max), eta_min(eta_min), t(0) {}
    
    void step() override {
        t++;
        float new_lr = eta_min + (base_lr - eta_min) * 
                      (1 + std::cos(M_PI * t / T_max)) / 2;
        static_cast<SGD*>(optimizer)->set_lr(new_lr);
    }
};

} // namespace optim
} // namespace autograd