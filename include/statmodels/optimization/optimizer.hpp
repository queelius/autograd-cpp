#pragma once

#include "../../autograd/autograd.hpp"
#include <functional>
#include <cmath>
#include <algorithm>

namespace statmodels {

// Learning rate schedule types
enum class LRSchedule {
    CONSTANT,       // Fixed learning rate
    LINEAR,         // Linear decay from initial to final
    EXPONENTIAL,    // Exponential decay
    COSINE,         // Cosine annealing
    STEP,          // Step decay at specific intervals
    POLYNOMIAL     // Polynomial decay
};

// Learning rate scheduler
class LRScheduler {
public:
    // Constructor with schedule type
    LRScheduler(LRSchedule schedule, float initial_lr, int max_iterations, 
                float final_lr = 1e-8f, float decay_rate = 0.9f, int step_size = 100)
        : schedule_(schedule), initial_lr_(initial_lr), max_iter_(max_iterations),
          final_lr_(final_lr), decay_rate_(decay_rate), step_size_(step_size) {
        
        // Ensure final_lr is not too small
        final_lr_ = std::max(final_lr_, 1e-10f);
    }
    
    // Get learning rate for current iteration
    float get_lr(int iteration) const {
        if (iteration >= max_iter_) {
            return final_lr_;
        }
        
        float lr = initial_lr_;
        float progress = static_cast<float>(iteration) / max_iter_;
        
        switch (schedule_) {
            case LRSchedule::CONSTANT:
                lr = initial_lr_;
                break;
                
            case LRSchedule::LINEAR:
                // Linear interpolation from initial to final
                lr = initial_lr_ * (1.0f - progress) + final_lr_ * progress;
                break;
                
            case LRSchedule::EXPONENTIAL:
                // Exponential decay: lr = initial_lr * decay_rate^(iter/step_size)
                lr = initial_lr_ * std::pow(decay_rate_, static_cast<float>(iteration) / step_size_);
                lr = std::max(lr, final_lr_);
                break;
                
            case LRSchedule::COSINE:
                // Cosine annealing from initial to final
                lr = final_lr_ + 0.5f * (initial_lr_ - final_lr_) * 
                     (1.0f + std::cos(M_PI * progress));
                break;
                
            case LRSchedule::STEP: {
                // Step decay every step_size iterations
                int n_steps = iteration / step_size_;
                lr = initial_lr_ * std::pow(decay_rate_, static_cast<float>(n_steps));
                lr = std::max(lr, final_lr_);
                break;
            }
                
            case LRSchedule::POLYNOMIAL:
                // Polynomial decay with power=2 (quadratic)
                lr = (initial_lr_ - final_lr_) * std::pow(1.0f - progress, 2.0f) + final_lr_;
                break;
        }
        
        return lr;
    }
    
    // Create a custom schedule from a function
    static std::function<float(int)> custom_schedule(
        std::function<float(int, int)> schedule_fn, int max_iterations) {
        return [schedule_fn, max_iterations](int iter) {
            return schedule_fn(iter, max_iterations);
        };
    }
    
private:
    LRSchedule schedule_;
    float initial_lr_;
    int max_iter_;
    float final_lr_;
    float decay_rate_;
    int step_size_;
};

// Simple gradient descent optimizer with scheduled learning rate
class ScheduledOptimizer {
public:
    ScheduledOptimizer(const LRScheduler& scheduler) : scheduler_(scheduler) {}
    
    // Perform one optimization step
    void step(std::vector<autograd::TensorPtr>& parameters, int iteration) {
        float lr = scheduler_.get_lr(iteration);
        
        for (auto& param : parameters) {
            for (size_t i = 0; i < param->size(); ++i) {
                // Simple gradient descent with scheduled learning rate
                param->data[i] -= lr * param->grad[i];
            }
        }
    }
    
    // Get current learning rate
    float get_current_lr(int iteration) const {
        return scheduler_.get_lr(iteration);
    }
    
private:
    LRScheduler scheduler_;
};

} // namespace statmodels