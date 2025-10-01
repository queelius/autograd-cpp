#pragma once

#include "rnn_base.hpp"
#include "../core/module.hpp"
#include "../../autograd/tensor.hpp"
#include "../../autograd/ops.hpp"
#include <cmath>
#include <iostream>

namespace autograd {
namespace nn {

// Simple RNN Cell
class RNNCell : public Module {
private:
    size_t input_size_;
    size_t hidden_size_;
    std::string activation_;
    
    Parameter W_ih_;  // Input-to-hidden weights
    Parameter W_hh_;  // Hidden-to-hidden weights
    Parameter b_ih_;  // Input-to-hidden bias
    Parameter b_hh_;  // Hidden-to-hidden bias
    
public:
    RNNCell(size_t input_size, size_t hidden_size, 
            const std::string& activation = "tanh",
            const std::string& name = "RNNCell")
        : Module(name), 
          input_size_(input_size), 
          hidden_size_(hidden_size),
          activation_(activation) {
        
        // Xavier initialization
        float std_ih = std::sqrt(2.0f / (input_size + hidden_size));
        float std_hh = std::sqrt(2.0f / (hidden_size + hidden_size));
        
        W_ih_ = Parameter(mul(randn({static_cast<int>(hidden_size), static_cast<int>(input_size)}), std_ih));
        W_hh_ = Parameter(mul(randn({static_cast<int>(hidden_size), static_cast<int>(hidden_size)}), std_hh));
        b_ih_ = Parameter(zeros({static_cast<int>(hidden_size)}));
        b_hh_ = Parameter(zeros({static_cast<int>(hidden_size)}));
        
        register_parameter("weight_ih", W_ih_);
        register_parameter("weight_hh", W_hh_);
        register_parameter("bias_ih", b_ih_);
        register_parameter("bias_hh", b_hh_);
    }
    
    TensorPtr forward_cell(TensorPtr input, TensorPtr hidden) {
        // h_t = activation(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
        auto i_h = linear(input, W_ih_, b_ih_);
        auto h_h = linear(hidden, W_hh_, b_hh_);
        auto pre_activation = add(i_h, h_h);
        
        // Apply activation
        if (activation_ == "tanh") {
            return tanh(pre_activation);
        } else if (activation_ == "relu") {
            return relu(pre_activation);
        } else {
            return pre_activation;  // Linear activation
        }
    }
    
    TensorPtr forward(TensorPtr input) override {
        throw std::runtime_error("Use forward_cell for RNNCell");
    }
    
private:
    // Simple linear transformation (input @ weight^T + bias)
    TensorPtr linear(TensorPtr input, Parameter weight, Parameter bias) {
        size_t batch_size = input->shape()[0];
        size_t in_features = input->shape()[1];
        size_t out_features = weight->shape()[0];
        
        auto output = std::make_shared<Tensor>(
            std::initializer_list<size_t>{batch_size, out_features},
            input->requires_grad || weight->requires_grad);
        
        // Matrix multiply and add bias
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t o = 0; o < out_features; ++o) {
                float sum = bias->data[o];
                for (size_t i = 0; i < in_features; ++i) {
                    sum += input->data[b * in_features + i] * 
                           weight->data[o * in_features + i];
                }
                output->data[b * out_features + o] = sum;
            }
        }
        
        // Set up backward pass
        if (output->requires_grad) {
            output->children = {input, weight, bias};
            output->is_leaf = false;
            
            output->backward_fn = [input, weight, bias, output, batch_size, in_features, out_features]() {
                // Gradient w.r.t input
                if (input->requires_grad) {
                    for (size_t b = 0; b < batch_size; ++b) {
                        for (size_t i = 0; i < in_features; ++i) {
                            float grad = 0;
                            for (size_t o = 0; o < out_features; ++o) {
                                grad += output->grad[b * out_features + o] * 
                                       weight->data[o * in_features + i];
                            }
                            input->grad[b * in_features + i] += grad;
                        }
                    }
                }
                
                // Gradient w.r.t weight
                if (weight->requires_grad) {
                    for (size_t o = 0; o < out_features; ++o) {
                        for (size_t i = 0; i < in_features; ++i) {
                            float grad = 0;
                            for (size_t b = 0; b < batch_size; ++b) {
                                grad += output->grad[b * out_features + o] * 
                                       input->data[b * in_features + i];
                            }
                            weight->grad[o * in_features + i] += grad;
                        }
                    }
                }
                
                // Gradient w.r.t bias
                if (bias->requires_grad) {
                    for (size_t o = 0; o < out_features; ++o) {
                        float grad = 0;
                        for (size_t b = 0; b < batch_size; ++b) {
                            grad += output->grad[b * out_features + o];
                        }
                        bias->grad[o] += grad;
                    }
                }
            };
        }
        
        return output;
    }
};

// Full RNN layer
class RNN : public RNNBase {
private:
    std::vector<std::shared_ptr<RNNCell>> cells_;
    std::string activation_;
    
public:
    RNN(size_t input_size, size_t hidden_size, size_t num_layers = 1,
        const std::string& activation = "tanh",
        bool batch_first = false, float dropout = 0.0f, bool bidirectional = false,
        const std::string& name = "RNN")
        : RNNBase(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, name),
          activation_(activation) {
        
        // Create RNN cells for each layer
        for (size_t layer = 0; layer < num_layers; ++layer) {
            size_t layer_input_size = (layer == 0) ? input_size : hidden_size;
            auto cell = std::make_shared<RNNCell>(layer_input_size, hidden_size, activation);
            cells_.push_back(cell);
            register_module("cell_" + std::to_string(layer), cell);
        }
    }
    
    // Process single time step through all layers
    TensorPtr cell_forward(TensorPtr input, TensorPtr hidden) override {
        auto h = hidden;
        
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            // Extract hidden state for this layer
            // In full implementation would properly slice multi-layer hidden states
            h = cells_[layer]->forward_cell(input, h);
            input = h;  // Output of this layer is input to next
        }
        
        return h;
    }
    
    std::string to_string() const override {
        return name_ + "(input_size=" + std::to_string(input_size_) +
               ", hidden_size=" + std::to_string(hidden_size_) +
               ", num_layers=" + std::to_string(num_layers_) +
               ", activation=" + activation_ +
               (bidirectional_ ? ", bidirectional=true" : "") + ")";
    }
};

} // namespace nn
} // namespace autograd