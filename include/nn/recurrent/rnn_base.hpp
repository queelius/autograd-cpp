#pragma once

#include "../core/module.hpp"
#include "../../autograd/tensor.hpp"
#include "../../autograd/ops.hpp"
#include <vector>
#include <memory>
#include <iostream>

namespace autograd {
namespace nn {

// Base class for all recurrent neural networks
class RNNBase : public Module {
protected:
    size_t input_size_;
    size_t hidden_size_;
    size_t num_layers_;
    bool batch_first_;
    float dropout_;
    bool bidirectional_;
    
    // Initialize hidden state with zeros
    TensorPtr init_hidden(size_t batch_size, size_t num_directions = 1) const {
        // For single layer, return 2D tensor [batch_size, hidden_size]
        // For multi-layer, would return 3D tensor [num_layers, batch_size, hidden_size]
        if (num_layers_ == 1 && num_directions == 1) {
            auto h = zeros({static_cast<int>(batch_size),
                           static_cast<int>(hidden_size_)});
            return h;
        } else {
            auto h = zeros({static_cast<int>(num_layers_ * num_directions), 
                           static_cast<int>(batch_size), 
                           static_cast<int>(hidden_size_)});
            return h;
        }
    }
    
public:
    RNNBase(size_t input_size, size_t hidden_size, size_t num_layers = 1,
            bool batch_first = false, float dropout = 0.0f, bool bidirectional = false,
            const std::string& name = "RNNBase")
        : Module(name), 
          input_size_(input_size), 
          hidden_size_(hidden_size),
          num_layers_(num_layers), 
          batch_first_(batch_first),
          dropout_(dropout), 
          bidirectional_(bidirectional) {}
    
    // Process a single time step
    virtual TensorPtr cell_forward(TensorPtr input, TensorPtr hidden) = 0;
    
    // Process entire sequence
    virtual std::pair<TensorPtr, TensorPtr> forward_sequence(
        TensorPtr input, TensorPtr hidden = nullptr) {
        
        // Input shape: [seq_len, batch_size, input_size] or [batch_size, seq_len, input_size]
        if (input->ndim() != 3) {
            throw std::runtime_error("RNN input must be 3D tensor");
        }
        
        size_t seq_len, batch_size;
        if (batch_first_) {
            batch_size = input->shape()[0];
            seq_len = input->shape()[1];
        } else {
            seq_len = input->shape()[0];
            batch_size = input->shape()[1];
        }
        
        // Initialize hidden state if not provided
        if (!hidden) {
            hidden = init_hidden(batch_size, bidirectional_ ? 2 : 1);
        }
        
        std::vector<TensorPtr> outputs;
        outputs.reserve(seq_len);
        
        // Process sequence
        for (size_t t = 0; t < seq_len; ++t) {
            // Extract time step
            TensorPtr x_t;
            if (batch_first_) {
                // Extract [:, t, :]
                x_t = slice_time_step(input, t, true);
            } else {
                // Extract [t, :, :]
                x_t = slice_time_step(input, t, false);
            }
            
            // Forward through cell
            hidden = cell_forward(x_t, hidden);
            outputs.push_back(hidden);
        }
        
        // Stack outputs along time dimension
        TensorPtr output = stack_outputs(outputs, batch_first_);
        
        return {output, hidden};
    }
    
    // Default forward returns both output and hidden state
    TensorPtr forward(TensorPtr input) override {
        auto [output, hidden] = forward_sequence(input);
        return output;
    }
    
protected:
    // Helper to slice a single time step from the input
    TensorPtr slice_time_step(TensorPtr input, size_t t, bool batch_first) const {
        size_t batch_size = batch_first ? input->shape()[0] : input->shape()[1];
        size_t input_size = input->shape()[2];
        
        auto slice = std::make_shared<Tensor>(
            std::initializer_list<size_t>{batch_size, input_size},
            input->requires_grad);
        
        if (batch_first) {
            // Copy from [:, t, :]
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < input_size; ++i) {
                    slice->data[b * input_size + i] = 
                        input->data[b * input->shape()[1] * input_size + t * input_size + i];
                }
            }
        } else {
            // Copy from [t, :, :]
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t i = 0; i < input_size; ++i) {
                    slice->data[b * input_size + i] = 
                        input->data[t * batch_size * input_size + b * input_size + i];
                }
            }
        }
        
        return slice;
    }
    
    // Helper to stack outputs into a sequence tensor
    TensorPtr stack_outputs(const std::vector<TensorPtr>& outputs, bool batch_first) const {
        if (outputs.empty()) return nullptr;
        
        size_t seq_len = outputs.size();
        size_t batch_size = outputs[0]->shape()[0];
        size_t hidden_size = outputs[0]->shape()[1];
        
        std::vector<size_t> shape;
        if (batch_first) {
            shape = {batch_size, seq_len, hidden_size};
        } else {
            shape = {seq_len, batch_size, hidden_size};
        }
        
        auto result = std::make_shared<Tensor>(shape, outputs[0]->requires_grad);
        
        for (size_t t = 0; t < seq_len; ++t) {
            auto& out_t = outputs[t];
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t h = 0; h < hidden_size; ++h) {
                    size_t idx;
                    if (batch_first) {
                        idx = b * seq_len * hidden_size + t * hidden_size + h;
                    } else {
                        idx = t * batch_size * hidden_size + b * hidden_size + h;
                    }
                    result->data[idx] = out_t->data[b * hidden_size + h];
                }
            }
        }
        
        return result;
    }
};

} // namespace nn
} // namespace autograd