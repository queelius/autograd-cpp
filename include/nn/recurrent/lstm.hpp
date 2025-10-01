#pragma once

#include "rnn_base.hpp"
#include "../core/module.hpp"
#include "../../autograd/tensor.hpp"
#include "../../autograd/ops.hpp"
#include <tuple>
#include <cmath>

namespace autograd {
namespace nn {

// LSTM Cell implementation
class LSTMCell : public Module {
private:
    size_t input_size_;
    size_t hidden_size_;
    
    // Weight matrices
    Parameter W_ih_;  // Input-to-hidden weights for all gates
    Parameter W_hh_;  // Hidden-to-hidden weights for all gates
    Parameter b_ih_;  // Input-to-hidden bias
    Parameter b_hh_;  // Hidden-to-hidden bias
    
    // Gate computation helper
    TensorPtr compute_gates(TensorPtr input, TensorPtr hidden) {
        // Compute input @ W_ih + hidden @ W_hh + b_ih + b_hh
        auto i_gates = linear(input, W_ih_, b_ih_);
        auto h_gates = linear(hidden, W_hh_, b_hh_);
        return add(i_gates, h_gates);
    }
    
public:
    LSTMCell(size_t input_size, size_t hidden_size, const std::string& name = "LSTMCell")
        : Module(name), input_size_(input_size), hidden_size_(hidden_size) {
        
        // Initialize weights (4 gates: i, f, g, o)
        size_t gate_size = 4 * hidden_size;
        
        // Xavier initialization
        float std_ih = std::sqrt(2.0f / (input_size + hidden_size));
        float std_hh = std::sqrt(2.0f / (hidden_size + hidden_size));
        
        W_ih_ = Parameter(mul(randn({static_cast<int>(gate_size), static_cast<int>(input_size)}), std_ih));
        W_hh_ = Parameter(mul(randn({static_cast<int>(gate_size), static_cast<int>(hidden_size)}), std_hh));
        b_ih_ = Parameter(zeros({static_cast<int>(gate_size)}));
        b_hh_ = Parameter(zeros({static_cast<int>(gate_size)}));
        
        register_parameter("weight_ih", W_ih_);
        register_parameter("weight_hh", W_hh_);
        register_parameter("bias_ih", b_ih_);
        register_parameter("bias_hh", b_hh_);
    }
    
    // Forward pass returns (hidden, cell) states
    std::pair<TensorPtr, TensorPtr> forward_cell(TensorPtr input, TensorPtr hidden, TensorPtr cell) {
        auto gates = compute_gates(input, hidden);
        
        // Split gates into i, f, g, o
        size_t batch_size = input->shape()[0];
        
        auto i_gate = sigmoid(slice_gate(gates, 0, batch_size, hidden_size_));  // Input gate
        auto f_gate = sigmoid(slice_gate(gates, 1, batch_size, hidden_size_));  // Forget gate
        auto g_gate = tanh(slice_gate(gates, 2, batch_size, hidden_size_));     // Cell gate
        auto o_gate = sigmoid(slice_gate(gates, 3, batch_size, hidden_size_));  // Output gate
        
        // New cell state: c_t = f * c_{t-1} + i * g
        auto new_cell = add(mul(f_gate, cell), mul(i_gate, g_gate));
        
        // New hidden state: h_t = o * tanh(c_t)
        auto new_hidden = mul(o_gate, tanh(new_cell));
        
        return {new_hidden, new_cell};
    }
    
    TensorPtr forward(TensorPtr input) override {
        // For single cell, expect input and concatenated (hidden, cell) state
        throw std::runtime_error("Use forward_cell for LSTMCell");
    }
    
private:
    // Helper to slice a specific gate from the concatenated gates tensor
    TensorPtr slice_gate(TensorPtr gates, size_t gate_idx, size_t batch_size, size_t hidden_size) {
        auto gate = std::make_shared<Tensor>(
            std::initializer_list<size_t>{batch_size, hidden_size},
            gates->requires_grad);
        
        size_t offset = gate_idx * hidden_size;
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t h = 0; h < hidden_size; ++h) {
                gate->data[b * hidden_size + h] = 
                    gates->data[b * (4 * hidden_size) + offset + h];
            }
        }
        
        return gate;
    }
    
    // Simple linear transformation
    TensorPtr linear(TensorPtr input, Parameter weight, Parameter bias) {
        // Simplified matrix multiply for demonstration
        size_t batch_size = input->shape()[0];
        size_t in_features = input->shape()[1];
        size_t out_features = weight->shape()[0];
        
        auto output = std::make_shared<Tensor>(
            std::initializer_list<size_t>{batch_size, out_features},
            input->requires_grad || weight->requires_grad);
        
        // output = input @ weight^T + bias
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t o = 0; o < out_features; ++o) {
                float sum = bias->data[o];
                for (size_t i = 0; i < in_features; ++i) {
                    sum += input->data[b * in_features + i] * weight->data[o * in_features + i];
                }
                output->data[b * out_features + o] = sum;
            }
        }
        
        return output;
    }
};

// Full LSTM layer
class LSTM : public RNNBase {
private:
    std::vector<std::shared_ptr<LSTMCell>> cells_;
    
public:
    LSTM(size_t input_size, size_t hidden_size, size_t num_layers = 1,
         bool batch_first = false, float dropout = 0.0f, bool bidirectional = false,
         const std::string& name = "LSTM")
        : RNNBase(input_size, hidden_size, num_layers, batch_first, dropout, bidirectional, name) {
        
        // Create LSTM cells for each layer
        for (size_t layer = 0; layer < num_layers; ++layer) {
            size_t layer_input_size = (layer == 0) ? input_size : hidden_size;
            auto cell = std::make_shared<LSTMCell>(layer_input_size, hidden_size);
            cells_.push_back(cell);
            register_module("cell_" + std::to_string(layer), cell);
        }
    }
    
    // Process single time step through all layers
    TensorPtr cell_forward(TensorPtr input, TensorPtr hidden) override {
        // For LSTM, hidden contains both h and c states
        // This is a simplified version
        auto h = hidden;
        
        for (size_t layer = 0; layer < num_layers_; ++layer) {
            // In full implementation, would split hidden/cell states properly
            auto [new_h, new_c] = cells_[layer]->forward_cell(input, h, h);
            h = new_h;
            input = new_h;  // Output of this layer is input to next
        }
        
        return h;
    }
    
    // Initialize both hidden and cell states
    std::pair<TensorPtr, TensorPtr> init_hidden_cell(size_t batch_size, size_t num_directions = 1) const {
        auto h = init_hidden(batch_size, num_directions);
        auto c = init_hidden(batch_size, num_directions);  // Cell state same shape as hidden
        return {h, c};
    }
    
    std::string to_string() const override {
        return name_ + "(input_size=" + std::to_string(input_size_) +
               ", hidden_size=" + std::to_string(hidden_size_) +
               ", num_layers=" + std::to_string(num_layers_) +
               (bidirectional_ ? ", bidirectional=true" : "") + ")";
    }
};

} // namespace nn
} // namespace autograd