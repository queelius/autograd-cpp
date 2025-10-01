#include <autograd/autograd.hpp>
#include <nn/recurrent/rnn.hpp>
#include <nn/core/module.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using namespace autograd;
using namespace autograd::nn;

// Generate sine wave data
std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
generate_sine_data(size_t n_samples, size_t seq_length, float dt = 0.1f) {
    std::vector<std::vector<float>> X, y;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> phase_dist(0, 2 * M_PI);
    std::uniform_real_distribution<float> freq_dist(0.5f, 2.0f);
    
    for (size_t i = 0; i < n_samples; ++i) {
        float phase = phase_dist(gen);
        float freq = freq_dist(gen);
        
        std::vector<float> sequence;
        for (size_t t = 0; t < seq_length + 1; ++t) {
            float value = std::sin(freq * t * dt + phase);
            sequence.push_back(value);
        }
        
        // Input is sequence[:-1], target is sequence[1:]
        X.push_back(std::vector<float>(sequence.begin(), sequence.end() - 1));
        y.push_back(std::vector<float>(sequence.begin() + 1, sequence.end()));
    }
    
    return {X, y};
}

// Simple model: RNN -> Linear layer for prediction
class SineWavePredictor : public Module {
private:
    std::shared_ptr<RNN> rnn_;
    Parameter W_out_;
    Parameter b_out_;
    size_t hidden_size_;
    
public:
    SineWavePredictor(size_t input_size, size_t hidden_size, size_t output_size = 1)
        : Module("SineWavePredictor"), hidden_size_(hidden_size) {
        
        // Create RNN layer
        rnn_ = std::make_shared<RNN>(input_size, hidden_size, 1, "tanh", false);
        register_module("rnn", rnn_);
        
        // Output projection layer
        float std = std::sqrt(2.0f / hidden_size);
        W_out_ = Parameter(mul(randn({static_cast<int>(output_size), static_cast<int>(hidden_size)}), std));
        b_out_ = Parameter(zeros({static_cast<int>(output_size)}));
        
        register_parameter("W_out", W_out_);
        register_parameter("b_out", b_out_);
    }
    
    TensorPtr forward(TensorPtr input) override {
        // input shape: [seq_len, batch_size, input_size]
        auto rnn_out = rnn_->forward(input);
        
        // Take the last time step output for prediction
        // rnn_out shape: [seq_len, batch_size, hidden_size]
        size_t seq_len = rnn_out->shape()[0];
        size_t batch_size = rnn_out->shape()[1];
        
        // Extract last time step [batch_size, hidden_size]
        auto last_hidden = std::make_shared<Tensor>(
            std::initializer_list<size_t>{batch_size, hidden_size_},
            rnn_out->requires_grad);
        
        size_t offset = (seq_len - 1) * batch_size * hidden_size_;
        for (size_t i = 0; i < batch_size * hidden_size_; ++i) {
            last_hidden->data[i] = rnn_out->data[offset + i];
        }
        
        // Project to output
        return linear_forward(last_hidden, W_out_, b_out_);
    }
    
    // Predict entire sequence
    TensorPtr forward_sequence(TensorPtr input) {
        // Process entire sequence through RNN
        auto rnn_out = rnn_->forward(input);
        
        // Project each time step to output
        size_t seq_len = rnn_out->shape()[0];
        size_t batch_size = rnn_out->shape()[1];
        
        auto output = std::make_shared<Tensor>(
            std::initializer_list<size_t>{seq_len, batch_size, 1},
            rnn_out->requires_grad);
        
        for (size_t t = 0; t < seq_len; ++t) {
            // Extract time step
            auto hidden_t = std::make_shared<Tensor>(
                std::initializer_list<size_t>{batch_size, hidden_size_},
                rnn_out->requires_grad);
            
            size_t offset = t * batch_size * hidden_size_;
            for (size_t i = 0; i < batch_size * hidden_size_; ++i) {
                hidden_t->data[i] = rnn_out->data[offset + i];
            }
            
            // Project to output
            auto out_t = linear_forward(hidden_t, W_out_, b_out_);
            
            // Store in output tensor
            for (size_t b = 0; b < batch_size; ++b) {
                output->data[t * batch_size + b] = out_t->data[b];
            }
        }
        
        return output;
    }
    
private:
    TensorPtr linear_forward(TensorPtr input, Parameter weight, Parameter bias) {
        size_t batch_size = input->shape()[0];
        size_t in_features = input->shape()[1];
        size_t out_features = weight->shape()[0];
        
        auto output = std::make_shared<Tensor>(
            std::initializer_list<size_t>{batch_size, out_features},
            input->requires_grad || weight->requires_grad);
        
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
        
        return output;
    }
};

// Mean squared error loss
TensorPtr mse_loss(TensorPtr pred, TensorPtr target) {
    auto diff = sub(pred, target);
    auto squared = mul(diff, diff);
    return mean(squared);
}

int main() {
    std::cout << "\n=== RNN Sine Wave Prediction Example ===" << std::endl;
    std::cout << "Training an RNN to predict the next value in a sine wave sequence\n" << std::endl;
    
    // Hyperparameters
    const size_t seq_length = 20;
    const size_t hidden_size = 32;
    const size_t n_train = 100;
    const size_t n_test = 20;
    const size_t batch_size = 10;
    const size_t n_epochs = 50;
    const float learning_rate = 0.01f;
    
    // Generate data
    std::cout << "Generating synthetic sine wave data..." << std::endl;
    auto [X_train, y_train] = generate_sine_data(n_train, seq_length);
    auto [X_test, y_test] = generate_sine_data(n_test, seq_length);
    
    std::cout << "  Training samples: " << n_train << std::endl;
    std::cout << "  Test samples: " << n_test << std::endl;
    std::cout << "  Sequence length: " << seq_length << std::endl;
    
    // Create model
    auto model = std::make_shared<SineWavePredictor>(1, hidden_size, 1);
    std::cout << "\nModel: " << model->to_string() << std::endl;
    std::cout << "Number of parameters: " << model->num_parameters() << std::endl;
    
    // Training loop
    std::cout << "\nTraining..." << std::endl;
    std::vector<float> train_losses, test_losses;
    
    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
        float epoch_loss = 0;
        
        // Process in batches
        for (size_t b = 0; b < n_train; b += batch_size) {
            size_t actual_batch_size = std::min(batch_size, n_train - b);
            
            // Prepare batch [seq_len, batch_size, 1]
            auto X_batch = zeros({static_cast<int>(seq_length), 
                                 static_cast<int>(actual_batch_size), 1});
            auto y_batch = zeros({static_cast<int>(seq_length), 
                                 static_cast<int>(actual_batch_size), 1});
            
            for (size_t i = 0; i < actual_batch_size; ++i) {
                for (size_t t = 0; t < seq_length; ++t) {
                    X_batch->data[t * actual_batch_size + i] = X_train[b + i][t];
                    y_batch->data[t * actual_batch_size + i] = y_train[b + i][t];
                }
            }
            
            // Forward pass
            model->zero_grad();
            auto predictions = model->forward_sequence(X_batch);
            auto loss = mse_loss(predictions, y_batch);
            
            // Backward pass
            loss->backward();
            
            // Update parameters (simple SGD)
            for (auto& param : model->parameters()) {
                for (size_t i = 0; i < param->size(); ++i) {
                    param->data[i] -= learning_rate * param->grad[i];
                }
            }
            
            epoch_loss += loss->data[0];
        }
        
        // Calculate test loss
        auto X_test_batch = zeros({static_cast<int>(seq_length), 
                                   static_cast<int>(n_test), 1});
        auto y_test_batch = zeros({static_cast<int>(seq_length), 
                                   static_cast<int>(n_test), 1});
        
        for (size_t i = 0; i < n_test; ++i) {
            for (size_t t = 0; t < seq_length; ++t) {
                X_test_batch->data[t * n_test + i] = X_test[i][t];
                y_test_batch->data[t * n_test + i] = y_test[i][t];
            }
        }
        
        auto test_pred = model->forward_sequence(X_test_batch);
        auto test_loss = mse_loss(test_pred, y_test_batch);
        
        train_losses.push_back(epoch_loss / (n_train / batch_size));
        test_losses.push_back(test_loss->data[0]);
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                     << " | Train Loss: " << std::fixed << std::setprecision(6) 
                     << train_losses.back()
                     << " | Test Loss: " << test_losses.back() << std::endl;
        }
    }
    
    // Test prediction quality
    std::cout << "\n=== Testing Prediction Quality ===" << std::endl;
    std::cout << "Generating a new sine wave and predicting next values..." << std::endl;
    
    // Generate a test sequence
    std::vector<float> test_seq;
    float test_phase = 0.5f;
    float test_freq = 1.0f;
    
    for (size_t t = 0; t < seq_length; ++t) {
        test_seq.push_back(std::sin(test_freq * t * 0.1f + test_phase));
    }
    
    std::cout << "\nLast 5 input values: ";
    for (size_t i = seq_length - 5; i < seq_length; ++i) {
        std::cout << std::fixed << std::setprecision(3) << test_seq[i] << " ";
    }
    
    // Predict next value
    auto test_input = zeros({static_cast<int>(seq_length), 1, 1});
    for (size_t t = 0; t < seq_length; ++t) {
        test_input->data[t] = test_seq[t];
    }
    
    auto prediction = model->forward_sequence(test_input);
    float predicted_next = prediction->data[seq_length - 1];  // Last time step prediction
    float actual_next = std::sin(test_freq * seq_length * 0.1f + test_phase);
    
    std::cout << "\n\nPredicted next value: " << std::fixed << std::setprecision(3) << predicted_next;
    std::cout << "\nActual next value:    " << actual_next;
    std::cout << "\nAbsolute error:       " << std::abs(predicted_next - actual_next) << std::endl;
    
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Final train loss: " << train_losses.back() << std::endl;
    std::cout << "Final test loss:  " << test_losses.back() << std::endl;
    
    return 0;
}