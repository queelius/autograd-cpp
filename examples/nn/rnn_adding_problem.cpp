#include <autograd/autograd.hpp>
#include <nn/recurrent/lstm.hpp>
#include <nn/recurrent/rnn.hpp>
#include <nn/core/module.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>

using namespace autograd;
using namespace autograd::nn;

// The Adding Problem: 
// Given a sequence of random numbers and a binary mask sequence,
// sum the two numbers indicated by the mask.
// This tests the RNN's ability to remember information over long sequences.

std::tuple<std::vector<std::vector<float>>, 
           std::vector<std::vector<float>>, 
           std::vector<float>>
generate_adding_problem_data(size_t n_samples, size_t seq_length) {
    std::vector<std::vector<float>> sequences;
    std::vector<std::vector<float>> masks;
    std::vector<float> targets;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> value_dist(0, 1);
    std::uniform_int_distribution<size_t> pos_dist(0, seq_length - 1);
    
    for (size_t i = 0; i < n_samples; ++i) {
        std::vector<float> seq(seq_length);
        std::vector<float> mask(seq_length, 0.0f);
        
        // Generate random sequence
        for (size_t t = 0; t < seq_length; ++t) {
            seq[t] = value_dist(gen);
        }
        
        // Pick two positions to mark
        size_t pos1 = pos_dist(gen);
        size_t pos2 = pos_dist(gen);
        while (pos2 == pos1) {
            pos2 = pos_dist(gen);
        }
        
        mask[pos1] = 1.0f;
        mask[pos2] = 1.0f;
        
        // Target is sum of marked values
        float target = seq[pos1] + seq[pos2];
        
        sequences.push_back(seq);
        masks.push_back(mask);
        targets.push_back(target);
    }
    
    return {sequences, masks, targets};
}

// Model for the adding problem
class AddingModel : public Module {
private:
    std::shared_ptr<LSTM> lstm_;
    Parameter W_out_;
    Parameter b_out_;
    size_t hidden_size_;
    
public:
    AddingModel(size_t input_size, size_t hidden_size, const std::string& name = "AddingModel")
        : Module(name), hidden_size_(hidden_size) {
        
        // LSTM to process sequences
        lstm_ = std::make_shared<LSTM>(input_size, hidden_size, 1, false);
        register_module("lstm", lstm_);
        
        // Output layer (regression to single value)
        float std = std::sqrt(2.0f / hidden_size);
        W_out_ = Parameter(mul(randn({1, static_cast<int>(hidden_size)}), std));
        b_out_ = Parameter(zeros({1}));
        
        register_parameter("W_out", W_out_);
        register_parameter("b_out", b_out_);
    }
    
    TensorPtr forward(TensorPtr input) override {
        // input shape: [seq_len, batch_size, 2] (value and mask concatenated)
        auto lstm_out = lstm_->forward(input);
        
        // Use final hidden state for prediction
        size_t seq_len = lstm_out->shape()[0];
        size_t batch_size = lstm_out->shape()[1];
        
        // Extract final hidden state
        auto final_hidden = std::make_shared<Tensor>(
            std::initializer_list<size_t>{batch_size, hidden_size_},
            lstm_out->requires_grad);
        
        size_t offset = (seq_len - 1) * batch_size * hidden_size_;
        for (size_t i = 0; i < batch_size * hidden_size_; ++i) {
            final_hidden->data[i] = lstm_out->data[offset + i];
        }
        
        // Project to single output
        auto output = std::make_shared<Tensor>(
            std::initializer_list<size_t>{batch_size, 1},
            final_hidden->requires_grad);
        
        for (size_t b = 0; b < batch_size; ++b) {
            float sum = b_out_->data[0];
            for (size_t h = 0; h < hidden_size_; ++h) {
                sum += final_hidden->data[b * hidden_size_ + h] * W_out_->data[h];
            }
            output->data[b] = sum;
        }
        
        return output;
    }
};

int main() {
    std::cout << "\n=== RNN Adding Problem ===" << std::endl;
    std::cout << "A classic test of RNN's ability to remember information over long sequences\n" << std::endl;
    
    // Problem parameters
    const size_t seq_length = 50;
    const size_t hidden_size = 128;
    const size_t n_train = 1000;
    const size_t n_test = 200;
    const size_t batch_size = 32;
    const size_t n_epochs = 100;
    const float learning_rate = 0.001f;
    
    std::cout << "Problem setup:" << std::endl;
    std::cout << "  Sequence length: " << seq_length << std::endl;
    std::cout << "  Training samples: " << n_train << std::endl;
    std::cout << "  Test samples: " << n_test << std::endl;
    
    // Generate data
    std::cout << "\nGenerating data..." << std::endl;
    auto [train_seq, train_mask, train_targets] = generate_adding_problem_data(n_train, seq_length);
    auto [test_seq, test_mask, test_targets] = generate_adding_problem_data(n_test, seq_length);
    
    // Create model
    auto model = std::make_shared<AddingModel>(2, hidden_size);  // 2 inputs: value and mask
    std::cout << "Model created with " << model->num_parameters() << " parameters" << std::endl;
    
    // Baseline MSE (always predicting 0.5)
    float baseline_mse = 0;
    for (float target : test_targets) {
        baseline_mse += (target - 0.5f) * (target - 0.5f);
    }
    baseline_mse /= n_test;
    std::cout << "Baseline MSE (predicting 0.5): " << std::fixed << std::setprecision(6) 
              << baseline_mse << std::endl;
    
    // Training
    std::cout << "\nTraining..." << std::endl;
    std::vector<float> train_losses, test_losses;
    
    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
        float epoch_loss = 0;
        int n_batches = 0;
        
        // Shuffle training data
        std::vector<size_t> indices(n_train);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));
        
        // Train in batches
        for (size_t b = 0; b < n_train; b += batch_size) {
            size_t actual_batch = std::min(batch_size, n_train - b);
            
            // Prepare batch
            auto input_batch = zeros({static_cast<int>(seq_length), 
                                     static_cast<int>(actual_batch), 2});
            auto target_batch = zeros({static_cast<int>(actual_batch), 1});
            
            for (size_t i = 0; i < actual_batch; ++i) {
                size_t idx = indices[b + i];
                target_batch->data[i] = train_targets[idx];
                
                for (size_t t = 0; t < seq_length; ++t) {
                    // Concatenate value and mask
                    input_batch->data[t * actual_batch * 2 + i * 2 + 0] = train_seq[idx][t];
                    input_batch->data[t * actual_batch * 2 + i * 2 + 1] = train_mask[idx][t];
                }
            }
            
            // Forward pass
            model->zero_grad();
            auto predictions = model->forward(input_batch);
            
            // MSE loss
            auto diff = sub(predictions, target_batch);
            auto loss = mean(mul(diff, diff));
            
            // Backward pass
            loss->backward();
            
            // Gradient clipping
            for (auto& param : model->parameters()) {
                float grad_norm = 0;
                for (size_t i = 0; i < param->size(); ++i) {
                    grad_norm += param->grad[i] * param->grad[i];
                }
                grad_norm = std::sqrt(grad_norm);
                
                if (grad_norm > 1.0f) {
                    for (size_t i = 0; i < param->size(); ++i) {
                        param->grad[i] /= grad_norm;
                    }
                }
            }
            
            // Update parameters
            for (auto& param : model->parameters()) {
                for (size_t i = 0; i < param->size(); ++i) {
                    param->data[i] -= learning_rate * param->grad[i];
                }
            }
            
            epoch_loss += loss->data[0];
            n_batches++;
        }
        
        // Test evaluation
        float test_loss = 0;
        for (size_t b = 0; b < n_test; b += batch_size) {
            size_t actual_batch = std::min(batch_size, n_test - b);
            
            auto input_batch = zeros({static_cast<int>(seq_length), 
                                     static_cast<int>(actual_batch), 2});
            auto target_batch = zeros({static_cast<int>(actual_batch), 1});
            
            for (size_t i = 0; i < actual_batch; ++i) {
                size_t idx = b + i;
                target_batch->data[i] = test_targets[idx];
                
                for (size_t t = 0; t < seq_length; ++t) {
                    input_batch->data[t * actual_batch * 2 + i * 2 + 0] = test_seq[idx][t];
                    input_batch->data[t * actual_batch * 2 + i * 2 + 1] = test_mask[idx][t];
                }
            }
            
            auto predictions = model->forward(input_batch);
            auto diff = sub(predictions, target_batch);
            auto loss = mean(mul(diff, diff));
            
            test_loss += loss->data[0] * actual_batch;
        }
        test_loss /= n_test;
        
        train_losses.push_back(epoch_loss / n_batches);
        test_losses.push_back(test_loss);
        
        if (epoch % 10 == 0 || epoch == n_epochs - 1) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                     << " | Train MSE: " << std::fixed << std::setprecision(6) << train_losses.back()
                     << " | Test MSE: " << test_losses.back();
            
            // Compare to baseline
            if (test_losses.back() < baseline_mse) {
                std::cout << " (better than baseline!)";
            }
            std::cout << std::endl;
        }
    }
    
    // Final evaluation with examples
    std::cout << "\n=== Final Evaluation ===" << std::endl;
    std::cout << "Showing predictions on a few test examples:\n" << std::endl;
    
    for (size_t i = 0; i < std::min(size_t(5), n_test); ++i) {
        // Prepare single sample
        auto input = zeros({static_cast<int>(seq_length), 1, 2});
        
        std::cout << "Example " << (i + 1) << ":" << std::endl;
        std::cout << "  Marked positions: ";
        float sum = 0;
        for (size_t t = 0; t < seq_length; ++t) {
            input->data[t * 2 + 0] = test_seq[i][t];
            input->data[t * 2 + 1] = test_mask[i][t];
            
            if (test_mask[i][t] > 0.5f) {
                std::cout << t << " (value=" << std::fixed << std::setprecision(3) 
                         << test_seq[i][t] << ") ";
                sum += test_seq[i][t];
            }
        }
        
        auto prediction = model->forward(input);
        
        std::cout << "\n  True sum: " << std::fixed << std::setprecision(4) << test_targets[i];
        std::cout << "\n  Predicted: " << prediction->data[0];
        std::cout << "\n  Error: " << std::abs(prediction->data[0] - test_targets[i]) << "\n" << std::endl;
    }
    
    std::cout << "Final test MSE: " << test_losses.back() << std::endl;
    std::cout << "Baseline MSE:   " << baseline_mse << std::endl;
    
    if (test_losses.back() < baseline_mse) {
        float improvement = (baseline_mse - test_losses.back()) / baseline_mse * 100;
        std::cout << "Model is " << std::fixed << std::setprecision(1) << improvement 
                  << "% better than baseline!" << std::endl;
    }
    
    return 0;
}