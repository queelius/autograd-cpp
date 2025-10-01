#include <autograd/autograd.hpp>
#include <nn/recurrent/lstm.hpp>
#include <nn/core/module.hpp>
#include <tokenizers/base/tokenizer.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <iomanip>
#include <algorithm>

using namespace autograd;
using namespace autograd::nn;

// Character-level language model
class CharRNN : public Module {
private:
    std::shared_ptr<LSTM> lstm_;
    Parameter W_out_;
    Parameter b_out_;
    size_t vocab_size_;
    size_t hidden_size_;
    
public:
    CharRNN(size_t vocab_size, size_t hidden_size, size_t num_layers = 2)
        : Module("CharRNN"), vocab_size_(vocab_size), hidden_size_(hidden_size) {
        
        // LSTM layers
        lstm_ = std::make_shared<LSTM>(vocab_size, hidden_size, num_layers, false);
        register_module("lstm", lstm_);
        
        // Output projection
        float std = std::sqrt(2.0f / hidden_size);
        W_out_ = Parameter(mul(randn({static_cast<int>(vocab_size), static_cast<int>(hidden_size)}), std));
        b_out_ = Parameter(zeros({static_cast<int>(vocab_size)}));
        
        register_parameter("W_out", W_out_);
        register_parameter("b_out", b_out_);
    }
    
    TensorPtr forward(TensorPtr input) override {
        // input shape: [seq_len, batch_size, vocab_size] (one-hot encoded)
        auto lstm_out = lstm_->forward(input);
        
        // Project each time step to vocabulary
        size_t seq_len = lstm_out->shape()[0];
        size_t batch_size = lstm_out->shape()[1];
        
        auto output = zeros({static_cast<int>(seq_len), 
                            static_cast<int>(batch_size), 
                            static_cast<int>(vocab_size_)});
        
        for (size_t t = 0; t < seq_len; ++t) {
            // Extract hidden state at time t
            auto hidden_t = std::make_shared<Tensor>(
                std::initializer_list<size_t>{batch_size, hidden_size_},
                lstm_out->requires_grad);
            
            size_t offset = t * batch_size * hidden_size_;
            for (size_t i = 0; i < batch_size * hidden_size_; ++i) {
                hidden_t->data[i] = lstm_out->data[offset + i];
            }
            
            // Project to vocabulary
            auto logits = linear_forward(hidden_t, W_out_, b_out_);
            
            // Store in output
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t v = 0; v < vocab_size_; ++v) {
                    output->data[t * batch_size * vocab_size_ + b * vocab_size_ + v] = 
                        logits->data[b * vocab_size_ + v];
                }
            }
        }
        
        return output;
    }
    
    // Generate text given a seed
    std::string generate(const std::string& seed, size_t length, 
                        const std::unordered_map<char, int>& char_to_idx,
                        const std::unordered_map<int, char>& idx_to_char,
                        float temperature = 1.0f) {
        std::string result = seed;
        std::mt19937 gen(42);
        
        // Convert seed to indices
        std::vector<int> indices;
        for (char c : seed) {
            indices.push_back(char_to_idx.at(c));
        }
        
        // Generate character by character
        for (size_t i = 0; i < length; ++i) {
            // Prepare input (last few characters)
            size_t context_len = std::min(size_t(20), indices.size());
            auto input = zeros({static_cast<int>(context_len), 1, static_cast<int>(vocab_size_)});
            
            for (size_t t = 0; t < context_len; ++t) {
                int idx = indices[indices.size() - context_len + t];
                input->data[t * vocab_size_ + idx] = 1.0f;  // One-hot encoding
            }
            
            // Forward pass
            auto output = forward(input);
            
            // Get logits for last time step
            std::vector<float> logits(vocab_size_);
            size_t last_t = context_len - 1;
            for (size_t v = 0; v < vocab_size_; ++v) {
                logits[v] = output->data[last_t * vocab_size_ + v] / temperature;
            }
            
            // Softmax
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum = 0;
            for (auto& l : logits) {
                l = std::exp(l - max_logit);
                sum += l;
            }
            for (auto& l : logits) {
                l /= sum;
            }
            
            // Sample from distribution
            std::discrete_distribution<> dist(logits.begin(), logits.end());
            int next_idx = dist(gen);
            
            indices.push_back(next_idx);
            result += idx_to_char.at(next_idx);
        }
        
        return result;
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

// Cross-entropy loss for language modeling
TensorPtr cross_entropy_loss(TensorPtr logits, TensorPtr targets) {
    // Simplified cross-entropy
    size_t batch_size = logits->shape()[0];
    size_t vocab_size = logits->shape()[1];
    
    auto loss = zeros({1});
    float total_loss = 0;
    
    for (size_t b = 0; b < batch_size; ++b) {
        // Find max for numerical stability
        float max_logit = -1e9f;
        for (size_t v = 0; v < vocab_size; ++v) {
            max_logit = std::max(max_logit, logits->data[b * vocab_size + v]);
        }
        
        // Compute softmax and loss
        float sum_exp = 0;
        for (size_t v = 0; v < vocab_size; ++v) {
            sum_exp += std::exp(logits->data[b * vocab_size + v] - max_logit);
        }
        
        // Find target class
        int target_idx = -1;
        for (size_t v = 0; v < vocab_size; ++v) {
            if (targets->data[b * vocab_size + v] > 0.5f) {
                target_idx = v;
                break;
            }
        }
        
        if (target_idx >= 0) {
            total_loss -= (logits->data[b * vocab_size + target_idx] - max_logit - std::log(sum_exp));
        }
    }
    
    loss->data[0] = total_loss / batch_size;
    return loss;
}

// Download and prepare Shakespeare text
std::string get_shakespeare_text() {
    // For this example, we'll use a small hardcoded excerpt
    // In production, you would download from Project Gutenberg
    std::string text = R"(
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
)";
    
    // Convert to lowercase for simplicity
    std::transform(text.begin(), text.end(), text.begin(), ::tolower);
    return text;
}

int main() {
    std::cout << "\n=== RNN Text Generation Example ===" << std::endl;
    std::cout << "Training a character-level LSTM on Shakespeare text\n" << std::endl;
    
    // Get text data
    std::string text = get_shakespeare_text();
    std::cout << "Text length: " << text.length() << " characters" << std::endl;
    
    // Build vocabulary
    std::unordered_map<char, int> char_to_idx;
    std::unordered_map<int, char> idx_to_char;
    int idx = 0;
    
    for (char c : text) {
        if (char_to_idx.find(c) == char_to_idx.end()) {
            char_to_idx[c] = idx;
            idx_to_char[idx] = c;
            idx++;
        }
    }
    
    size_t vocab_size = char_to_idx.size();
    std::cout << "Vocabulary size: " << vocab_size << " unique characters" << std::endl;
    
    // Prepare training sequences
    size_t seq_length = 20;
    std::vector<std::vector<int>> sequences;
    std::vector<std::vector<int>> targets;
    
    for (size_t i = 0; i < text.length() - seq_length; ++i) {
        std::vector<int> seq, target;
        for (size_t j = 0; j < seq_length; ++j) {
            seq.push_back(char_to_idx[text[i + j]]);
            target.push_back(char_to_idx[text[i + j + 1]]);
        }
        sequences.push_back(seq);
        targets.push_back(target);
    }
    
    std::cout << "Number of training sequences: " << sequences.size() << std::endl;
    
    // Create model
    size_t hidden_size = 128;
    auto model = std::make_shared<CharRNN>(vocab_size, hidden_size, 2);
    std::cout << "\nModel created with " << model->num_parameters() << " parameters" << std::endl;
    
    // Training parameters
    size_t batch_size = 32;
    size_t n_epochs = 100;
    float learning_rate = 0.001f;
    
    std::cout << "\nTraining for " << n_epochs << " epochs..." << std::endl;
    
    // Training loop (simplified)
    for (size_t epoch = 0; epoch < n_epochs; ++epoch) {
        float epoch_loss = 0;
        size_t n_batches = 0;
        
        // Process in batches
        for (size_t b = 0; b < sequences.size(); b += batch_size) {
            size_t actual_batch_size = std::min(batch_size, sequences.size() - b);
            
            // Prepare batch data
            auto X_batch = zeros({static_cast<int>(seq_length), 
                                 static_cast<int>(actual_batch_size), 
                                 static_cast<int>(vocab_size)});
            auto y_batch = zeros({static_cast<int>(seq_length), 
                                 static_cast<int>(actual_batch_size), 
                                 static_cast<int>(vocab_size)});
            
            for (size_t i = 0; i < actual_batch_size; ++i) {
                for (size_t t = 0; t < seq_length; ++t) {
                    // One-hot encoding
                    X_batch->data[t * actual_batch_size * vocab_size + 
                                i * vocab_size + sequences[b + i][t]] = 1.0f;
                    y_batch->data[t * actual_batch_size * vocab_size + 
                                i * vocab_size + targets[b + i][t]] = 1.0f;
                }
            }
            
            // Forward pass
            model->zero_grad();
            auto output = model->forward(X_batch);
            
            // Calculate loss for each time step and average
            float batch_loss = 0;
            for (size_t t = 0; t < seq_length; ++t) {
                auto logits_t = std::make_shared<Tensor>(
                    std::initializer_list<size_t>{actual_batch_size, vocab_size}, true);
                auto targets_t = std::make_shared<Tensor>(
                    std::initializer_list<size_t>{actual_batch_size, vocab_size}, false);
                
                // Extract time step
                for (size_t i = 0; i < actual_batch_size * vocab_size; ++i) {
                    logits_t->data[i] = output->data[t * actual_batch_size * vocab_size + i];
                    targets_t->data[i] = y_batch->data[t * actual_batch_size * vocab_size + i];
                }
                
                auto loss = cross_entropy_loss(logits_t, targets_t);
                batch_loss += loss->data[0];
            }
            
            epoch_loss += batch_loss / seq_length;
            n_batches++;
            
            // Simple gradient descent update
            for (auto& param : model->parameters()) {
                for (size_t i = 0; i < param->size(); ++i) {
                    // Gradient clipping
                    param->grad[i] = std::max(-5.0f, std::min(5.0f, param->grad[i]));
                    param->data[i] -= learning_rate * param->grad[i];
                }
            }
        }
        
        if (epoch % 20 == 0) {
            std::cout << "Epoch " << std::setw(3) << epoch 
                     << " | Loss: " << std::fixed << std::setprecision(4) 
                     << (epoch_loss / n_batches) << std::endl;
            
            // Generate sample text
            std::string seed = "to be";
            std::string generated = model->generate(seed, 50, char_to_idx, idx_to_char, 0.8f);
            std::cout << "  Sample: \"" << generated << "\"" << std::endl;
        }
    }
    
    // Generate final samples
    std::cout << "\n=== Text Generation ===" << std::endl;
    
    std::vector<std::string> seeds = {"to be", "the ", "and ", "that "};
    std::vector<float> temperatures = {0.5f, 0.8f, 1.0f};
    
    for (const auto& seed : seeds) {
        std::cout << "\nSeed: \"" << seed << "\"" << std::endl;
        for (float temp : temperatures) {
            std::string generated = model->generate(seed, 100, char_to_idx, idx_to_char, temp);
            std::cout << "  T=" << temp << ": \"" << generated << "\"" << std::endl;
        }
    }
    
    std::cout << "\nTraining complete!" << std::endl;
    
    return 0;
}