#pragma once

#include "../autograd/tensor.hpp"
#include "../autograd/ops.hpp"
#include <cmath>
#include <vector>

namespace autograd {
namespace nn {

// ============================================================================
// Attention Mechanisms
// ============================================================================

// Scaled dot-product attention
inline TensorPtr attention(TensorPtr q, TensorPtr k, TensorPtr v, 
                          bool causal_mask = false, float dropout_p = 0.0f) {
    if (q->ndim() != 2 || k->ndim() != 2 || v->ndim() != 2) {
        throw std::runtime_error("Attention expects 2D tensors [seq_len, hidden_dim]");
    }
    
    size_t seq_len = q->shape()[0];
    size_t d_k = q->shape()[1];
    
    if (k->shape()[0] != seq_len || v->shape()[0] != seq_len) {
        throw std::runtime_error("Sequence length mismatch in attention");
    }
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    
    // Compute attention scores: Q @ K^T * scale
    auto scores = std::make_shared<Tensor>(std::initializer_list<size_t>{seq_len, seq_len}, 
                                           q->requires_grad || k->requires_grad);
    scores->is_leaf = false;
    
    // Compute Q @ K^T with scaling
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < seq_len; ++j) {
            float dot = 0;
            #pragma omp simd reduction(+:dot)
            for (size_t d = 0; d < d_k; ++d) {
                dot += q->data[i * d_k + d] * k->data[j * d_k + d];
            }
            scores->data[i * seq_len + j] = dot * scale;
            
            // Apply causal mask
            if (causal_mask && j > i) {
                scores->data[i * seq_len + j] = -1e9f;
            }
        }
    }
    
    // Softmax over last dimension
    auto attn_weights = std::make_shared<Tensor>(std::initializer_list<size_t>{seq_len, seq_len}, 
                                                 scores->requires_grad);
    attn_weights->is_leaf = false;
    
    #pragma omp parallel for
    for (size_t i = 0; i < seq_len; ++i) {
        // Find max for numerical stability
        float max_val = -1e9f;
        for (size_t j = 0; j < seq_len; ++j) {
            if (!causal_mask || j <= i) {
                max_val = std::max(max_val, scores->data[i * seq_len + j]);
            }
        }
        
        // Compute exp and sum
        float sum = 0;
        for (size_t j = 0; j < seq_len; ++j) {
            attn_weights->data[i * seq_len + j] = std::exp(scores->data[i * seq_len + j] - max_val);
            sum += attn_weights->data[i * seq_len + j];
        }
        
        // Normalize
        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < seq_len; ++j) {
            attn_weights->data[i * seq_len + j] *= inv_sum;
        }
    }
    
    // Apply attention to values: attn_weights @ V
    auto out = std::make_shared<Tensor>(std::initializer_list<size_t>{seq_len, d_k}, 
                                        q->requires_grad || k->requires_grad || v->requires_grad);
    out->children = {q, k, v};
    out->is_leaf = false;
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_k; ++j) {
            float sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t s = 0; s < seq_len; ++s) {
                sum += attn_weights->data[i * seq_len + s] * v->data[s * d_k + j];
            }
            out->data[i * d_k + j] = sum;
        }
    }
    
    // Store intermediates for backward pass
    auto scores_ptr = scores;
    auto weights_ptr = attn_weights;
    
    // Backward pass
    if (out->requires_grad) {
        out->backward_fn = [q, k, v, out, scores_ptr, weights_ptr, seq_len, d_k, scale, causal_mask]() {
            // Gradient w.r.t V: weights^T @ out_grad
            if (v->requires_grad) {
                #pragma omp parallel for collapse(2)
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < d_k; ++j) {
                        float sum = 0;
                        for (size_t s = 0; s < seq_len; ++s) {
                            sum += weights_ptr->data[s * seq_len + i] * out->grad[s * d_k + j];
                        }
                        v->grad[i * d_k + j] += sum;
                    }
                }
            }
            
            // Gradient w.r.t attention weights
            std::vector<float> d_weights(seq_len * seq_len, 0);
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    float sum = 0;
                    for (size_t d = 0; d < d_k; ++d) {
                        sum += out->grad[i * d_k + d] * v->data[j * d_k + d];
                    }
                    d_weights[i * seq_len + j] = sum;
                }
            }
            
            // Gradient through softmax
            std::vector<float> d_scores(seq_len * seq_len, 0);
            #pragma omp parallel for
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    float sum = 0;
                    for (size_t s = 0; s < seq_len; ++s) {
                        float kronecker = (j == s) ? 1.0f : 0.0f;
                        sum += d_weights[i * seq_len + s] * 
                               weights_ptr->data[i * seq_len + j] * 
                               (kronecker - weights_ptr->data[i * seq_len + s]);
                    }
                    d_scores[i * seq_len + j] = sum;
                }
            }
            
            // Gradient w.r.t Q and K
            if (q->requires_grad || k->requires_grad) {
                #pragma omp parallel for collapse(2)
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t d = 0; d < d_k; ++d) {
                        if (q->requires_grad) {
                            float q_grad = 0;
                            for (size_t j = 0; j < seq_len; ++j) {
                                q_grad += d_scores[i * seq_len + j] * k->data[j * d_k + d] * scale;
                            }
                            q->grad[i * d_k + d] += q_grad;
                        }
                        
                        if (k->requires_grad) {
                            float k_grad = 0;
                            for (size_t j = 0; j < seq_len; ++j) {
                                k_grad += d_scores[j * seq_len + i] * q->data[j * d_k + d] * scale;
                            }
                            k->grad[i * d_k + d] += k_grad;
                        }
                    }
                }
            }
        };
    }
    
    return out;
}

// ============================================================================
// Normalization
// ============================================================================

// Layer normalization
inline TensorPtr layer_norm(TensorPtr x, TensorPtr gamma, TensorPtr beta, float eps = 1e-5f) {
    if (x->ndim() != 2) {
        throw std::runtime_error("LayerNorm expects 2D input [batch, features]");
    }
    
    size_t batch = x->shape()[0];
    size_t features = x->shape()[1];
    
    if (gamma->size() != features || beta->size() != features) {
        throw std::runtime_error("Gamma and beta must have size equal to features");
    }
    
    auto out = std::make_shared<Tensor>(x->shape(), x->requires_grad || gamma->requires_grad || beta->requires_grad);
    out->children = {x, gamma, beta};
    out->is_leaf = false;
    
    // Store mean and variance for backward pass
    std::vector<float> means(batch), vars(batch);
    
    // Forward pass
    #pragma omp parallel for
    for (size_t i = 0; i < batch; ++i) {
        // Compute mean
        float mean = 0;
        #pragma omp simd reduction(+:mean)
        for (size_t j = 0; j < features; ++j) {
            mean += x->data[i * features + j];
        }
        mean /= features;
        means[i] = mean;
        
        // Compute variance
        float var = 0;
        #pragma omp simd reduction(+:var)
        for (size_t j = 0; j < features; ++j) {
            float diff = x->data[i * features + j] - mean;
            var += diff * diff;
        }
        var /= features;
        vars[i] = var;
        
        // Normalize and scale
        float std_inv = 1.0f / std::sqrt(var + eps);
        #pragma omp simd
        for (size_t j = 0; j < features; ++j) {
            float normalized = (x->data[i * features + j] - mean) * std_inv;
            out->data[i * features + j] = normalized * gamma->data[j] + beta->data[j];
        }
    }
    
    // Backward pass
    if (out->requires_grad) {
        out->backward_fn = [x, gamma, beta, out, means, vars, batch, features, eps]() {
            #pragma omp parallel for
            for (size_t i = 0; i < batch; ++i) {
                float mean = means[i];
                float var = vars[i];
                float std = std::sqrt(var + eps);
                float std_inv = 1.0f / std;
                
                // Intermediate gradients
                float d_var = 0;
                float d_mean = 0;
                
                for (size_t j = 0; j < features; ++j) {
                    float normalized = (x->data[i * features + j] - mean) * std_inv;
                    float d_out = out->grad[i * features + j];
                    
                    // Gradient for gamma and beta
                    if (gamma->requires_grad) {
                        #pragma omp atomic
                        gamma->grad[j] += d_out * normalized;
                    }
                    if (beta->requires_grad) {
                        #pragma omp atomic
                        beta->grad[j] += d_out;
                    }
                    
                    // Accumulate gradients for variance and mean
                    float d_norm = d_out * gamma->data[j];
                    d_var += d_norm * (x->data[i * features + j] - mean) * (-0.5f) * std_inv * std_inv * std_inv;
                    d_mean += d_norm * (-std_inv);
                }
                
                d_mean += d_var * (-2.0f) * 
                         std::accumulate(x->data + i * features, 
                                        x->data + (i + 1) * features, 0.0f,
                                        [mean](float acc, float val) { return acc + (val - mean); }) / features;
                
                // Gradient for x
                if (x->requires_grad) {
                    #pragma omp simd
                    for (size_t j = 0; j < features; ++j) {
                        float d_norm = out->grad[i * features + j] * gamma->data[j];
                        x->grad[i * features + j] += d_norm * std_inv + 
                                                     d_var * 2.0f * (x->data[i * features + j] - mean) / features +
                                                     d_mean / features;
                    }
                }
            }
        };
    }
    
    return out;
}

// ============================================================================
// Loss Functions
// ============================================================================

// Cross-entropy loss for classification
inline TensorPtr cross_entropy_loss(TensorPtr logits, const std::vector<size_t>& targets) {
    if (logits->ndim() != 2) {
        throw std::runtime_error("cross_entropy_loss expects 2D logits [batch, vocab]");
    }
    
    size_t batch = logits->shape()[0];
    size_t vocab = logits->shape()[1];
    
    if (targets.size() != batch) {
        throw std::runtime_error("Number of targets must match batch size");
    }
    
    auto loss = std::make_shared<Tensor>(std::initializer_list<size_t>{1}, logits->requires_grad);
    loss->children = {logits};
    loss->is_leaf = false;
    
    // Store probabilities for backward
    auto probs = std::make_shared<Tensor>(logits->shape(), false);
    
    float total_loss = 0;
    
    // Forward pass
    #pragma omp parallel for reduction(+:total_loss)
    for (size_t i = 0; i < batch; ++i) {
        // Find max for numerical stability
        float max_val = -1e9f;
        for (size_t j = 0; j < vocab; ++j) {
            max_val = std::max(max_val, logits->data[i * vocab + j]);
        }
        
        // Compute softmax
        float sum = 0;
        for (size_t j = 0; j < vocab; ++j) {
            probs->data[i * vocab + j] = std::exp(logits->data[i * vocab + j] - max_val);
            sum += probs->data[i * vocab + j];
        }
        
        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < vocab; ++j) {
            probs->data[i * vocab + j] *= inv_sum;
        }
        
        // Cross-entropy loss
        total_loss += -std::log(probs->data[i * vocab + targets[i]] + 1e-8f);
    }
    
    loss->data[0] = total_loss / batch;
    
    // Backward pass
    if (loss->requires_grad) {
        auto probs_ptr = probs;
        auto targets_copy = targets;
        
        loss->backward_fn = [logits, probs_ptr, targets_copy, batch, vocab]() {
            if (logits->requires_grad) {
                float scale = 1.0f / batch;
                
                #pragma omp parallel for collapse(2)
                for (size_t i = 0; i < batch; ++i) {
                    for (size_t j = 0; j < vocab; ++j) {
                        float grad = probs_ptr->data[i * vocab + j];
                        if (j == targets_copy[i]) {
                            grad -= 1.0f;
                        }
                        logits->grad[i * vocab + j] += grad * scale;
                    }
                }
            }
        };
    }
    
    return loss;
}

// MSE loss for regression
inline TensorPtr mse_loss(TensorPtr pred, TensorPtr target) {
    if (pred->shape() != target->shape()) {
        throw std::runtime_error("Shape mismatch in MSE loss");
    }
    
    auto loss = std::make_shared<Tensor>(std::initializer_list<size_t>{1}, 
                                         pred->requires_grad || target->requires_grad);
    loss->children = {pred, target};
    loss->is_leaf = false;
    
    // Forward pass
    float total = 0;
    #pragma omp parallel for simd reduction(+:total)
    for (size_t i = 0; i < pred->size(); ++i) {
        float diff = pred->data[i] - target->data[i];
        total += diff * diff;
    }
    loss->data[0] = total / pred->size();
    
    // Backward pass
    if (loss->requires_grad) {
        loss->backward_fn = [pred, target, loss]() {
            float scale = 2.0f / pred->size();
            
            if (pred->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < pred->size(); ++i) {
                    pred->grad[i] += scale * (pred->data[i] - target->data[i]) * loss->grad[0];
                }
            }
            
            if (target->requires_grad) {
                #pragma omp parallel for simd
                for (size_t i = 0; i < target->size(); ++i) {
                    target->grad[i] -= scale * (pred->data[i] - target->data[i]) * loss->grad[0];
                }
            }
        };
    }
    
    return loss;
}

} // namespace nn
} // namespace autograd