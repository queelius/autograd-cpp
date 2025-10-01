#pragma once

#include "../../autograd/tensor.hpp"
#include "../../autograd/ops.hpp"
#include <cmath>
#include <vector>
#include <algorithm>

namespace autograd {
namespace nn {

// Scaled dot-product attention
inline TensorPtr scaled_dot_product_attention(
    TensorPtr q, TensorPtr k, TensorPtr v, 
    bool causal_mask = false, 
    float dropout_p = 0.0f) {
    
    if (q->ndim() != 2 || k->ndim() != 2 || v->ndim() != 2) {
        throw std::runtime_error("Attention expects 2D tensors [seq_len, hidden_dim]");
    }
    
    size_t seq_len = q->shape()[0];
    size_t d_k = q->shape()[1];
    size_t d_v = v->shape()[1];
    
    if (k->shape()[0] != seq_len || v->shape()[0] != seq_len) {
        throw std::runtime_error("Sequence length mismatch in attention");
    }
    
    if (k->shape()[1] != d_k) {
        throw std::runtime_error("Key dimension mismatch");
    }
    
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
    
    // Compute attention scores: Q @ K^T * scale
    auto scores = std::make_shared<Tensor>(
        std::initializer_list<size_t>{seq_len, seq_len}, 
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
    auto attn_weights = std::make_shared<Tensor>(
        std::initializer_list<size_t>{seq_len, seq_len}, 
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
    auto out = std::make_shared<Tensor>(
        std::initializer_list<size_t>{seq_len, d_v}, 
        q->requires_grad || k->requires_grad || v->requires_grad);
    out->children = {q, k, v};
    out->is_leaf = false;
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < d_v; ++j) {
            float sum = 0;
            #pragma omp simd reduction(+:sum)
            for (size_t s = 0; s < seq_len; ++s) {
                sum += attn_weights->data[i * seq_len + s] * v->data[s * d_v + j];
            }
            out->data[i * d_v + j] = sum;
        }
    }
    
    // Store intermediates for backward pass
    auto scores_ptr = scores;
    auto weights_ptr = attn_weights;
    
    // Backward pass
    if (out->requires_grad) {
        out->backward_fn = [q, k, v, out, scores_ptr, weights_ptr, seq_len, d_k, d_v, scale, causal_mask]() {
            // Gradient w.r.t V: weights^T @ out_grad
            if (v->requires_grad) {
                #pragma omp parallel for collapse(2)
                for (size_t i = 0; i < seq_len; ++i) {
                    for (size_t j = 0; j < d_v; ++j) {
                        float sum = 0;
                        for (size_t s = 0; s < seq_len; ++s) {
                            sum += weights_ptr->data[s * seq_len + i] * out->grad[s * d_v + j];
                        }
                        v->grad[i * d_v + j] += sum;
                    }
                }
            }
            
            // Gradient w.r.t attention weights
            std::vector<float> d_weights(seq_len * seq_len, 0);
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < seq_len; ++i) {
                for (size_t j = 0; j < seq_len; ++j) {
                    float sum = 0;
                    for (size_t d = 0; d < d_v; ++d) {
                        sum += out->grad[i * d_v + d] * v->data[j * d_v + d];
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

// Alias for backward compatibility
inline TensorPtr attention(TensorPtr q, TensorPtr k, TensorPtr v, 
                          bool causal_mask = false, float dropout_p = 0.0f) {
    return scaled_dot_product_attention(q, k, v, causal_mask, dropout_p);
}

} // namespace nn
} // namespace autograd