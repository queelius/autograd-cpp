#include "../include/autograd/tensor.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

namespace autograd {

// Backward pass implementation
void Tensor::backward() {
    if (_size != 1) {
        throw std::runtime_error("backward() can only be called on scalar tensors");
    }
    
    // Initialize gradient
    grad[0] = 1.0f;
    
    // Build topological order
    std::vector<TensorPtr> topo;
    std::unordered_set<Tensor*> visited;
    
    std::function<void(TensorPtr)> build_topo = [&](TensorPtr node) {
        if (visited.find(node.get()) == visited.end()) {
            visited.insert(node.get());
            for (auto& child : node->children) {
                if (child->requires_grad) {
                    build_topo(child);
                }
            }
            topo.push_back(node);
        }
    };
    
    build_topo(shared_from_this());
    
    // Execute backward functions in reverse topological order
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) {
            (*it)->backward_fn();
        }
    }
}

// View operation
TensorPtr Tensor::view(std::initializer_list<size_t> new_shape) {
    size_t new_size = 1;
    int infer_dim = -1;
    size_t known_size = 1;
    
    int i = 0;
    for (auto s : new_shape) {
        if (s == -1) {
            if (infer_dim != -1) {
                throw std::runtime_error("Only one dimension can be inferred");
            }
            infer_dim = i;
        } else {
            known_size *= s;
        }
        i++;
    }
    
    if (infer_dim != -1) {
        if (_size % known_size != 0) {
            throw std::runtime_error("Cannot infer dimension - size mismatch");
        }
        // Modify new_shape to include inferred dimension
        std::vector<size_t> final_shape;
        i = 0;
        for (auto s : new_shape) {
            if (i == infer_dim) {
                final_shape.push_back(_size / known_size);
            } else {
                final_shape.push_back(s);
            }
            i++;
        }
        
        auto result = std::make_shared<Tensor>(final_shape, requires_grad);
        std::memcpy(result->data, data, _size * sizeof(float));
        
        if (requires_grad) {
            result->children = {shared_from_this()};
            result->backward_fn = [this, result]() {
                #pragma omp simd
                for (size_t i = 0; i < _size; ++i) {
                    this->grad[i] += result->grad[i];
                }
            };
        }
        
        return result;
    } else {
        for (auto s : new_shape) {
            new_size *= s;
        }
        
        if (new_size != _size) {
            throw std::runtime_error("View size mismatch");
        }
        
        auto result = std::make_shared<Tensor>(new_shape, requires_grad);
        std::memcpy(result->data, data, _size * sizeof(float));
        
        if (requires_grad) {
            result->children = {shared_from_this()};
            result->backward_fn = [this, result]() {
                #pragma omp simd
                for (size_t i = 0; i < _size; ++i) {
                    this->grad[i] += result->grad[i];
                }
            };
        }
        
        return result;
    }
}

// Transpose operation (2D only)
TensorPtr Tensor::transpose(int dim1, int dim2) {
    if (_shape.size() != 2) {
        throw std::runtime_error("Transpose currently only supports 2D tensors");
    }
    
    size_t rows = _shape[0];
    size_t cols = _shape[1];
    
    auto result = std::make_shared<Tensor>(std::initializer_list<size_t>{cols, rows}, requires_grad);
    
    // Transpose data
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result->data[j * rows + i] = data[i * cols + j];
        }
    }
    
    if (requires_grad) {
        result->children = {shared_from_this()};
        result->backward_fn = [this, result, rows, cols]() {
            #pragma omp parallel for collapse(2)
            for (size_t i = 0; i < cols; ++i) {
                for (size_t j = 0; j < rows; ++j) {
                    this->grad[j * cols + i] += result->grad[i * rows + j];
                }
            }
        };
    }
    
    return result;
}

// Random initialization functions
void Tensor::randn(float mean, float std) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    for (size_t i = 0; i < _size; ++i) {
        data[i] = dist(gen);
    }
}

void Tensor::uniform(float low, float high) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);
    
    for (size_t i = 0; i < _size; ++i) {
        data[i] = dist(gen);
    }
}

void Tensor::xavier_uniform(size_t fan_in, size_t fan_out) {
    float limit = std::sqrt(6.0f / (fan_in + fan_out));
    uniform(-limit, limit);
}

void Tensor::kaiming_uniform(size_t fan_in) {
    float limit = std::sqrt(3.0f / fan_in);
    uniform(-limit, limit);
}

} // namespace autograd