#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <algorithm>
#include <random>

namespace autograd {

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

// Core tensor class with automatic differentiation
class Tensor : public std::enable_shared_from_this<Tensor> {
private:
    size_t _size;
    std::vector<size_t> _shape;
    std::vector<size_t> _strides;
    
    void compute_strides() {
        _strides.resize(_shape.size());
        size_t stride = 1;
        for (int i = _shape.size() - 1; i >= 0; --i) {
            _strides[i] = stride;
            stride *= _shape[i];
        }
    }

public:
    // Data and gradients - aligned for SIMD
    float* data;
    float* grad;
    
    // Computation graph
    std::vector<TensorPtr> children;
    std::function<void()> backward_fn;
    bool requires_grad;
    bool is_leaf;
    
    // Shape and size
    size_t size() const { return _size; }
    const std::vector<size_t>& shape() const { return _shape; }
    const std::vector<size_t>& strides() const { return _strides; }
    size_t ndim() const { return _shape.size(); }
    
    // Constructors
    Tensor(std::initializer_list<size_t> shape_init, bool requires_grad = true)
        : _shape(shape_init), requires_grad(requires_grad), is_leaf(true) {
        
        _size = 1;
        for (auto s : _shape) _size *= s;
        
        compute_strides();
        
        // Aligned allocation for SIMD
        data = static_cast<float*>(std::aligned_alloc(32, _size * sizeof(float)));
        grad = static_cast<float*>(std::aligned_alloc(32, _size * sizeof(float)));
        
        std::memset(data, 0, _size * sizeof(float));
        std::memset(grad, 0, _size * sizeof(float));
    }
    
    Tensor(std::vector<size_t> shape_vec, bool requires_grad = true)
        : _shape(shape_vec), requires_grad(requires_grad), is_leaf(true) {
        
        _size = 1;
        for (auto s : _shape) _size *= s;
        
        compute_strides();
        
        // Aligned allocation for SIMD
        data = static_cast<float*>(std::aligned_alloc(32, _size * sizeof(float)));
        grad = static_cast<float*>(std::aligned_alloc(32, _size * sizeof(float)));
        
        std::memset(data, 0, _size * sizeof(float));
        std::memset(grad, 0, _size * sizeof(float));
    }
    
    // Constructor from data
    Tensor(const float* data_ptr, std::initializer_list<size_t> shape_init, bool requires_grad = true)
        : _shape(shape_init), requires_grad(requires_grad), is_leaf(true) {
        
        _size = 1;
        for (auto s : _shape) _size *= s;
        
        compute_strides();
        
        data = static_cast<float*>(std::aligned_alloc(32, _size * sizeof(float)));
        grad = static_cast<float*>(std::aligned_alloc(32, _size * sizeof(float)));
        
        std::memcpy(data, data_ptr, _size * sizeof(float));
        std::memset(grad, 0, _size * sizeof(float));
    }
    
    ~Tensor() {
        std::free(data);
        std::free(grad);
    }
    
    // Disable copy, enable move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;
    
    // Zero gradients
    void zero_grad() {
        std::memset(grad, 0, _size * sizeof(float));
    }
    
    // Backward pass
    void backward();
    
    // Utility functions
    void fill(float value) {
        std::fill(data, data + _size, value);
    }
    
    float item() const {
        if (_size != 1) {
            throw std::runtime_error("item() only works for scalar tensors");
        }
        return data[0];
    }
    
    // Indexing
    float& operator[](size_t idx) { return data[idx]; }
    const float& operator[](size_t idx) const { return data[idx]; }
    
    float& at(std::initializer_list<size_t> indices) {
        size_t idx = 0;
        size_t i = 0;
        for (auto index : indices) {
            idx += index * _strides[i++];
        }
        return data[idx];
    }
    
    // Shape manipulation
    TensorPtr view(std::initializer_list<size_t> new_shape);
    TensorPtr reshape(std::initializer_list<size_t> new_shape) { return view(new_shape); }
    TensorPtr transpose(int dim1 = 0, int dim2 = 1);
    
    // Initialization
    void randn(float mean = 0, float std = 1);
    void uniform(float low = 0, float high = 1);
    void xavier_uniform(size_t fan_in, size_t fan_out);
    void kaiming_uniform(size_t fan_in);
    
    // Detach from graph
    TensorPtr detach() {
        auto result = std::make_shared<Tensor>(std::vector<size_t>(_shape), false);
        std::memcpy(result->data, data, _size * sizeof(float));
        return result;
    }
    
    // Clone with gradients
    TensorPtr clone() {
        auto result = std::make_shared<Tensor>(std::vector<size_t>(_shape), requires_grad);
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
};

// Factory functions
inline TensorPtr zeros(std::initializer_list<size_t> shape, bool requires_grad = true) {
    return std::make_shared<Tensor>(shape, requires_grad);
}

inline TensorPtr ones(std::initializer_list<size_t> shape, bool requires_grad = true) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    t->fill(1.0f);
    return t;
}

inline TensorPtr full(std::initializer_list<size_t> shape, float value, bool requires_grad = true) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    t->fill(value);
    return t;
}

inline TensorPtr randn(std::initializer_list<size_t> shape, float mean = 0, float std = 1, bool requires_grad = true) {
    auto t = std::make_shared<Tensor>(shape, requires_grad);
    t->randn(mean, std);
    return t;
}

inline TensorPtr from_vector(const std::vector<float>& vec, std::initializer_list<size_t> shape, bool requires_grad = true) {
    return std::make_shared<Tensor>(vec.data(), shape, requires_grad);
}

} // namespace autograd