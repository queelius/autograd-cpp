#pragma once

#include "../../autograd/tensor.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <string>
#include <functional>

namespace autograd {
namespace nn {

// Forward declarations
class Module;
using ModulePtr = std::shared_ptr<Module>;

// Parameter class for managing trainable tensors
class Parameter {
private:
    TensorPtr tensor_;
    std::string name_;
    
public:
    Parameter() = default;
    
    explicit Parameter(TensorPtr tensor, const std::string& name = "") 
        : tensor_(tensor), name_(name) {
        if (tensor_) {
            tensor_->requires_grad = true;
        }
    }
    
    // Implicit conversion to TensorPtr for ease of use
    operator TensorPtr() const { return tensor_; }
    TensorPtr operator->() const { return tensor_; }
    
    TensorPtr data() const { return tensor_; }
    void set_data(TensorPtr tensor) { 
        tensor_ = tensor;
        if (tensor_) {
            tensor_->requires_grad = true;
        }
    }
    
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    
    bool is_valid() const { return tensor_ != nullptr; }
};

// Base class for all neural network modules
class Module {
protected:
    std::unordered_map<std::string, Parameter> parameters_;
    std::unordered_map<std::string, ModulePtr> submodules_;
    std::unordered_map<std::string, TensorPtr> buffers_;  // Non-trainable state
    bool training_ = true;
    std::string name_;
    
public:
    Module(const std::string& name = "Module") : name_(name) {}
    virtual ~Module() = default;
    
    // Core forward pass - must be implemented by derived classes
    virtual TensorPtr forward(TensorPtr input) = 0;
    
    // Convenience operator() that calls forward
    TensorPtr operator()(TensorPtr input) {
        return forward(input);
    }
    
    // Multiple input forward (for modules that need it)
    virtual std::vector<TensorPtr> forward(const std::vector<TensorPtr>& inputs) {
        if (inputs.size() != 1) {
            throw std::runtime_error(name_ + " expects single input");
        }
        return {forward(inputs[0])};
    }
    
    // Parameter management
    void register_parameter(const std::string& name, Parameter param) {
        if (param.is_valid()) {
            param.set_name(name);
            parameters_[name] = param;
        }
    }
    
    void register_buffer(const std::string& name, TensorPtr buffer) {
        if (buffer) {
            buffer->requires_grad = false;  // Buffers are not trainable
            buffers_[name] = buffer;
        }
    }
    
    void register_module(const std::string& name, ModulePtr module) {
        if (module) {
            submodules_[name] = module;
        }
    }
    
    // Get all parameters recursively
    std::vector<TensorPtr> parameters() const {
        std::vector<TensorPtr> params;
        
        // Add own parameters
        for (const auto& [name, param] : parameters_) {
            if (param.is_valid()) {
                params.push_back(param.data());
            }
        }
        
        // Add submodule parameters recursively
        for (const auto& [name, module] : submodules_) {
            auto sub_params = module->parameters();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        
        return params;
    }
    
    // Named parameters for debugging
    std::unordered_map<std::string, TensorPtr> named_parameters() const {
        std::unordered_map<std::string, TensorPtr> named_params;
        
        // Add own parameters
        for (const auto& [name, param] : parameters_) {
            if (param.is_valid()) {
                named_params[name_ + "." + name] = param.data();
            }
        }
        
        // Add submodule parameters
        for (const auto& [module_name, module] : submodules_) {
            auto sub_params = module->named_parameters();
            for (const auto& [param_name, param] : sub_params) {
                named_params[name_ + "." + module_name + "." + param_name] = param;
            }
        }
        
        return named_params;
    }
    
    // Training mode management
    void train(bool mode = true) {
        training_ = mode;
        for (auto& [name, module] : submodules_) {
            module->train(mode);
        }
    }
    
    void eval() {
        train(false);
    }
    
    bool is_training() const {
        return training_;
    }
    
    // Zero gradients for all parameters
    void zero_grad() {
        for (auto& param : parameters()) {
            param->zero_grad();
        }
    }
    
    // Apply a function to all parameters
    void apply(std::function<void(TensorPtr)> fn) {
        for (auto& param : parameters()) {
            fn(param);
        }
    }
    
    // Module information
    const std::string& name() const { return name_; }
    void set_name(const std::string& n) { name_ = n; }
    
    // Count parameters
    size_t num_parameters() const {
        size_t count = 0;
        for (const auto& param : parameters()) {
            count += param->size();
        }
        return count;
    }
    
    // String representation
    virtual std::string to_string() const {
        return name_ + "()";
    }
};

// Sequential container for chaining modules
class Sequential : public Module {
private:
    std::vector<ModulePtr> modules_;
    
public:
    Sequential(const std::string& name = "Sequential") : Module(name) {}
    
    Sequential(std::initializer_list<ModulePtr> modules, 
               const std::string& name = "Sequential") 
        : Module(name), modules_(modules) {
        for (size_t i = 0; i < modules_.size(); ++i) {
            register_module(std::to_string(i), modules_[i]);
        }
    }
    
    void add(ModulePtr module) {
        modules_.push_back(module);
        register_module(std::to_string(modules_.size() - 1), module);
    }
    
    TensorPtr forward(TensorPtr input) override {
        for (auto& module : modules_) {
            input = module->forward(input);
        }
        return input;
    }
    
    size_t size() const { return modules_.size(); }
    
    ModulePtr operator[](size_t idx) {
        if (idx >= modules_.size()) {
            throw std::out_of_range("Sequential index out of range");
        }
        return modules_[idx];
    }
    
    std::string to_string() const override {
        std::string str = name_ + "(\n";
        for (size_t i = 0; i < modules_.size(); ++i) {
            str += "  (" + std::to_string(i) + "): " + modules_[i]->to_string() + "\n";
        }
        str += ")";
        return str;
    }
};

} // namespace nn
} // namespace autograd