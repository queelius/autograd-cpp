#include <autograd/autograd.hpp>
#include <iostream>

using namespace autograd;

int main() {
    std::cout << "=== Simple Gradient Computation ===" << std::endl;
    
    // Create tensors with gradients enabled
    auto x = from_vector({2.0f, 3.0f}, {2}, true);
    auto y = from_vector({1.0f, 4.0f}, {2}, true);
    
    // Compute z = x * y + x^2
    auto z = add(mul(x, y), mul(x, x));
    
    // Sum to get scalar
    auto loss = sum(z);
    
    std::cout << "x = [2, 3]" << std::endl;
    std::cout << "y = [1, 4]" << std::endl;
    std::cout << "z = x * y + x^2 = [" << z->data[0] << ", " << z->data[1] << "]" << std::endl;
    std::cout << "loss = sum(z) = " << loss->data[0] << std::endl;
    
    // Compute gradients
    loss->backward();
    
    std::cout << "\nGradients:" << std::endl;
    std::cout << "∂loss/∂x = [" << x->grad[0] << ", " << x->grad[1] << "]" << std::endl;
    std::cout << "∂loss/∂y = [" << y->grad[0] << ", " << y->grad[1] << "]" << std::endl;
    
    // Analytical: ∂loss/∂x = y + 2x = [1+4, 4+6] = [5, 10]
    //            ∂loss/∂y = x = [2, 3]
    
    return 0;
}