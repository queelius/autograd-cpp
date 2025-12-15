# autograd-cpp

A lightweight, high-performance C++ automatic differentiation library using computational graphs.

## Features

- **Computational Graph-based AD**: Forward and backward propagation through dynamic graphs
- **Jacobian & Hessian**: First and second-order derivative computations
- **Optimizers**: SGD with learning rate scheduling (linear, exponential, cosine, polynomial)
- **Header-mostly**: Minimal dependencies, easy integration
- **CMake Package**: FetchContent support for seamless integration

## Quick Start

### Using in Your Project (CMake FetchContent)

```cmake
include(FetchContent)

FetchContent_Declare(
    autograd_cpp
    GIT_REPOSITORY https://github.com/queelius/autograd-cpp.git
    GIT_TAG main
)

set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(autograd_cpp)

target_link_libraries(your_app PRIVATE autograd::autograd)
```

### Example Code

```cpp
#include <autograd/autograd.hpp>

using namespace autograd;

int main() {
    // Create computation graph
    auto x = constant(3.0);
    auto y = constant(4.0);
    auto z = mul(x, y);        // z = x * y
    auto result = add(z, constant(2.0));  // result = z + 2

    // Compute gradients
    result->backward();

    std::cout << "Result: " << result->data[0] << std::endl;  // 14
    std::cout << "dz/dx: " << x->grad[0] << std::endl;        // 4
    std::cout << "dz/dy: " << y->grad[0] << std::endl;        // 3

    return 0;
}
```

## Core Components

- **`tensor.hpp`**: Tensor class with gradient tracking
- **`ops.hpp`**: Operations (add, mul, exp, log, matmul, cholesky, solve_triangular, etc.)
- **`jacobian.hpp`**: Jacobian matrix computation
- **`hessian.hpp`**: Hessian matrix computation
- **`optim.hpp`**: SGD optimizer with learning rate schedules

### Broadcasting Support

Binary operations support scalar broadcasting - when one operand has size 1, it broadcasts to match the other:

```cpp
auto x = randn({100, 3});        // [100, 3] tensor
auto bias = from_vector({1.0f}, {1});  // scalar [1]
auto result = add(x, bias);      // bias broadcasts to [100, 3]
```

Operations with broadcasting: `add`, `sub`, `mul`, `div`, `minimum`, `maximum`

### Helper Functions

```cpp
auto x = randn({3, 4});
auto z = zeros_like(x);          // Same shape as x, filled with 0
auto o = ones_like(x);           // Same shape as x, filled with 1
auto xT = transpose(x);          // Transpose 2D tensor with gradient support
```

### Linear Algebra Operations (New!)

The library now includes advanced linear algebra operations with full gradient support:

- **`cholesky(A, lower)`**: Cholesky decomposition for positive-definite matrices
- **`solve_triangular(A, b, lower, transpose)`**: Solve triangular systems Ax = b

These enable efficient implementation of:
- Multivariate normal distributions
- Gaussian process regression
- Linear mixed models
- Kalman filtering

Example usage:
```cpp
// Solve Ax = b using Cholesky decomposition
auto L = cholesky(A, true);                    // A = LL^T
auto y = solve_triangular(L, b, true, false);  // Solve Ly = b
auto x = solve_triangular(L, y, true, true);   // Solve L^Tx = y
```

See `examples/mvn_example.cpp` for a complete multivariate normal distribution implementation.

## Building Examples

```bash
git clone https://github.com/queelius/autograd-cpp.git
cd autograd-cpp
mkdir build && cd build
cmake ..
make -j$(nproc)

# Run examples
./examples/simple_gradients
./examples/hessian_demo
```

## Requirements

- C++17 or later
- CMake 3.14+
- Optional: OpenMP for parallelization

## Use Cases

This library provides the **core automatic differentiation engine**. Build on top of it for:

- Neural networks and deep learning
- Statistical modeling and inference
- Physics simulations requiring gradients
- Optimization algorithms
- Scientific computing

## Design Philosophy

autograd-cpp is designed to be:

- **Minimal**: Core AD functionality only - build your domain-specific features on top
- **Efficient**: Optimized for performance with optional OpenMP parallelization
- **Flexible**: Dynamic computational graphs for arbitrary computations
- **Portable**: Standard C++17, works on any platform

## License

[Specify your license]

## Contributing

Contributions welcome! This is the core AD engine - domain-specific extensions (neural networks, statistical models, etc.) should be separate packages that depend on autograd-cpp.
