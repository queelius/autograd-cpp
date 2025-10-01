# FluxCore: The Mathematics of Automatic Differentiation

## Why Computational Graphs?

In scientific computing, machine learning, and optimization, we constantly need derivatives. Whether training neural networks, solving differential equations, or optimizing engineering designs, the ability to compute gradients efficiently and accurately is fundamental. FluxCore provides a high-performance C++ library for automatic differentiation through computational graphs.

## The Computational Graph

A computational graph represents a function `f` as a directed acyclic graph where:
- **Nodes** represent operations (add, multiply, exp, log, etc.)
- **Edges** represent data flow between operations
- **Leaves** are input variables
- **Root** is the output

Consider a simple function: `f(x, y) = (x * y) + exp(x)`

```
     x ----+---> [*] ---> [+] ---> f
           |       ^        ^
           |       |        |
           +-----> y      [exp]
           |                ^
           +----------------+
```

Each node stores:
1. Its computed value (forward pass)
2. Its gradient with respect to the output (backward pass)
3. The operation's derivative rule

This structure enables us to:
- Compute `f(x)` by traversing forward
- Compute `∇f` by traversing backward
- Track exact derivatives through complex compositions

## The Loss Function and Backpropagation

In machine learning, we typically minimize a loss function `L(θ)` where `θ` are model parameters. The computational graph lets us compute `∂L/∂θᵢ` for all parameters efficiently through **backpropagation**:

1. **Forward Pass**: Compute all node values from inputs to loss
2. **Backward Pass**: Compute gradients from loss back to parameters

The key insight: the chain rule naturally follows the graph structure:
```
∂L/∂θ = ∂L/∂z · ∂z/∂y · ∂y/∂θ
```

Each node only needs to know its local gradient - the graph handles the global composition.

## Beyond Gradients: Jacobians and Hessians

While gradients (first derivatives) enable optimization, higher-order derivatives unlock more sophisticated algorithms:

### Jacobians: First-Order for Vector Functions

For `f: ℝⁿ → ℝᵐ`, the Jacobian matrix `J` has entries:
```
J[i,j] = ∂fᵢ/∂xⱼ
```

**Why Jacobians matter:**
- **Sensitivity analysis**: How do outputs respond to input perturbations?
- **Newton's method**: For solving systems of equations
- **Implicit differentiation**: For constrained optimization
- **Change of variables**: In probability and physics

**Computational approach:**
- Forward-mode AD: Efficient when n << m (few inputs, many outputs)
- Reverse-mode AD: Efficient when n >> m (many inputs, few outputs)
- For full Jacobian: Need min(n, m) passes through the graph

### Hessians: Second-Order Information

The Hessian matrix `H` contains second derivatives:
```
H[i,j] = ∂²f/∂xᵢ∂xⱼ
```

**Why Hessians matter:**
- **Newton's method**: Quadratic convergence vs linear for gradient descent
- **Trust region methods**: Better step size selection
- **Uncertainty quantification**: Hessian inverse approximates parameter covariance
- **Critical point analysis**: Determine if minima, maxima, or saddle points
- **Preconditioning**: Improve optimization convergence

**Computational approach:**
While technically the Jacobian of the gradient, computing the full Hessian is expensive (O(n²)). FluxCore provides:
- **Exact Hessian**: When accuracy is critical
- **Diagonal approximation**: O(n) cost, often sufficient
- **Hessian-vector products**: For iterative methods
- **Quasi-Newton updates**: L-BFGS style approximations

## Efficient Approximations

Real-world problems often have thousands or millions of parameters. FluxCore implements several approximation strategies:

### Gradient Approximations
- **Stochastic gradients**: Subsample data for efficiency
- **Gradient checkpointing**: Trade computation for memory

### Jacobian Approximations
- **Matrix-free methods**: Compute J·v without forming J
- **Randomized SVD**: Low-rank approximations
- **Sparse Jacobians**: Exploit structure

### Hessian Approximations
- **Diagonal-only**: H_diag via Hutchinson's estimator
- **Block-diagonal**: For neural networks, per-layer blocks
- **Low-rank updates**: BFGS/L-BFGS maintain H⁻¹ approximation
- **Gauss-Newton**: For least-squares, H ≈ J^T·J
- **Fisher Information**: For probabilistic models

## The General Theme: Differential Operators

The computational graph is fundamentally a **differential operator engine**. It can compute:

1. **Directional derivatives**: ∇f·v (gradient dot product)
2. **Divergence**: ∇·F (trace of Jacobian)
3. **Curl**: ∇×F (in 3D, antisymmetric part of Jacobian)
4. **Laplacian**: ∇²f (trace of Hessian)
5. **Vector-Jacobian products**: v^T·J (reverse-mode)
6. **Jacobian-vector products**: J·v (forward-mode)
7. **Hessian-vector products**: H·v (for iterative solvers)

## Taylor Series and Local Approximations

While we *could* use the computational graph to build Taylor approximations:
```
f(x) ≈ f(c) + ∇f(c)·(x-c) + ½(x-c)^T·H(c)·(x-c) + ...
```

This is typically less useful than direct evaluation because:
- Convergence radius may be small
- Higher-order terms are expensive
- Direct evaluation through the graph is often faster

However, Taylor approximations are valuable for:
- **Trust region methods**: Local quadratic models
- **Uncertainty propagation**: Linear/quadratic error analysis
- **Sensitivity analysis**: Understanding local behavior

## Applications Beyond Machine Learning

The computational graph framework extends far beyond neural networks:

### Scientific Computing
- **Automatic differentiation for PDEs**: Compute sensitivities in simulations
- **Optimization-based design**: Gradients for engineering parameters
- **Inverse problems**: Match models to observations

### Statistics
- **Maximum likelihood**: Gradients and Hessians for parameter estimation
- **Bayesian inference**: Laplace approximation uses Hessian
- **Generalized linear models**: Newton-Raphson iterations

### Physics
- **Hamiltonian dynamics**: Preserve symplectic structure
- **Variational methods**: Minimize energy functionals
- **Quantum mechanics**: Expectation value gradients

### Finance
- **Greeks**: Sensitivities of option prices
- **Risk management**: Portfolio optimization
- **Calibration**: Fit models to market data

## Design Philosophy

FluxCore prioritizes:
1. **Accuracy**: Exact derivatives, not finite differences
2. **Efficiency**: Optimal complexity algorithms, SIMD vectorization
3. **Flexibility**: From single gradients to full Hessians
4. **Practicality**: Approximations when exact computation is infeasible

The computational graph is not just a tool for backpropagation - it's a general framework for differential computation. By providing efficient access to gradients, Jacobians, Hessians, and their approximations, FluxCore enables sophisticated numerical algorithms across diverse domains.

## Getting Started

```cpp
#include <fluxcore/fluxcore.hpp>

// Define your function through operations
auto f = [](auto x) { return sum(exp(x) + x * x); };

// Get gradient
auto grad = gradient(f(x), x);

// Get Jacobian
auto J = compute_jacobian(f, x);

// Get Hessian (exact or approximate)
auto H = compute_hessian(f(x), x);
auto H_diag = hessian_diagonal_stochastic(f(x), x);

// Use for optimization
QuasiNewtonOptimizer opt(x);
opt.step(f, x);  // Uses L-BFGS approximation internally
```

The power of automatic differentiation is that you write your function naturally, and FluxCore handles all the derivatives - exactly, efficiently, and automatically.