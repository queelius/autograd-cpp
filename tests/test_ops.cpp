#include "catch2.hpp"
#include "test_helpers.hpp"
#include "../include/autograd/tensor.hpp"
#include "../include/autograd/ops.hpp"
#include <cmath>

using namespace autograd;

// Helper function for numerical gradient checking
// Computes df/dx using finite differences
float numerical_gradient(std::function<TensorPtr(TensorPtr)> f, TensorPtr x, size_t idx, float eps = 1e-4f) {
    float original = x->data[idx];

    x->data[idx] = original + eps;
    auto y_plus = f(x);
    float f_plus = y_plus->item();

    x->data[idx] = original - eps;
    auto y_minus = f(x);
    float f_minus = y_minus->item();

    x->data[idx] = original;

    return (f_plus - f_minus) / (2 * eps);
}

TEST_CASE("Addition operation", "[ops]") {
    SECTION("Forward pass correctness") {
        std::vector<float> a_data = {1, 2, 3};
        std::vector<float> b_data = {4, 5, 6};

        auto a = from_vector(a_data, {3});
        auto b = from_vector(b_data, {3});

        auto c = add(a, b);

        REQUIRE(c->data[0] == 5);
        REQUIRE(c->data[1] == 7);
        REQUIRE(c->data[2] == 9);
    }

    SECTION("Backward pass gradient correctness") {
        auto a = from_vector({1.0f, 2.0f}, {2}, true);
        auto b = from_vector({3.0f, 4.0f}, {2}, true);

        auto c = add(a, b);
        auto out = sum(c);

        out->backward();

        // d(sum(a+b))/da = [1, 1]
        // d(sum(a+b))/db = [1, 1]
        REQUIRE(a->grad[0] == 1.0f);
        REQUIRE(a->grad[1] == 1.0f);
        REQUIRE(b->grad[0] == 1.0f);
        REQUIRE(b->grad[1] == 1.0f);
    }

    SECTION("Shape mismatch throws error") {
        auto a = zeros({2, 3});
        auto b = zeros({3, 2});

        REQUIRE_THROWS_AS(add(a, b), std::runtime_error);
    }
}

TEST_CASE("Subtraction operation", "[ops]") {
    SECTION("Forward pass correctness") {
        std::vector<float> a_data = {5, 7, 9};
        std::vector<float> b_data = {1, 2, 3};

        auto a = from_vector(a_data, {3});
        auto b = from_vector(b_data, {3});

        auto c = sub(a, b);

        REQUIRE(c->data[0] == 4);
        REQUIRE(c->data[1] == 5);
        REQUIRE(c->data[2] == 6);
    }

    SECTION("Backward pass gradient correctness") {
        auto a = from_vector({5.0f, 7.0f}, {2}, true);
        auto b = from_vector({1.0f, 2.0f}, {2}, true);

        auto c = sub(a, b);
        auto out = sum(c);

        out->backward();

        // d(sum(a-b))/da = [1, 1]
        // d(sum(a-b))/db = [-1, -1]
        REQUIRE(a->grad[0] == 1.0f);
        REQUIRE(a->grad[1] == 1.0f);
        REQUIRE(b->grad[0] == -1.0f);
        REQUIRE(b->grad[1] == -1.0f);
    }
}

TEST_CASE("Element-wise multiplication", "[ops]") {
    SECTION("Forward pass correctness") {
        std::vector<float> a_data = {2, 3, 4};
        std::vector<float> b_data = {5, 6, 7};

        auto a = from_vector(a_data, {3});
        auto b = from_vector(b_data, {3});

        auto c = mul(a, b);

        REQUIRE(c->data[0] == 10);
        REQUIRE(c->data[1] == 18);
        REQUIRE(c->data[2] == 28);
    }

    SECTION("Backward pass gradient correctness") {
        auto a = from_vector({2.0f, 3.0f}, {2}, true);
        auto b = from_vector({5.0f, 7.0f}, {2}, true);

        auto c = mul(a, b);
        auto out = sum(c);

        out->backward();

        // d(sum(a*b))/da = b
        // d(sum(a*b))/db = a
        REQUIRE(a->grad[0] == 5.0f);
        REQUIRE(a->grad[1] == 7.0f);
        REQUIRE(b->grad[0] == 2.0f);
        REQUIRE(b->grad[1] == 3.0f);
    }

    SECTION("Scalar multiplication forward") {
        auto a = from_vector({1.0f, 2.0f, 3.0f}, {3});
        auto c = mul(a, 2.0f);

        REQUIRE(c->data[0] == 2.0f);
        REQUIRE(c->data[1] == 4.0f);
        REQUIRE(c->data[2] == 6.0f);
    }

    SECTION("Scalar multiplication backward") {
        auto a = from_vector({1.0f, 2.0f}, {2}, true);
        auto c = mul(a, 3.0f);
        auto out = sum(c);

        out->backward();

        // d(sum(3*a))/da = [3, 3]
        REQUIRE(a->grad[0] == 3.0f);
        REQUIRE(a->grad[1] == 3.0f);
    }
}

TEST_CASE("Element-wise division", "[ops]") {
    SECTION("Forward pass correctness") {
        std::vector<float> a_data = {10, 20, 30};
        std::vector<float> b_data = {2, 4, 5};

        auto a = from_vector(a_data, {3});
        auto b = from_vector(b_data, {3});

        auto c = div(a, b);

        REQUIRE(c->data[0] == 5.0f);
        REQUIRE(c->data[1] == 5.0f);
        REQUIRE(c->data[2] == 6.0f);
    }

    SECTION("Backward pass gradient correctness") {
        auto a = from_vector({10.0f, 20.0f}, {2}, true);
        auto b = from_vector({2.0f, 4.0f}, {2}, true);

        auto c = div(a, b);
        auto out = sum(c);

        out->backward();

        // d(a/b)/da = 1/b
        REQUIRE(approx_equal(a->grad[0], 1.0f / 2.0f));
        REQUIRE(approx_equal(a->grad[1], 1.0f / 4.0f));

        // d(a/b)/db = -a/b^2
        REQUIRE(approx_equal(b->grad[0], -10.0f / 4.0f));
        REQUIRE(approx_equal(b->grad[1], -20.0f / 16.0f));
    }

    SECTION("Scalar division") {
        auto a = from_vector({10.0f, 20.0f}, {2});
        auto c = div(a, 2.0f);

        REQUIRE(c->data[0] == 5.0f);
        REQUIRE(c->data[1] == 10.0f);
    }
}

TEST_CASE("Matrix multiplication", "[ops]") {
    SECTION("Forward pass correctness") {
        // A = [[1, 2],      B = [[5, 6],
        //      [3, 4]]           [7, 8]]
        std::vector<float> a_data = {1, 2, 3, 4};
        std::vector<float> b_data = {5, 6, 7, 8};

        auto a = from_vector(a_data, {2, 2});
        auto b = from_vector(b_data, {2, 2});

        auto c = matmul(a, b);

        // C = [[1*5+2*7, 1*6+2*8],    = [[19, 22],
        //      [3*5+4*7, 3*6+4*8]]       [43, 50]]
        REQUIRE(c->data[0] == 19);
        REQUIRE(c->data[1] == 22);
        REQUIRE(c->data[2] == 43);
        REQUIRE(c->data[3] == 50);
    }

    SECTION("Shape compatibility check") {
        auto a = zeros({2, 3});
        auto b = zeros({4, 2});

        REQUIRE_THROWS_AS(matmul(a, b), std::runtime_error);
    }

    SECTION("Backward pass gradient correctness") {
        auto a = from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
        auto b = from_vector({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, true);

        auto c = matmul(a, b);
        auto out = sum(c);

        out->backward();

        // Gradient of a: should be b^T summed over output dimensions
        // Gradient of b: should be a^T summed over output dimensions
        REQUIRE(a->grad[0] > 0);  // Basic sanity check
        REQUIRE(b->grad[0] > 0);
    }
}

TEST_CASE("Exponential function", "[ops]") {
    SECTION("Forward pass correctness") {
        auto x = from_vector({0.0f, 1.0f, 2.0f}, {3});
        auto y = exp(x);

        REQUIRE(approx_equal(y->data[0], 1.0f));
        REQUIRE(approx_equal(y->data[1], std::exp(1.0f)));
        REQUIRE(approx_equal(y->data[2], std::exp(2.0f)));
    }

    SECTION("Backward pass gradient correctness") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto y = exp(x);
        auto out = sum(y);

        out->backward();

        // d(exp(x))/dx = exp(x)
        REQUIRE(approx_equal(x->grad[0], std::exp(1.0f)));
        REQUIRE(approx_equal(x->grad[1], std::exp(2.0f)));
    }

    SECTION("Numerical gradient check") {
        auto x = from_vector({0.5f}, {1}, true);

        auto f = [](TensorPtr x) { return exp(x); };
        float numerical_grad = numerical_gradient(f, x, 0);

        auto y = exp(x);
        y->backward();

        REQUIRE(approx_equal(x->grad[0], numerical_grad));
    }
}

TEST_CASE("Logarithm function", "[ops]") {
    SECTION("Forward pass correctness") {
        auto x = from_vector({1.0f, std::exp(1.0f), std::exp(2.0f)}, {3});
        auto y = log(x);

        REQUIRE(approx_equal(y->data[0], 0.0f));
        REQUIRE(approx_equal(y->data[1], 1.0f));
        REQUIRE(approx_equal(y->data[2], 2.0f));
    }

    SECTION("Backward pass gradient correctness") {
        auto x = from_vector({2.0f, 4.0f}, {2}, true);
        auto y = log(x);
        auto out = sum(y);

        out->backward();

        // d(log(x))/dx = 1/x
        REQUIRE(approx_equal(x->grad[0], 1.0f / 2.0f));
        REQUIRE(approx_equal(x->grad[1], 1.0f / 4.0f));
    }
}

TEST_CASE("ReLU activation", "[ops]") {
    SECTION("Forward pass correctness") {
        auto x = from_vector({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, {5});
        auto y = relu(x);

        REQUIRE(y->data[0] == 0.0f);
        REQUIRE(y->data[1] == 0.0f);
        REQUIRE(y->data[2] == 0.0f);
        REQUIRE(y->data[3] == 1.0f);
        REQUIRE(y->data[4] == 2.0f);
    }

    SECTION("Backward pass gradient correctness") {
        auto x = from_vector({-1.0f, 0.0f, 1.0f, 2.0f}, {4}, true);
        auto y = relu(x);
        auto out = sum(y);

        out->backward();

        // d(relu(x))/dx = 1 if x > 0, else 0
        REQUIRE(x->grad[0] == 0.0f);
        REQUIRE(x->grad[1] == 0.0f);
        REQUIRE(x->grad[2] == 1.0f);
        REQUIRE(x->grad[3] == 1.0f);
    }
}

TEST_CASE("Sigmoid activation", "[ops]") {
    SECTION("Forward pass correctness") {
        auto x = from_vector({0.0f}, {1});
        auto y = sigmoid(x);

        REQUIRE(approx_equal(y->data[0], 0.5f));
    }

    SECTION("Backward pass gradient correctness") {
        auto x = from_vector({0.0f}, {1}, true);
        auto y = sigmoid(x);
        y->backward();

        // d(sigmoid(x))/dx at x=0 = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        REQUIRE(approx_equal(x->grad[0], 0.25f));
    }

    SECTION("Numerical gradient check") {
        auto x = from_vector({1.5f}, {1}, true);

        auto f = [](TensorPtr x) { return sigmoid(x); };
        float numerical_grad = numerical_gradient(f, x, 0);

        auto y = sigmoid(x);
        y->backward();

        REQUIRE(approx_equal(x->grad[0], numerical_grad));
    }
}

TEST_CASE("Tanh activation", "[ops]") {
    SECTION("Forward pass correctness") {
        auto x = from_vector({0.0f}, {1});
        auto y = tanh(x);

        REQUIRE(approx_equal(y->data[0], 0.0f));
    }

    SECTION("Backward pass gradient correctness") {
        auto x = from_vector({0.0f}, {1}, true);
        auto y = tanh(x);
        y->backward();

        // d(tanh(x))/dx at x=0 = 1 - tanh(0)^2 = 1
        REQUIRE(approx_equal(x->grad[0], 1.0f));
    }

    SECTION("Numerical gradient check") {
        auto x = from_vector({0.5f}, {1}, true);

        auto f = [](TensorPtr x) { return tanh(x); };
        float numerical_grad = numerical_gradient(f, x, 0);

        auto y = tanh(x);
        y->backward();

        REQUIRE(approx_equal(x->grad[0], numerical_grad));
    }
}

TEST_CASE("GELU activation", "[ops]") {
    SECTION("Forward pass produces reasonable values") {
        auto x = from_vector({-2.0f, 0.0f, 2.0f}, {3});
        auto y = gelu(x);

        // GELU(0) ≈ 0
        REQUIRE(approx_equal(y->data[1], 0.0f, 0.1f));
        // GELU should be monotonic
        REQUIRE(y->data[0] < y->data[1]);
        REQUIRE(y->data[1] < y->data[2]);
    }

    SECTION("Numerical gradient check") {
        auto x = from_vector({0.5f}, {1}, true);

        auto f = [](TensorPtr x) { return gelu(x); };
        float numerical_grad = numerical_gradient(f, x, 0);

        auto y = gelu(x);
        y->backward();

        REQUIRE(approx_equal(x->grad[0], numerical_grad, 1e-2f));
    }
}

TEST_CASE("Sum reduction", "[ops]") {
    SECTION("Sum all elements") {
        auto x = from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4});
        auto s = sum(x);

        REQUIRE(s->size() == 1);
        REQUIRE(s->data[0] == 10.0f);
    }

    SECTION("Backward pass distributes gradient") {
        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);
        auto s = sum(x);

        s->backward();

        // Gradient should be 1 for all elements
        REQUIRE(x->grad[0] == 1.0f);
        REQUIRE(x->grad[1] == 1.0f);
        REQUIRE(x->grad[2] == 1.0f);
    }
}

TEST_CASE("Mean reduction", "[ops]") {
    SECTION("Mean all elements") {
        auto x = from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4});
        auto m = mean(x);

        REQUIRE(m->size() == 1);
        REQUIRE(approx_equal(m->data[0], 2.5f));
    }

    SECTION("Backward pass distributes scaled gradient") {
        auto x = from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, true);
        auto m = mean(x);

        m->backward();

        // Gradient should be 1/n for all elements
        float expected_grad = 1.0f / 4.0f;
        REQUIRE(approx_equal(x->grad[0], expected_grad));
        REQUIRE(approx_equal(x->grad[1], expected_grad));
        REQUIRE(approx_equal(x->grad[2], expected_grad));
        REQUIRE(approx_equal(x->grad[3], expected_grad));
    }
}

TEST_CASE("Complex gradient computation", "[ops][gradient]") {
    SECTION("Chain rule through multiple operations") {
        // f(x) = sum(sigmoid(x * 2))
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x2 = mul(x, 2.0f);
        auto sig = sigmoid(x2);
        auto result = sum(sig);

        result->backward();

        // Verify gradients are non-zero and reasonable
        REQUIRE(x->grad[0] != 0.0f);
        REQUIRE(x->grad[1] != 0.0f);
        REQUIRE(std::abs(x->grad[0]) < 1.0f);  // Sigmoid derivative is bounded
    }

    SECTION("Gradient through matrix operations") {
        // f(W, x) = sum(W @ x)
        auto W = from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
        auto x = from_vector({5.0f, 6.0f}, {2, 1}, true);

        auto y = matmul(W, x);
        auto result = sum(y);

        result->backward();

        // Both W and x should have non-zero gradients
        REQUIRE(W->grad[0] != 0.0f);
        REQUIRE(x->grad[0] != 0.0f);
    }
}
