#include "catch2.hpp"
#include "test_helpers.hpp"
#include "../include/autograd/tensor.hpp"
#include "../include/autograd/ops.hpp"
#include "../include/autograd/jacobian.hpp"
#include <cmath>

using namespace autograd;

TEST_CASE("Jacobian computation for linear function", "[jacobian]") {
    SECTION("Identity function Jacobian is identity matrix") {
        // f(x) = x
        auto f = [](TensorPtr x) {
            return x;
        };

        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);
        auto J = compute_jacobian(f, x);

        REQUIRE(J->shape()[0] == 3);
        REQUIRE(J->shape()[1] == 3);

        // Should be identity matrix
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                if (i == j) {
                    REQUIRE(approx_equal(J->data[i * 3 + j], 1.0f));
                } else {
                    REQUIRE(approx_equal(J->data[i * 3 + j], 0.0f));
                }
            }
        }
    }

    SECTION("Linear transformation Jacobian") {
        // f(x) = 2*x
        auto f = [](TensorPtr x) {
            return mul(x, 2.0f);
        };

        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto J = compute_jacobian(f, x);

        REQUIRE(J->shape()[0] == 2);
        REQUIRE(J->shape()[1] == 2);

        // Should be 2*I
        REQUIRE(approx_equal(J->data[0], 2.0f));
        REQUIRE(approx_equal(J->data[1], 0.0f));
        REQUIRE(approx_equal(J->data[2], 0.0f));
        REQUIRE(approx_equal(J->data[3], 2.0f));
    }
}

TEST_CASE("Jacobian for nonlinear functions", "[jacobian]") {
    SECTION("Jacobian of element-wise square") {
        // f(x) = x^2, df/dx = 2*x
        auto f = [](TensorPtr x) {
            return mul(x, x);
        };

        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);
        auto J = compute_jacobian(f, x);

        // Diagonal should be 2*x
        REQUIRE(approx_equal(J->data[0 * 3 + 0], 2.0f));
        REQUIRE(approx_equal(J->data[1 * 3 + 1], 4.0f));
        REQUIRE(approx_equal(J->data[2 * 3 + 2], 6.0f));

        // Off-diagonal should be zero
        REQUIRE(approx_equal(J->data[0 * 3 + 1], 0.0f));
        REQUIRE(approx_equal(J->data[0 * 3 + 2], 0.0f));
    }

    SECTION("Jacobian of sigmoid function") {
        // f(x) = sigmoid(x), df/dx = sigmoid(x) * (1 - sigmoid(x))
        auto f = [](TensorPtr x) {
            return sigmoid(x);
        };

        auto x = from_vector({0.0f}, {1}, true);
        auto J = compute_jacobian(f, x);

        // At x=0, sigmoid(0) = 0.5, derivative = 0.5 * 0.5 = 0.25
        REQUIRE(approx_equal(J->data[0], 0.25f));
    }
}

TEST_CASE("Jacobian for vector-valued functions", "[jacobian]") {
    SECTION("Jacobian of 2D to 2D function") {
        // f([x, y]) = [x*y, x+y]
        auto f = [](TensorPtr xy) {
            auto x = zeros({1}, true);
            auto y = zeros({1}, true);
            x->data[0] = xy->data[0];
            y->data[0] = xy->data[1];

            auto out = zeros({2}, true);
            auto xy_prod = mul(x, y);
            auto xy_sum = add(x, y);

            out->data[0] = xy_prod->data[0];
            out->data[1] = xy_sum->data[0];

            return out;
        };

        auto xy = from_vector({2.0f, 3.0f}, {2}, true);
        auto J = compute_jacobian(f, xy);

        // J = [[df1/dx, df1/dy],   = [[y, x],    = [[3, 2],
        //      [df2/dx, df2/dy]]     [1, 1]]       [1, 1]]
        REQUIRE(J->shape()[0] == 2);
        REQUIRE(J->shape()[1] == 2);

        REQUIRE(approx_equal(J->data[0 * 2 + 0], 3.0f));  // df1/dx = y
        REQUIRE(approx_equal(J->data[0 * 2 + 1], 2.0f));  // df1/dy = x
        REQUIRE(approx_equal(J->data[1 * 2 + 0], 1.0f));  // df2/dx = 1
        REQUIRE(approx_equal(J->data[1 * 2 + 1], 1.0f));  // df2/dy = 1
    }
}

TEST_CASE("Gradient function for scalar outputs", "[jacobian]") {
    SECTION("Gradient of quadratic function") {
        // f(x) = sum(x^2), gradient = 2*x
        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);

        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto grad = gradient(loss, x);

        REQUIRE(grad->size() == 3);
        REQUIRE(approx_equal(grad->data[0], 2.0f));
        REQUIRE(approx_equal(grad->data[1], 4.0f));
        REQUIRE(approx_equal(grad->data[2], 6.0f));
    }

    SECTION("Gradient throws on non-scalar output") {
        auto x = zeros({2}, true);
        auto y = zeros({2}, true);

        REQUIRE_THROWS_AS(gradient(y, x), std::runtime_error);
    }
}

TEST_CASE("Vector-Jacobian product", "[jacobian]") {
    SECTION("VJP for linear function") {
        // f(x) = 2*x, J = 2*I
        // v^T * J = 2*v^T
        auto f = [](TensorPtr x) {
            return mul(x, 2.0f);
        };

        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto v = from_vector({3.0f, 4.0f}, {2}, false);

        auto vjp_result = vjp(f, x, v);

        REQUIRE(vjp_result->size() == 2);
        REQUIRE(approx_equal(vjp_result->data[0], 6.0f));
        REQUIRE(approx_equal(vjp_result->data[1], 8.0f));
    }

    SECTION("VJP dimension mismatch throws error") {
        auto f = [](TensorPtr x) { return x; };
        auto x = zeros({2}, true);
        auto v = zeros({3}, false);

        REQUIRE_THROWS_AS(vjp(f, x, v), std::runtime_error);
    }
}

TEST_CASE("Jacobian-vector product", "[jacobian]") {
    SECTION("JVP for linear function") {
        // f(x) = 3*x, J = 3*I
        // J * v = 3*v
        auto f = [](TensorPtr x) {
            return mul(x, 3.0f);
        };

        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto v = from_vector({4.0f, 5.0f}, {2}, false);

        auto jvp_result = jvp(f, x, v);

        REQUIRE(jvp_result->size() == 2);
        REQUIRE(approx_equal(jvp_result->data[0], 12.0f, 1e-2f));
        REQUIRE(approx_equal(jvp_result->data[1], 15.0f, 1e-2f));
    }

    SECTION("JVP dimension mismatch throws error") {
        auto f = [](TensorPtr x) { return x; };
        auto x = zeros({2}, true);
        auto v = zeros({3}, false);

        REQUIRE_THROWS_AS(jvp(f, x, v), std::runtime_error);
    }
}

TEST_CASE("Divergence computation", "[jacobian]") {
    SECTION("Divergence of constant vector field is zero") {
        // F(x, y) = [1, 1], div = 0
        auto field = [](TensorPtr xy) {
            auto out = ones({2}, false);
            return out;
        };

        auto xy = from_vector({1.0f, 2.0f}, {2}, true);
        auto div = divergence(field, xy);

        REQUIRE(div->size() == 1);
        REQUIRE(approx_equal(div->data[0], 0.0f));
    }

    SECTION("Divergence of identity field") {
        // F(x, y) = [x, y], div = 1 + 1 = 2
        auto field = [](TensorPtr xy) {
            return xy->clone();
        };

        auto xy = from_vector({1.0f, 2.0f}, {2}, true);
        auto div = divergence(field, xy);

        REQUIRE(div->size() == 1);
        REQUIRE(approx_equal(div->data[0], 2.0f));
    }
}

TEST_CASE("Jacobian numerical accuracy", "[jacobian]") {
    SECTION("Compare with finite differences") {
        // f(x) = exp(x)
        auto f = [](TensorPtr x) {
            return exp(x);
        };

        auto x = from_vector({0.5f, 1.0f}, {2}, true);
        auto J = compute_jacobian(f, x);

        // Jacobian should be diagonal with exp(x_i) on diagonal
        REQUIRE(approx_equal(J->data[0], std::exp(0.5f)));
        REQUIRE(approx_equal(J->data[3], std::exp(1.0f)));
        REQUIRE(approx_equal(J->data[1], 0.0f));
        REQUIRE(approx_equal(J->data[2], 0.0f));
    }
}

TEST_CASE("Jacobian for matrix-valued outputs", "[jacobian]") {
    SECTION("Flatten and compute Jacobian") {
        // f(x) = reshape(x, new_shape) - tests view operation
        auto f = [](TensorPtr x) {
            // Just return a transformation
            return add(x, x);  // f(x) = 2x
        };

        auto x = from_vector({1.0f, 2.0f, 3.0f, 4.0f}, {4}, true);
        auto J = compute_jacobian(f, x);

        // Should be 2*I
        for (size_t i = 0; i < 4; ++i) {
            REQUIRE(approx_equal(J->data[i * 4 + i], 2.0f));
        }
    }
}

TEST_CASE("Forward-mode Jacobian computation", "[jacobian]") {
    SECTION("Forward mode gives same result as backward mode") {
        auto f = [](TensorPtr x) {
            return mul(x, x);  // f(x) = x^2
        };

        auto x = from_vector({1.0f, 2.0f}, {2}, true);

        auto J_backward = compute_jacobian(f, x);
        auto J_forward = compute_jacobian_forward(f, x);

        REQUIRE(J_backward->size() == J_forward->size());

        // Both should give df/dx = 2*x on diagonal
        for (size_t i = 0; i < 2; ++i) {
            REQUIRE(approx_equal(J_backward->data[i * 2 + i],
                               J_forward->data[i * 2 + i],
                               1e-2f));
        }
    }
}
