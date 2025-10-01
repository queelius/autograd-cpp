#include "catch2.hpp"
#include "test_helpers.hpp"
#include "../include/autograd/tensor.hpp"
#include "../include/autograd/ops.hpp"
#include "../include/autograd/hessian.hpp"
#include <cmath>

using namespace autograd;

TEST_CASE("Hessian diagonal computation", "[hessian]") {
    SECTION("Diagonal Hessian for quadratic function") {
        // f(x) = sum(x^2)
        // First derivative: 2*x
        // Second derivative: 2
        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);
        auto diag = hessian->diagonal();

        REQUIRE(diag->size() == 3);
        // All diagonal elements should be 2
        REQUIRE(approx_equal(diag->data[0], 2.0f));
        REQUIRE(approx_equal(diag->data[1], 2.0f));
        REQUIRE(approx_equal(diag->data[2], 2.0f));
    }

    SECTION("Diagonal Hessian for cubic function") {
        // f(x) = sum(x^3)
        // First derivative: 3*x^2
        // Second derivative: 6*x
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto x_cube = mul(x_sq, x);
        auto loss = sum(x_cube);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);
        auto diag = hessian->diagonal();

        REQUIRE(diag->size() == 2);
        REQUIRE(approx_equal(diag->data[0], 6.0f));
        REQUIRE(approx_equal(diag->data[1], 12.0f));
    }
}

TEST_CASE("Hessian matrix-vector product", "[hessian]") {
    SECTION("Hessian-vector product for quadratic") {
        // f(x) = x^T x, H = 2*I
        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);

        auto v = from_vector({1.0f, 0.0f, 1.0f}, {3}, false);
        auto Hv = hessian->matvec(v);

        REQUIRE(Hv->size() == 3);
        // H*v = 2*I*v = 2*v
        REQUIRE(approx_equal(Hv->data[0], 2.0f));
        REQUIRE(approx_equal(Hv->data[1], 0.0f));
        REQUIRE(approx_equal(Hv->data[2], 2.0f));
    }
}

TEST_CASE("Hessian solve operation", "[hessian]") {
    SECTION("Solve H*x = b for diagonal Hessian") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);

        auto b = from_vector({4.0f, 8.0f}, {2}, false);
        auto solution = hessian->solve(b);

        // H = 2*I, so H^(-1) = 0.5*I
        // Solution should be 0.5*b
        REQUIRE(approx_equal(solution->data[0], 2.0f));
        REQUIRE(approx_equal(solution->data[1], 4.0f));
    }
}

TEST_CASE("Hessian trace", "[hessian]") {
    SECTION("Trace of diagonal Hessian") {
        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);

        float trace = hessian->trace();

        // Trace of 2*I for 3D is 6
        REQUIRE(approx_equal(trace, 6.0f));
    }
}

TEST_CASE("Hessian add diagonal regularization", "[hessian]") {
    SECTION("Add lambda*I to Hessian") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);

        hessian->addDiagonal(1.0f);

        auto diag = hessian->diagonal();

        // Original diagonal was 2, after adding 1 should be 3
        REQUIRE(approx_equal(diag->data[0], 3.0f));
        REQUIRE(approx_equal(diag->data[1], 3.0f));
    }
}

TEST_CASE("Hessian positive definiteness check", "[hessian]") {
    SECTION("Positive definite Hessian") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);

        REQUIRE(hessian->isPositiveDefinite() == true);
    }
}

TEST_CASE("Stochastic diagonal Hessian", "[hessian]") {
    SECTION("Stochastic estimator approximates diagonal") {
        // For a simple quadratic, stochastic estimator should work well
        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::STOCHASTIC_DIAGONAL);

        auto diag = hessian->diagonal();

        // Should approximate 2 (with some variance due to stochastic nature)
        REQUIRE(diag->size() == 3);
        // Allow more tolerance for stochastic method
        REQUIRE(approx_equal(diag->data[0], 2.0f, 0.5f));
        REQUIRE(approx_equal(diag->data[1], 2.0f, 0.5f));
        REQUIRE(approx_equal(diag->data[2], 2.0f, 0.5f));
    }
}

TEST_CASE("L-BFGS Hessian approximation", "[hessian]") {
    SECTION("L-BFGS update and solve") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto loss = sum(mul(x, x));

        auto hessian = Hessian::compute(loss, x, HessianMethod::LBFGS);

        // Create mock gradient update
        auto x_old = from_vector({1.0f, 2.0f}, {2}, false);
        auto g_old = from_vector({2.0f, 4.0f}, {2}, false);  // grad of x^2
        auto x_new = from_vector({0.9f, 1.9f}, {2}, false);
        auto g_new = from_vector({1.8f, 3.8f}, {2}, false);

        // This would normally be done during optimization
        // Just test that the interface works
        REQUIRE(hessian->dimension() == 2);
        REQUIRE(hessian->storage() == HessianStorage::IMPLICIT);
    }
}

TEST_CASE("Hessian utility functions", "[hessian]") {
    SECTION("compute_hessian convenience function") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto H = compute_hessian(loss, x);

        REQUIRE(H->size() == 4);  // 2x2 matrix
    }

    SECTION("hessian_diagonal convenience function") {
        auto x = from_vector({1.0f, 2.0f, 3.0f}, {3}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto diag = hessian_diagonal(loss, x);

        REQUIRE(diag->size() == 3);
        REQUIRE(approx_equal(diag->data[0], 2.0f));
    }

    SECTION("laplacian function") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto lap = laplacian(loss, x);

        REQUIRE(lap->size() == 1);
        // Laplacian = trace of Hessian = 2 + 2 = 4
        REQUIRE(approx_equal(lap->data[0], 4.0f));
    }
}

TEST_CASE("Gauss-Newton Hessian approximation", "[hessian]") {
    SECTION("Gauss-Newton diagonal for linear residuals") {
        // residual(x) = x (identity), Jacobian = I
        // Gauss-Newton: H ≈ 2*J^T*J = 2*I
        auto residual_fn = [](TensorPtr x) {
            return x->clone();
        };

        auto x = from_vector({1.0f, 2.0f}, {2}, true);

        auto diag = gauss_newton_hessian_diag(residual_fn, x);

        REQUIRE(diag->size() == 2);
        // J^T*J diagonal for identity J is 1, times 2 is 2
        REQUIRE(approx_equal(diag->data[0], 2.0f));
        REQUIRE(approx_equal(diag->data[1], 2.0f));
    }
}

TEST_CASE("Hessian dimension validation", "[hessian]") {
    SECTION("Hessian requires scalar loss") {
        auto x = zeros({2}, true);
        auto y = zeros({2}, true);  // Non-scalar output

        REQUIRE_THROWS_AS(Hessian::compute(y, x), std::runtime_error);
    }
}

TEST_CASE("Hessian for different function types", "[hessian]") {
    SECTION("Hessian of exponential function") {
        // f(x) = sum(exp(x))
        // First derivative: exp(x)
        // Second derivative: exp(x)
        auto x = from_vector({0.0f, 1.0f}, {2}, true);
        auto exp_x = exp(x);
        auto loss = sum(exp_x);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);
        auto diag = hessian->diagonal();

        REQUIRE(diag->size() == 2);
        REQUIRE(approx_equal(diag->data[0], std::exp(0.0f)));
        REQUIRE(approx_equal(diag->data[1], std::exp(1.0f)));
    }

    SECTION("Hessian of sigmoid function") {
        // f(x) = sum(sigmoid(x))
        // Second derivative is more complex
        auto x = from_vector({0.0f}, {1}, true);
        auto sig_x = sigmoid(x);
        auto loss = sum(sig_x);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);
        auto diag = hessian->diagonal();

        REQUIRE(diag->size() == 1);
        // At x=0, second derivative of sigmoid is 0
        REQUIRE(approx_equal(diag->data[0], 0.0f, 0.1f));
    }
}

TEST_CASE("Hessian access methods", "[hessian]") {
    SECTION("at() method for dense Hessian") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::EXACT);

        // Diagonal elements should be 2
        float h00 = hessian->at(0, 0);
        float h11 = hessian->at(1, 1);

        REQUIRE(approx_equal(h00, 2.0f));
        REQUIRE(approx_equal(h11, 2.0f));
    }

    SECTION("at() method for diagonal Hessian") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::DIAGONAL_ONLY);

        // Diagonal access
        float h00 = hessian->at(0, 0);
        float h11 = hessian->at(1, 1);

        REQUIRE(approx_equal(h00, 2.0f));
        REQUIRE(approx_equal(h11, 2.0f));

        // Off-diagonal should be zero
        REQUIRE(hessian->at(0, 1) == 0.0f);
    }
}

TEST_CASE("QuasiNewtonApprox backward compatibility", "[hessian]") {
    SECTION("QuasiNewtonApprox interface works") {
        QuasiNewtonApprox approx(3, 5);

        // Test basic interface
        auto diag = approx.get_diagonal();
        REQUIRE(diag->size() == 3);

        auto grad = ones({3}, false);
        auto direction = approx.apply_inverse(grad);
        REQUIRE(direction->size() == 3);
    }
}

TEST_CASE("Hessian symmetry", "[hessian]") {
    SECTION("Exact Hessian should be symmetric") {
        auto x = from_vector({1.0f, 2.0f}, {2}, true);
        auto x_sq = mul(x, x);
        auto loss = sum(x_sq);

        auto hessian = Hessian::compute(loss, x, HessianMethod::EXACT);

        // Check symmetry: H[i,j] == H[j,i]
        float h01 = hessian->at(0, 1);
        float h10 = hessian->at(1, 0);

        REQUIRE(approx_equal(h01, h10));
    }
}
