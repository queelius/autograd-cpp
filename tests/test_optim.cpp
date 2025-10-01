#include "catch2.hpp"
#include "test_helpers.hpp"
#include "../include/autograd/tensor.hpp"
#include "../include/autograd/ops.hpp"
#include "../include/autograd/optim.hpp"
#include <cmath>

using namespace autograd;
using namespace autograd::optim;

TEST_CASE("SGD optimizer basic functionality", "[optim]") {
    SECTION("SGD takes step in gradient direction") {
        // Minimize f(x) = x^2
        // Gradient = 2*x
        auto x = from_vector({10.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        SGD optimizer(params, 0.1f, 0.0f);  // lr=0.1, no momentum

        // Compute gradient
        auto loss = mul(x, x);
        auto total_loss = sum(loss);
        total_loss->backward();

        float initial_x = x->data[0];
        float initial_grad = x->grad[0];

        optimizer.step();

        // x should decrease since gradient is positive
        REQUIRE(x->data[0] < initial_x);

        // With lr=0.1 and grad=20, step should be -2
        REQUIRE(approx_equal(x->data[0], initial_x - 0.1f * initial_grad));
    }

    SECTION("SGD with momentum accelerates convergence") {
        auto x = from_vector({5.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        SGD optimizer(params, 0.1f, 0.9f);  // lr=0.1, momentum=0.9

        // Take multiple steps
        for (int i = 0; i < 5; ++i) {
            x->zero_grad();
            auto loss = mul(x, x);
            auto total_loss = sum(loss);
            total_loss->backward();
            optimizer.step();
        }

        // Should have moved toward zero
        REQUIRE(std::abs(x->data[0]) < 5.0f);
    }

    SECTION("SGD zero_grad clears all parameter gradients") {
        auto x = from_vector({1.0f}, {1}, true);
        auto y = from_vector({2.0f}, {1}, true);

        x->grad[0] = 5.0f;
        y->grad[0] = 10.0f;

        std::vector<TensorPtr> params = {x, y};
        SGD optimizer(params, 0.1f);

        optimizer.zero_grad();

        REQUIRE(x->grad[0] == 0.0f);
        REQUIRE(y->grad[0] == 0.0f);
    }

    SECTION("SGD learning rate getter/setter") {
        auto x = from_vector({1.0f}, {1}, true);
        std::vector<TensorPtr> params = {x};

        SGD optimizer(params, 0.1f);

        REQUIRE(optimizer.get_lr() == 0.1f);

        optimizer.set_lr(0.01f);

        REQUIRE(optimizer.get_lr() == 0.01f);
    }
}

TEST_CASE("SGD weight decay", "[optim]") {
    SECTION("Weight decay adds L2 regularization") {
        auto x = from_vector({1.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        SGD optimizer(params, 0.1f, 0.0f, 0.01f);  // weight_decay=0.01

        // Compute gradient (zero gradient from loss)
        x->zero_grad();
        auto loss = zeros({1}, true);
        loss->backward();

        float initial_x = x->data[0];

        optimizer.step();

        // Even with zero gradient from loss, weight decay should shrink parameters
        // Update: x = x - lr * (grad + weight_decay * x)
        //       = 1.0 - 0.1 * (0 + 0.01 * 1.0)
        //       = 1.0 - 0.001 = 0.999
        REQUIRE(x->data[0] < initial_x);
    }
}

TEST_CASE("Adam optimizer", "[optim]") {
    SECTION("Adam updates parameters") {
        auto x = from_vector({10.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        Adam optimizer(params, 0.1f);

        // Compute gradient
        auto loss = mul(x, x);
        auto total_loss = sum(loss);
        total_loss->backward();

        float initial_x = x->data[0];

        optimizer.step();

        // x should decrease
        REQUIRE(x->data[0] < initial_x);
    }

    SECTION("Adam converges on simple quadratic") {
        auto x = from_vector({5.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        Adam optimizer(params, 0.1f);

        // Run optimization for multiple steps
        for (int i = 0; i < 20; ++i) {
            x->zero_grad();
            auto loss = mul(x, x);
            auto total_loss = sum(loss);
            total_loss->backward();
            optimizer.step();
        }

        // Should converge close to zero
        REQUIRE(std::abs(x->data[0]) < 1.0f);
    }

    SECTION("Adam with multiple parameters") {
        auto x = from_vector({5.0f, -3.0f}, {2}, true);

        std::vector<TensorPtr> params = {x};
        Adam optimizer(params, 0.1f);

        for (int i = 0; i < 20; ++i) {
            x->zero_grad();
            auto loss = mul(x, x);
            auto total_loss = sum(loss);
            total_loss->backward();
            optimizer.step();
        }

        // Both should converge toward zero
        REQUIRE(std::abs(x->data[0]) < 1.0f);
        REQUIRE(std::abs(x->data[1]) < 1.0f);
    }
}

TEST_CASE("AdamW optimizer", "[optim]") {
    SECTION("AdamW decoupled weight decay") {
        auto x = from_vector({1.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        AdamW optimizer(params, 0.1f, 0.9f, 0.999f, 1e-8f, 0.01f);

        // Even with zero gradient, weight decay should affect parameters
        x->zero_grad();
        auto loss = zeros({1}, true);
        loss->backward();

        float initial_x = x->data[0];

        optimizer.step();

        // Weight decay should reduce parameter
        REQUIRE(x->data[0] < initial_x);
    }

    SECTION("AdamW optimizes effectively") {
        auto x = from_vector({10.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        AdamW optimizer(params, 0.1f);

        for (int i = 0; i < 20; ++i) {
            x->zero_grad();
            auto loss = mul(x, x);
            auto total_loss = sum(loss);
            total_loss->backward();
            optimizer.step();
        }

        REQUIRE(std::abs(x->data[0]) < 2.0f);
    }
}

TEST_CASE("RMSprop optimizer", "[optim]") {
    SECTION("RMSprop updates parameters") {
        auto x = from_vector({10.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        RMSprop optimizer(params, 0.01f);

        // Compute gradient
        auto loss = mul(x, x);
        auto total_loss = sum(loss);
        total_loss->backward();

        float initial_x = x->data[0];

        optimizer.step();

        // x should decrease
        REQUIRE(x->data[0] < initial_x);
    }

    SECTION("RMSprop converges on quadratic") {
        auto x = from_vector({5.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        RMSprop optimizer(params, 0.1f);

        for (int i = 0; i < 30; ++i) {
            x->zero_grad();
            auto loss = mul(x, x);
            auto total_loss = sum(loss);
            total_loss->backward();
            optimizer.step();
        }

        REQUIRE(std::abs(x->data[0]) < 1.0f);
    }
}

TEST_CASE("Learning rate schedulers", "[optim]") {
    SECTION("ExponentialLR decays learning rate") {
        auto x = from_vector({1.0f}, {1}, true);
        std::vector<TensorPtr> params = {x};

        SGD optimizer(params, 1.0f);
        ExponentialLR scheduler(&optimizer, 0.9f);

        float initial_lr = optimizer.get_lr();
        REQUIRE(initial_lr == 1.0f);

        scheduler.step();
        REQUIRE(approx_equal(optimizer.get_lr(), 0.9f));

        scheduler.step();
        REQUIRE(approx_equal(optimizer.get_lr(), 0.81f));
    }

    SECTION("CosineAnnealingLR follows cosine schedule") {
        auto x = from_vector({1.0f}, {1}, true);
        std::vector<TensorPtr> params = {x};

        SGD optimizer(params, 1.0f);
        CosineAnnealingLR scheduler(&optimizer, 100, 0.0f);

        float initial_lr = optimizer.get_lr();
        REQUIRE(initial_lr == 1.0f);

        // Take steps and verify learning rate decreases
        scheduler.step();
        float lr_after_1 = optimizer.get_lr();
        REQUIRE(lr_after_1 < initial_lr);

        // After many steps, should approach eta_min (0.0)
        for (int i = 0; i < 99; ++i) {
            scheduler.step();
        }

        REQUIRE(optimizer.get_lr() < 0.1f);
    }
}

TEST_CASE("Optimizer convergence on different functions", "[optim]") {
    SECTION("Minimize Rosenbrock function with Adam") {
        // f(x, y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1, 1)
        auto params = from_vector({0.0f, 0.0f}, {2}, true);

        std::vector<TensorPtr> param_vec = {params};
        Adam optimizer(param_vec, 0.01f);

        for (int i = 0; i < 100; ++i) {
            params->zero_grad();

            auto x = zeros({1}, true);
            auto y = zeros({1}, true);
            x->data[0] = params->data[0];
            y->data[0] = params->data[1];

            // (1-x)^2
            auto one = ones({1}, false);
            auto one_minus_x = sub(one, x);
            auto term1 = mul(one_minus_x, one_minus_x);

            // (y - x^2)^2
            auto x_sq = mul(x, x);
            auto y_minus_x_sq = sub(y, x_sq);
            auto term2_inner = mul(y_minus_x_sq, y_minus_x_sq);
            auto term2 = mul(term2_inner, 100.0f);

            auto loss = add(term1, term2);

            loss->backward();

            // Manually copy gradients (since we split params)
            params->grad[0] = x->grad[0];
            params->grad[1] = y->grad[0];

            optimizer.step();
        }

        // Should be closer to (1, 1) than starting point
        REQUIRE(std::abs(params->data[0] - 1.0f) < 0.5f);
        REQUIRE(std::abs(params->data[1] - 1.0f) < 0.5f);
    }
}

TEST_CASE("Optimizer with multiple parameter tensors", "[optim]") {
    SECTION("Optimize two separate tensors") {
        auto x = from_vector({5.0f}, {1}, true);
        auto y = from_vector({-3.0f}, {1}, true);

        std::vector<TensorPtr> params = {x, y};
        Adam optimizer(params, 0.1f);

        for (int i = 0; i < 20; ++i) {
            optimizer.zero_grad();

            // Loss = x^2 + y^2
            auto x_sq = mul(x, x);
            auto y_sq = mul(y, y);
            auto loss = add(x_sq, y_sq);
            auto total = sum(loss);

            total->backward();
            optimizer.step();
        }

        // Both should converge to zero
        REQUIRE(std::abs(x->data[0]) < 0.5f);
        REQUIRE(std::abs(y->data[0]) < 0.5f);
    }
}

TEST_CASE("Optimizer numerical stability", "[optim]") {
    SECTION("Adam handles very small gradients") {
        auto x = from_vector({1.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        Adam optimizer(params, 0.001f);

        // Very small gradient
        x->zero_grad();
        x->grad[0] = 1e-8f;

        float initial_x = x->data[0];
        optimizer.step();

        // Should update slightly
        REQUIRE(x->data[0] != initial_x);
    }

    SECTION("Adam epsilon prevents division by zero") {
        auto x = from_vector({1.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        Adam optimizer(params, 0.1f, 0.9f, 0.999f, 1e-8f);

        // Zero gradient should not cause issues
        x->zero_grad();

        REQUIRE_NOTHROW(optimizer.step());
    }
}

TEST_CASE("Optimizer gradient accumulation", "[optim]") {
    SECTION("Multiple backward passes before step") {
        auto x = from_vector({1.0f}, {1}, true);

        std::vector<TensorPtr> params = {x};
        SGD optimizer(params, 0.1f, 0.0f);

        // Accumulate gradients from multiple losses
        for (int i = 0; i < 3; ++i) {
            auto loss = mul(x, x);
            auto total = sum(loss);
            total->backward();  // Accumulates gradients
        }

        // Gradient should be 3 * 2*x = 6*x
        REQUIRE(approx_equal(x->grad[0], 6.0f * x->data[0]));

        optimizer.step();
        optimizer.zero_grad();

        REQUIRE(x->grad[0] == 0.0f);
    }
}
