#include "catch2.hpp"
#include "test_helpers.hpp"
#include "../include/autograd/tensor.hpp"
#include <cmath>

using namespace autograd;

TEST_CASE("Tensor creation and basic properties", "[tensor]") {
    SECTION("Create tensor with shape") {
        auto t = std::make_shared<Tensor>(std::initializer_list<size_t>{2, 3});

        REQUIRE(t->size() == 6);
        REQUIRE(t->ndim() == 2);
        REQUIRE(t->shape()[0] == 2);
        REQUIRE(t->shape()[1] == 3);
        REQUIRE(t->requires_grad == true);
        REQUIRE(t->is_leaf == true);
    }

    SECTION("Create tensor without gradient tracking") {
        auto t = std::make_shared<Tensor>(std::initializer_list<size_t>{3, 4}, false);

        REQUIRE(t->size() == 12);
        REQUIRE(t->requires_grad == false);
    }

    SECTION("Tensor initializes to zeros") {
        auto t = std::make_shared<Tensor>(std::initializer_list<size_t>{2, 2});

        for (size_t i = 0; i < t->size(); ++i) {
            REQUIRE(t->data[i] == 0.0f);
            REQUIRE(t->grad[i] == 0.0f);
        }
    }
}

TEST_CASE("Tensor factory functions", "[tensor]") {
    SECTION("zeros creates zero tensor") {
        auto t = zeros({3, 2});

        REQUIRE(t->size() == 6);
        for (size_t i = 0; i < t->size(); ++i) {
            REQUIRE(t->data[i] == 0.0f);
        }
    }

    SECTION("ones creates tensor filled with ones") {
        auto t = ones({2, 3});

        REQUIRE(t->size() == 6);
        for (size_t i = 0; i < t->size(); ++i) {
            REQUIRE(t->data[i] == 1.0f);
        }
    }

    SECTION("full creates tensor with specified value") {
        auto t = full({2, 2}, 5.0f);

        for (size_t i = 0; i < t->size(); ++i) {
            REQUIRE(t->data[i] == 5.0f);
        }
    }

    SECTION("from_vector creates tensor from std::vector") {
        std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
        auto t = from_vector(data, {2, 2});

        REQUIRE(t->size() == 4);
        REQUIRE(t->data[0] == 1.0f);
        REQUIRE(t->data[1] == 2.0f);
        REQUIRE(t->data[2] == 3.0f);
        REQUIRE(t->data[3] == 4.0f);
    }
}

TEST_CASE("Tensor data access", "[tensor]") {
    SECTION("Index operator access") {
        auto t = zeros({3});
        t->data[0] = 1.0f;
        t->data[1] = 2.0f;
        t->data[2] = 3.0f;

        REQUIRE((*t)[0] == 1.0f);
        REQUIRE((*t)[1] == 2.0f);
        REQUIRE((*t)[2] == 3.0f);
    }

    SECTION("item() returns scalar value") {
        auto t = zeros({1});
        t->data[0] = 42.0f;

        REQUIRE(t->item() == 42.0f);
    }

    SECTION("item() throws on non-scalar tensor") {
        auto t = zeros({2, 2});

        REQUIRE_THROWS_AS(t->item(), std::runtime_error);
    }

    SECTION("at() multi-dimensional indexing") {
        auto t = zeros({2, 3});
        t->data[0 * 3 + 1] = 5.0f;  // [0, 1] = 5.0

        REQUIRE(t->at({0, 1}) == 5.0f);
    }
}

TEST_CASE("Tensor operations", "[tensor]") {
    SECTION("fill sets all elements to value") {
        auto t = zeros({3, 2});
        t->fill(7.0f);

        for (size_t i = 0; i < t->size(); ++i) {
            REQUIRE(t->data[i] == 7.0f);
        }
    }

    SECTION("zero_grad clears gradients") {
        auto t = zeros({2, 2});
        t->grad[0] = 1.0f;
        t->grad[1] = 2.0f;

        t->zero_grad();

        for (size_t i = 0; i < t->size(); ++i) {
            REQUIRE(t->grad[i] == 0.0f);
        }
    }
}

TEST_CASE("Tensor view and reshape", "[tensor]") {
    SECTION("view reshapes tensor") {
        auto t = zeros({6});
        for (size_t i = 0; i < 6; ++i) {
            t->data[i] = static_cast<float>(i);
        }

        auto v = t->view({2, 3});

        REQUIRE(v->shape()[0] == 2);
        REQUIRE(v->shape()[1] == 3);
        REQUIRE(v->size() == 6);

        // Data should be same
        for (size_t i = 0; i < 6; ++i) {
            REQUIRE(v->data[i] == static_cast<float>(i));
        }
    }

    SECTION("view throws on size mismatch") {
        auto t = zeros({6});

        REQUIRE_THROWS_AS(t->view({2, 2}), std::runtime_error);
    }

    SECTION("reshape is alias for view") {
        auto t = zeros({4});
        auto r = t->reshape({2, 2});

        REQUIRE(r->shape()[0] == 2);
        REQUIRE(r->shape()[1] == 2);
    }
}

TEST_CASE("Tensor transpose", "[tensor]") {
    SECTION("transpose swaps dimensions") {
        std::vector<float> data = {1, 2, 3, 4, 5, 6};
        auto t = from_vector(data, {2, 3});

        auto transposed = t->transpose();

        REQUIRE(transposed->shape()[0] == 3);
        REQUIRE(transposed->shape()[1] == 2);

        // Check transposition correctness
        // Original:  [[1, 2, 3],
        //             [4, 5, 6]]
        // Transposed: [[1, 4],
        //              [2, 5],
        //              [3, 6]]
        REQUIRE(transposed->data[0] == 1);
        REQUIRE(transposed->data[1] == 4);
        REQUIRE(transposed->data[2] == 2);
        REQUIRE(transposed->data[3] == 5);
        REQUIRE(transposed->data[4] == 3);
        REQUIRE(transposed->data[5] == 6);
    }

    SECTION("transpose throws on non-2D tensor") {
        auto t = zeros({2, 2, 2});

        REQUIRE_THROWS_AS(t->transpose(), std::runtime_error);
    }
}

TEST_CASE("Tensor gradient tracking", "[tensor]") {
    SECTION("clone creates copy with gradient tracking") {
        std::vector<float> data = {1, 2, 3};
        auto t = from_vector(data, {3}, true);

        auto cloned = t->clone();

        REQUIRE(cloned->size() == 3);
        REQUIRE(cloned->data[0] == 1);
        REQUIRE(cloned->data[1] == 2);
        REQUIRE(cloned->data[2] == 3);
        REQUIRE(cloned->requires_grad == true);
    }

    SECTION("detach creates copy without gradient tracking") {
        std::vector<float> data = {1, 2, 3};
        auto t = from_vector(data, {3}, true);

        auto detached = t->detach();

        REQUIRE(detached->size() == 3);
        REQUIRE(detached->data[0] == 1);
        REQUIRE(detached->requires_grad == false);
    }
}

TEST_CASE("Tensor initialization methods", "[tensor]") {
    SECTION("randn creates normally distributed values") {
        auto t = zeros({1000});
        t->randn(0.0f, 1.0f);

        // Check mean is approximately 0
        float sum = 0;
        for (size_t i = 0; i < t->size(); ++i) {
            sum += t->data[i];
        }
        float mean = sum / t->size();

        REQUIRE(std::abs(mean) < 0.1f);  // Should be close to 0
    }

    SECTION("uniform creates uniformly distributed values") {
        auto t = zeros({100});
        t->uniform(0.0f, 1.0f);

        // All values should be in [0, 1]
        for (size_t i = 0; i < t->size(); ++i) {
            REQUIRE(t->data[i] >= 0.0f);
            REQUIRE(t->data[i] <= 1.0f);
        }
    }
}

TEST_CASE("Backward pass basic functionality", "[tensor][gradient]") {
    SECTION("backward throws on non-scalar tensor") {
        auto t = zeros({2, 2});

        REQUIRE_THROWS_AS(t->backward(), std::runtime_error);
    }

    SECTION("backward sets gradient to 1 for scalar") {
        auto t = zeros({1});
        t->backward();

        REQUIRE(t->grad[0] == 1.0f);
    }
}

TEST_CASE("Tensor strides calculation", "[tensor]") {
    SECTION("1D tensor strides") {
        auto t = zeros({5});
        const auto& strides = t->strides();

        REQUIRE(strides.size() == 1);
        REQUIRE(strides[0] == 1);
    }

    SECTION("2D tensor strides") {
        auto t = zeros({3, 4});
        const auto& strides = t->strides();

        REQUIRE(strides.size() == 2);
        REQUIRE(strides[0] == 4);
        REQUIRE(strides[1] == 1);
    }

    SECTION("3D tensor strides") {
        auto t = zeros({2, 3, 4});
        const auto& strides = t->strides();

        REQUIRE(strides.size() == 3);
        REQUIRE(strides[0] == 12);
        REQUIRE(strides[1] == 4);
        REQUIRE(strides[2] == 1);
    }
}
