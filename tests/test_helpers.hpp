#pragma once
#include <cmath>

// Helper function to check if two floats are approximately equal
inline bool approx_equal(float a, float b, float epsilon = 1e-5f) {
    return std::abs(a - b) < epsilon;
}
