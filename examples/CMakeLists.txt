cmake_minimum_required(VERSION 3.14)

# Neural network examples
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nn")
    # RNN sine wave prediction
    add_executable(rnn_sine_wave nn/rnn_sine_wave.cpp)
    target_link_libraries(rnn_sine_wave autograd)
    if(TARGET autograd_impl)
        target_link_libraries(rnn_sine_wave autograd_impl)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rnn_sine_wave OpenMP::OpenMP_CXX)
    endif()
    
    # RNN text generation
    add_executable(rnn_text_generation nn/rnn_text_generation.cpp)
    target_link_libraries(rnn_text_generation autograd)
    if(TARGET autograd_impl)
        target_link_libraries(rnn_text_generation autograd_impl)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rnn_text_generation OpenMP::OpenMP_CXX)
    endif()
    
    # RNN adding problem
    add_executable(rnn_adding_problem nn/rnn_adding_problem.cpp)
    target_link_libraries(rnn_adding_problem autograd)
    if(TARGET autograd_impl)
        target_link_libraries(rnn_adding_problem autograd_impl)
    endif()
    if(OpenMP_CXX_FOUND)
        target_link_libraries(rnn_adding_problem OpenMP::OpenMP_CXX)
    endif()
endif()

# Simple gradients example
add_executable(simple_gradients simple_gradients.cpp)
target_link_libraries(simple_gradients autograd)
if(TARGET autograd_impl)
    target_link_libraries(simple_gradients autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(simple_gradients OpenMP::OpenMP_CXX)
endif()

# Exponential MLE example
add_executable(exponential_mle exponential_mle.cpp)
target_link_libraries(exponential_mle autograd)
if(TARGET autograd_impl)
    target_link_libraries(exponential_mle autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(exponential_mle OpenMP::OpenMP_CXX)
endif()

# Simple Weibull MLE example
add_executable(weibull_mle_simple weibull_mle_simple.cpp)
target_link_libraries(weibull_mle_simple autograd)
if(TARGET autograd_impl)
    target_link_libraries(weibull_mle_simple autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(weibull_mle_simple OpenMP::OpenMP_CXX)
endif()

# Exponential bootstrap example
add_executable(exponential_bootstrap exponential_bootstrap.cpp)
target_link_libraries(exponential_bootstrap autograd)
if(TARGET autograd_impl)
    target_link_libraries(exponential_bootstrap autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(exponential_bootstrap OpenMP::OpenMP_CXX)
endif()

# Linear regression example
add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression autograd)
if(TARGET autograd_impl)
    target_link_libraries(linear_regression autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(linear_regression OpenMP::OpenMP_CXX)
endif()

# Logistic regression example
add_executable(logistic_regression logistic_regression.cpp)
target_link_libraries(logistic_regression autograd)
if(TARGET autograd_impl)
    target_link_libraries(logistic_regression autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(logistic_regression OpenMP::OpenMP_CXX)
endif()

# Hessian demonstration example
add_executable(hessian_demo hessian_demo.cpp)
target_link_libraries(hessian_demo autograd)
if(TARGET autograd_impl)
    target_link_libraries(hessian_demo autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(hessian_demo OpenMP::OpenMP_CXX)
endif()

# Simple Hessian demonstration example
add_executable(hessian_simple hessian_simple.cpp)
target_link_libraries(hessian_simple autograd)
if(TARGET autograd_impl)
    target_link_libraries(hessian_simple autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(hessian_simple OpenMP::OpenMP_CXX)
endif()