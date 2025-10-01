cmake_minimum_required(VERSION 3.14)

# Simple gradients example
add_executable(simple_gradients simple_gradients.cpp)
target_link_libraries(simple_gradients autograd)
if(TARGET autograd_impl)
    target_link_libraries(simple_gradients autograd_impl)
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(simple_gradients OpenMP::OpenMP_CXX)
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
