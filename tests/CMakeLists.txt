# Test configuration for autograd-cpp

# Add test executable
add_executable(autograd_tests
    test_main.cpp
    test_tensor.cpp
    test_ops.cpp
    test_jacobian.cpp
    test_hessian.cpp
    test_optim.cpp
)

# Link against autograd library
target_link_libraries(autograd_tests PRIVATE autograd)

# Link against implementation library if it exists
if(TARGET autograd_impl)
    target_link_libraries(autograd_tests PRIVATE autograd_impl)
endif()

# Include directories
target_include_directories(autograd_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Enable C++17
target_compile_features(autograd_tests PRIVATE cxx_std_17)

# Add compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(autograd_tests PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# Register tests with CTest
add_test(NAME tensor_tests COMMAND autograd_tests "[tensor]")
add_test(NAME ops_tests COMMAND autograd_tests "[ops]")
add_test(NAME jacobian_tests COMMAND autograd_tests "[jacobian]")
add_test(NAME hessian_tests COMMAND autograd_tests "[hessian]")
add_test(NAME optim_tests COMMAND autograd_tests "[optim]")
add_test(NAME all_tests COMMAND autograd_tests)

# Set test properties
set_tests_properties(all_tests PROPERTIES
    TIMEOUT 300
)

# Enable code coverage if requested
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(autograd_tests PRIVATE --coverage)
        target_link_options(autograd_tests PRIVATE --coverage)
    endif()
endif()

# Print test configuration
message(STATUS "")
message(STATUS "Test configuration:")
message(STATUS "  Test executable: autograd_tests")
message(STATUS "  Test files:")
message(STATUS "    - test_tensor.cpp")
message(STATUS "    - test_ops.cpp")
message(STATUS "    - test_jacobian.cpp")
message(STATUS "    - test_hessian.cpp")
message(STATUS "    - test_optim.cpp")
message(STATUS "")
