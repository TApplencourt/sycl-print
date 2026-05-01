#ifndef TEST_INC
#define TEST_NAME formatter
#define TEST_INC "test_formatter.cpp"
#include "sycl_std_formatters.hpp"
#include "test_select_body.inc"
#else

// sycl::range<N>
RUN(PRINTLN("range = {}", sycl::range<1>{4}));
RUN(PRINTLN("range = {}", sycl::range<2>{4, 8}));
RUN(PRINTLN("range = {}", sycl::range<3>{4, 8, 16}));

// sycl::id<N>
RUN(PRINTLN("id = {}", sycl::id<1>{2}));
RUN(PRINTLN("id = {}", sycl::id<2>{2, 5}));
RUN(PRINTLN("id = {}", sycl::id<3>{1, 2, 3}));

// Mixed primitive + custom (auto-indexed)
RUN(PRINTLN("step {} of {}: range={}", 3, 100, sycl::range<3>{4, 8, 16}));

// PRINT (no newline) variant
RUN(PRINT("[{}]\n", sycl::id<2>{0, 0}));

// Pure primitives still go through unchanged path
RUN(PRINTLN("plain {} works too", 42));

#endif
