#ifndef TEST_INC
#define TEST_NAME formatter
#define TEST_INC "test_formatter.cpp"
#include "sycl_std_formatters.hpp"
#include "test_select_body.inc"
#else

// sycl::range<N>
RUN(PRINTLNF("range = {}", sycl::range<1>{4}));
RUN(PRINTLNF("range = {}", sycl::range<2>{4, 8}));
RUN(PRINTLNF("range = {}", sycl::range<3>{4, 8, 16}));

// sycl::id<N>
RUN(PRINTLNF("id = {}", sycl::id<1>{2}));
RUN(PRINTLNF("id = {}", sycl::id<2>{2, 5}));
RUN(PRINTLNF("id = {}", sycl::id<3>{1, 2, 3}));

// Mixed primitive + custom (auto-indexed)
RUN(PRINTLNF("step {} of {}: range={}", 3, 100, sycl::range<3>{4, 8, 16}));

// PRINTF (no newline) variant
RUN(PRINTF("[{}]\n", sycl::id<2>{0, 0}));

// All-primitive call through the formatter-aware macro path
RUN(PRINTLNF("plain {} works too", 42));

#endif
